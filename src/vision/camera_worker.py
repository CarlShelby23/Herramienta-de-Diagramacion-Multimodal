from __future__ import annotations

import threading
import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from src.config import CalibrationConfig
from src.models.events import GestureEventType, GestureFrame, GestureState
from src.vision.filters import CursorSmoother2D, DwellStateMachine
from src.vision.gesture_classifier import GestureClassifier
from src.vision.hand_tracker import HandTracker


class CameraVisionWorker(QThread):
    """Hilo dedicado de captura de cámara + MediaPipe + máquina Dwell Time.

    Desacopla totalmente OpenCV/MediaPipe del hilo de UI (PySide6) y garantiza
    liberación limpia de recursos al detenerse.
    """

    gesture_updated = Signal(object)   # GestureFrame
    frame_ready = Signal(QImage)       # Para vista previa en vivo en WF-01
    status_changed = Signal(str, bool) # (mensaje, camera_ok)

    def __init__(
        self,
        calibration: CalibrationConfig,
        canvas_width: int = 1280,
        canvas_height: int = 720,
        emit_preview: bool = False,
        hand_grace_period_ms: int = 300,
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._calibration = calibration
        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._emit_preview = emit_preview
        self._running = False

        self._classifier = GestureClassifier(calibration)
        self._dwell = DwellStateMachine(
            dwell_time_ms=calibration.dwell_time_ms,
            hand_grace_period_ms=hand_grace_period_ms,
        )
        self._smoother = CursorSmoother2D()

    def set_canvas_size(self, width: int, height: int) -> None:
        with self._lock:
            self._canvas_width = max(320, int(width))
            self._canvas_height = max(240, int(height))

    def set_emit_preview(self, enabled: bool) -> None:
        with self._lock:
            self._emit_preview = enabled

    def update_calibration(self, calibration: CalibrationConfig) -> None:
        with self._lock:
            self._calibration = calibration
            self._classifier.update_calibration(calibration)
            self._dwell.update_dwell_time(calibration.dwell_time_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        self._running = True
        tracker = HandTracker(model_path=self._calibration.hand_landmarker_model_path)
        if not tracker.is_ready:
            self.status_changed.emit(
                tracker.error_message or "MediaPipe no disponible", False
            )

        cap = cv2.VideoCapture(self._calibration.camera_index)
        if not cap.isOpened():
            self.status_changed.emit("No se detecta cámara web activa", False)
            self.gesture_updated.emit(
                GestureFrame(
                    timestamp_ms=int(time.time() * 1000),
                    raw_gesture=GestureState.HAND_LOST,
                    stable_gesture=GestureState.HAND_LOST,
                    event=GestureEventType.HAND_LOST,
                    camera_ok=False,
                )
            )
            tracker.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.status_changed.emit("Cámara activa", True)

        frame_counter = 0
        fps_window_start = time.monotonic()
        current_fps = 0.0
        consecutive_failures = 0

        try:
            while self._running:
                ok, frame_bgr = cap.read()
                now_ms = int(time.time() * 1000)
                if not ok or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        self.status_changed.emit("Señal de cámara perdida", False)
                        self.gesture_updated.emit(
                            GestureFrame(
                                timestamp_ms=now_ms,
                                raw_gesture=GestureState.HAND_LOST,
                                stable_gesture=GestureState.HAND_LOST,
                                event=GestureEventType.HAND_LOST,
                                camera_ok=False,
                            )
                        )
                    self.msleep(30)
                    continue

                if consecutive_failures >= 10:
                    self.status_changed.emit("Cámara recuperada", True)
                consecutive_failures = 0

                frame_counter += 1
                elapsed_fps = time.monotonic() - fps_window_start
                if elapsed_fps >= 0.5:
                    current_fps = frame_counter / elapsed_fps
                    frame_counter = 0
                    fps_window_start = time.monotonic()

                frame_bgr = cv2.flip(frame_bgr, 1)
                h, w = frame_bgr.shape[:2]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                landmarks = tracker.detect(frame_rgb) if tracker.is_ready else []

                with self._lock:
                    canvas_w = self._canvas_width
                    canvas_h = self._canvas_height
                    emit_preview = self._emit_preview

                    if len(landmarks) == 21:
                        cls = self._classifier.classify(landmarks, w, h)
                        stable, event, progress = self._dwell.update(cls.raw_gesture, now_ms)
                        raw_cx = cls.cursor_norm_x * canvas_w
                        raw_cy = cls.cursor_norm_y * canvas_h
                        smooth_x, smooth_y = self._smoother.filter(raw_cx, raw_cy, now_ms)

                        packet = GestureFrame(
                            timestamp_ms=now_ms,
                            raw_gesture=cls.raw_gesture,
                            stable_gesture=stable,
                            event=event,
                            dwell_progress=progress,
                            cursor_norm_x=cls.cursor_norm_x,
                            cursor_norm_y=cls.cursor_norm_y,
                            cursor_x=smooth_x,
                            cursor_y=smooth_y,
                            pinch_distance_px=cls.pinch_distance_px,
                            effective_threshold_px=cls.effective_threshold_px,
                            hand_scale_px=cls.hand_scale_px,
                            hand_detected=True,
                            landmarks_count=21,
                            fps=current_fps,
                            camera_ok=True,
                        )
                    else:
                        stable, event, progress = self._dwell.update(
                            GestureState.HAND_LOST, now_ms
                        )
                        if stable == GestureState.HAND_LOST:
                            self._smoother.reset()
                            self._classifier.reset()
                        packet = GestureFrame(
                            timestamp_ms=now_ms,
                            raw_gesture=GestureState.HAND_LOST,
                            stable_gesture=stable,
                            event=event,
                            dwell_progress=progress,
                            hand_detected=False,
                            landmarks_count=0,
                            fps=current_fps,
                            camera_ok=True,
                        )

                self.gesture_updated.emit(packet)

                if emit_preview:
                    preview_bgr = frame_bgr.copy()
                    if len(landmarks) == 21:
                        HandTracker.draw_landmarks(preview_bgr, landmarks)
                    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
                    qimg = QImage(
                        preview_rgb.data,
                        preview_rgb.shape[1],
                        preview_rgb.shape[0],
                        preview_rgb.strides[0],
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(qimg)
        finally:
            cap.release()
            tracker.close()
