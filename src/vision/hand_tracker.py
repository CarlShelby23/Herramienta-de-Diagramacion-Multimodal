from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.vision.gesture_classifier import Landmark2D

# Conexiones estándar de los 21 landmarks de mano (MediaPipe Hands)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    """Wrapper dual compatible con MediaPipe 0.10.x (solutions.hands) y 1.x (tasks.vision).

    Detecta los 21 landmarks de una mano y dibuja el esqueleto sobre el fotograma de
    vista previa en WF-01 sin depender de `cv2.imshow`.
    """

    def __init__(
        self,
        model_path: str | Path = "models/hand_landmarker.task",
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
    ) -> None:
        self.backend_name: str = "unavailable"
        self.error_message: str | None = None
        self._solutions_hands: Any = None
        self._tasks_landmarker: Any = None
        self._mp: Any = None

        try:
            import mediapipe as mp

            self._mp = mp
        except Exception as exc:
            self.error_message = f"No se pudo importar MediaPipe: {exc}"
            return

        # Ruta 1: MediaPipe 0.10.x (mp.solutions.hands)
        if hasattr(self._mp, "solutions") and hasattr(self._mp.solutions, "hands"):
            try:
                self._solutions_hands = self._mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.backend_name = "mediapipe.solutions.hands"
                return
            except Exception as exc:
                self.error_message = f"Error inicializando solutions.hands: {exc}"

        # Ruta 2: MediaPipe 1.x / Tasks API (HandLandmarker)
        if hasattr(self._mp, "tasks"):
            resolved = Path(model_path)
            if not resolved.exists():
                self.error_message = (
                    f"Falta el modelo '{resolved}'. Ejecuta: python scripts/download_models.py"
                )
                return
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(resolved)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._tasks_landmarker = vision.HandLandmarker.create_from_options(options)
                self.backend_name = "mediapipe.tasks.HandLandmarker"
            except Exception as exc:
                self.error_message = f"Error inicializando HandLandmarker: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._solutions_hands is not None or self._tasks_landmarker is not None

    def detect(self, frame_rgb: np.ndarray) -> list[Landmark2D]:
        if self._solutions_hands is not None:
            result = self._solutions_hands.process(frame_rgb)
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                return [
                    Landmark2D(float(p.x), float(p.y), float(p.z))
                    for p in hand.landmark
                ]
            return []

        if self._tasks_landmarker is not None:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame_rgb),
            )
            result = self._tasks_landmarker.detect(mp_image)
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                return [
                    Landmark2D(float(p.x), float(p.y), float(p.z))
                    for p in hand
                ]
            return []

        return []

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, landmarks: list[Landmark2D]) -> np.ndarray:
        if len(landmarks) < 21:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[start_idx], pts[end_idx], (59, 130, 246), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(pts):
            color = (124, 58, 237) if idx in (4, 8) else (16, 185, 129)
            radius = 5 if idx in (4, 8) else 3
            cv2.circle(frame_bgr, (x, y), radius, color, -1, cv2.LINE_AA)
        return frame_bgr

    def close(self) -> None:
        if self._solutions_hands is not None:
            try:
                self._solutions_hands.close()
            except Exception:
                pass
            self._solutions_hands = None
        if self._tasks_landmarker is not None:
            try:
                self._tasks_landmarker.close()
            except Exception:
                pass
            self._tasks_landmarker = None
