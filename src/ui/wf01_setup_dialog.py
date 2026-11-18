from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from src.audio.asr_worker import build_asr_engine
from src.audio.audio_monitor import AudioLevelMonitor
from src.config import CalibrationConfig, ConditionLevel, SessionConfig
from src.models.events import GestureFrame
from src.vision.camera_worker import CameraVisionWorker


class WF01SetupDialog(QDialog):
    """WF-01 — Configuración Pre-Sesión, Checklist de Periféricos y Calibración Dinámica."""

    def __init__(self, start_hardware: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("WF-01 | Configuración Pre-Sesión y Calibración")
        self.setMinimumSize(940, 650)

        self._calibration = CalibrationConfig()
        self._session_config: SessionConfig | None = None

        self._camera_ok = False
        self._landmarks_ok = False
        self._mic_ok = False
        self._asr_ok = False
        self._latest_pinch_px: float | None = None
        self._latest_hand_scale_px: float | None = None

        self._camera_worker: CameraVisionWorker | None = None
        self._audio_monitor: AudioLevelMonitor | None = None

        self._build_ui()
        self._check_asr_model()

        if start_hardware:
            self._start_hardware_checks()

    @property
    def session_config(self) -> SessionConfig | None:
        return self._session_config

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 18, 20, 18)
        root_layout.setSpacing(14)

        # Encabezado
        title = QLabel("Configuración Pre-Sesión y Calibración (WF-01)")
        title.setObjectName("title")
        subtitle = QLabel(
            "Anonimización del participante, verificación en vivo de periféricos "
            "y calibración dinámica de pellizco y Dwell Time."
        )
        subtitle.setObjectName("subtitle")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        # Cuerpo en 2 columnas
        columns = QHBoxLayout()
        columns.setSpacing(16)
        root_layout.addLayout(columns, stretch=1)

        # --- Columna Izquierda: Datos de Sesión y Calibración ---
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        columns.addLayout(left_col, stretch=5)

        # Tarjeta 1: Participante y Condición
        card_session = QFrame()
        card_session.setObjectName("card")
        lay_s = QVBoxLayout(card_session)
        lay_s.setContentsMargins(16, 14, 16, 14)
        lay_s.setSpacing(10)

        lbl_s1 = QLabel("1. Datos de Sesión Experimental")
        lbl_s1.setObjectName("sectionHeader")
        lay_s.addWidget(lbl_s1)

        lay_s.addWidget(QLabel("Código anónimo de participante:"))
        self.input_participant = QLineEdit("PART-2026-001")
        self.input_participant.setPlaceholderText("Ej. PART-2026-014")
        lay_s.addWidget(self.input_participant)

        lay_s.addWidget(QLabel("Condición / complejidad de tarea:"))
        self._condition_group = QButtonGroup(self)
        cond_grid = QGridLayout()
        cond_grid.setSpacing(8)
        self._condition_radios: list[tuple[QRadioButton, ConditionLevel]] = []
        for idx, level in enumerate(ConditionLevel):
            rb = QRadioButton(level.value)
            if level == ConditionLevel.MEDIUM:
                rb.setChecked(True)
            self._condition_group.addButton(rb, idx)
            self._condition_radios.append((rb, level))
            cond_grid.addWidget(rb, idx // 2, idx % 2)
        lay_s.addLayout(cond_grid)
        left_col.addWidget(card_session)

        # Tarjeta 2: Calibración Dinámica
        card_cal = QFrame()
        card_cal.setObjectName("card")
        lay_c = QVBoxLayout(card_cal)
        lay_c.setContentsMargins(16, 14, 16, 14)
        lay_c.setSpacing(10)

        lbl_c = QLabel("2. Calibración Dinámica (Anti Falsos Positivos)")
        lbl_c.setObjectName("sectionHeader")
        lay_c.addWidget(lbl_c)

        pinch_row = QHBoxLayout()
        self.lbl_pinch_live = QLabel("Distancia pulgar-índice actual: -- px")
        self.btn_capture_pinch = QPushButton("Capturar Pellizco Actual")
        self.btn_capture_pinch.clicked.connect(self._capture_current_pinch)
        pinch_row.addWidget(self.lbl_pinch_live, stretch=1)
        pinch_row.addWidget(self.btn_capture_pinch)
        lay_c.addLayout(pinch_row)

        thresh_row = QHBoxLayout()
        self.lbl_threshold_val = QLabel("Umbral de pellizco: < 30 px")
        self.slider_pinch = QSlider(Qt.Orientation.Horizontal)
        self.slider_pinch.setRange(15, 70)
        self.slider_pinch.setValue(int(self._calibration.pinch_threshold_px))
        self.slider_pinch.valueChanged.connect(self._on_pinch_slider_changed)
        thresh_row.addWidget(self.lbl_threshold_val)
        thresh_row.addWidget(self.slider_pinch, stretch=1)
        lay_c.addLayout(thresh_row)

        dwell_row = QHBoxLayout()
        dwell_row.addWidget(QLabel("Dwell Time (confirmación gestual):"))
        self.spin_dwell = QSpinBox()
        self.spin_dwell.setRange(150, 600)
        self.spin_dwell.setSingleStep(25)
        self.spin_dwell.setSuffix(" ms")
        self.spin_dwell.setValue(self._calibration.dwell_time_ms)
        self.spin_dwell.valueChanged.connect(self._on_dwell_changed)
        dwell_row.addWidget(self.spin_dwell)
        dwell_row.addStretch()
        lay_c.addLayout(dwell_row)

        self.chk_simulation = QCheckBox(
            "Modo simulación / contingencia (permite ratón y consola de comandos si falta cámara o micrófono)"
        )
        self.chk_simulation.toggled.connect(self._update_start_button_state)
        lay_c.addWidget(self.chk_simulation)

        left_col.addWidget(card_cal)
        left_col.addStretch()

        # --- Columna Derecha: Checklist en Vivo de Periféricos ---
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        columns.addLayout(right_col, stretch=6)

        card_hw = QFrame()
        card_hw.setObjectName("card")
        lay_hw = QVBoxLayout(card_hw)
        lay_hw.setContentsMargins(16, 14, 16, 14)
        lay_hw.setSpacing(10)

        lbl_hw = QLabel("3. Checklist en Vivo de Periféricos")
        lbl_hw.setObjectName("sectionHeader")
        lay_hw.addWidget(lbl_hw)

        # Vista previa de cámara + 21 landmarks
        self.lbl_camera_preview = QLabel("Iniciando cámara y MediaPipe...")
        self.lbl_camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera_preview.setMinimumSize(400, 260)
        self.lbl_camera_preview.setStyleSheet(
            "background-color: #0F172A; color: #CBD5E1; border-radius: 8px;"
        )
        lay_hw.addWidget(self.lbl_camera_preview)

        # Badges de cámara / FPS / landmarks
        cam_status_row = QHBoxLayout()
        self.lbl_cam_status = QLabel("● Cámara: Verificando...")
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_landmarks = QLabel("Landmarks: 0/21")
        cam_status_row.addWidget(self.lbl_cam_status, stretch=2)
        cam_status_row.addWidget(self.lbl_fps, stretch=1)
        cam_status_row.addWidget(self.lbl_landmarks, stretch=1)
        lay_hw.addLayout(cam_status_row)

        # Separador visual
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #E2E8F0;")
        lay_hw.addWidget(sep)

        # Micrófono y ruido base en dB
        mic_row = QHBoxLayout()
        self.lbl_mic_status = QLabel("● Micrófono: Verificando...")
        self.lbl_noise_db = QLabel("Ruido base: -- dB")
        mic_row.addWidget(self.lbl_mic_status, stretch=2)
        mic_row.addWidget(self.lbl_noise_db, stretch=1)
        lay_hw.addLayout(mic_row)

        self.bar_mic_db = QProgressBar()
        self.bar_mic_db.setRange(0, 80)  # Mapea -80..0 dB a 0..80
        self.bar_mic_db.setValue(0)
        self.bar_mic_db.setFormat("Nivel actual: -80 dB")
        lay_hw.addWidget(self.bar_mic_db)

        self.lbl_asr_status = QLabel("● Motor ASR offline: Verificando...")
        self.lbl_asr_status.setWordWrap(True)
        lay_hw.addWidget(self.lbl_asr_status)

        right_col.addWidget(card_hw)

        # Barra inferior de acciones
        footer = QHBoxLayout()
        self.lbl_readiness = QLabel("Verifica periféricos o activa modo simulación para continuar.")
        self.lbl_readiness.setObjectName("subtitle")
        self.btn_start = QPushButton("Iniciar Sesión Multimodal →")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.clicked.connect(self._on_start_clicked)
        footer.addWidget(self.lbl_readiness, stretch=1)
        footer.addWidget(self.btn_start)
        root_layout.addLayout(footer)

        self._update_start_button_state()

    def _check_asr_model(self) -> None:
        engine, msg = build_asr_engine(self._calibration)
        self._asr_ok = engine is not None
        if self._asr_ok:
            self.lbl_asr_status.setText(f"● Motor ASR: {msg}")
            self.lbl_asr_status.setStyleSheet("color: #15803D; font-weight: 600;")
        else:
            self.lbl_asr_status.setText(f"● Motor ASR: {msg}")
            self.lbl_asr_status.setStyleSheet("color: #B45309; font-weight: 600;")

    def _start_hardware_checks(self) -> None:
        self._camera_worker = CameraVisionWorker(
            calibration=self._calibration,
            canvas_width=1280,
            canvas_height=720,
            emit_preview=True,
        )
        self._camera_worker.frame_ready.connect(self._on_camera_frame)
        self._camera_worker.gesture_updated.connect(self._on_gesture_update)
        self._camera_worker.status_changed.connect(self._on_camera_status)
        self._camera_worker.start()

        self._audio_monitor = AudioLevelMonitor(
            device_index=self._calibration.mic_device_index
        )
        self._audio_monitor.level_updated.connect(self._on_audio_level)
        self._audio_monitor.status_changed.connect(self._on_mic_status)
        self._audio_monitor.start()

    def _stop_hardware_checks(self) -> None:
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None
        if self._audio_monitor is not None:
            self._audio_monitor.stop()
            self._audio_monitor = None

    def _on_camera_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.lbl_camera_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.lbl_camera_preview.setPixmap(pixmap)

    def _on_camera_status(self, message: str, ok: bool) -> None:
        self._camera_ok = ok
        color = "#15803D" if ok else "#DC2626"
        self.lbl_cam_status.setText(f"● Cámara: {message}")
        self.lbl_cam_status.setStyleSheet(f"color: {color}; font-weight: 600;")
        if not ok:
            self.lbl_camera_preview.setText(f"Cámara no disponible\n({message})")
        self._update_start_button_state()

    def _on_gesture_update(self, frame: GestureFrame) -> None:
        self.lbl_fps.setText(f"FPS: {frame.fps:.1f}")
        self._landmarks_ok = frame.landmarks_count == 21
        lm_color = "#15803D" if self._landmarks_ok else "#64748B"
        self.lbl_landmarks.setText(f"Landmarks: {frame.landmarks_count}/21")
        self.lbl_landmarks.setStyleSheet(f"color: {lm_color}; font-weight: 600;")

        if frame.pinch_distance_px is not None:
            self._latest_pinch_px = frame.pinch_distance_px
            self._latest_hand_scale_px = frame.hand_scale_px
            self.lbl_pinch_live.setText(
                f"Distancia pulgar-índice actual: {frame.pinch_distance_px:.1f} px "
                f"(gesto: {frame.stable_gesture.value})"
            )
        else:
            self.lbl_pinch_live.setText("Distancia pulgar-índice actual: muestra tu mano")

    def _on_mic_status(self, message: str, ok: bool) -> None:
        self._mic_ok = ok
        color = "#15803D" if ok else "#DC2626"
        self.lbl_mic_status.setText(f"● Micrófono: {message}")
        self.lbl_mic_status.setStyleSheet(f"color: {color}; font-weight: 600;")
        self._update_start_button_state()

    def _on_audio_level(self, current_db: float, baseline_db: float) -> None:
        self._calibration.ambient_noise_db = baseline_db
        self.lbl_noise_db.setText(f"Ruido base: {baseline_db:.1f} dB")
        bar_val = max(0, min(80, int(current_db + 80.0)))
        self.bar_mic_db.setValue(bar_val)
        self.bar_mic_db.setFormat(f"Nivel actual: {current_db:.1f} dB")

    def _on_pinch_slider_changed(self, value: int) -> None:
        self._calibration.pinch_threshold_px = float(value)
        self.lbl_threshold_val.setText(f"Umbral de pellizco: < {value} px")
        if self._camera_worker is not None:
            self._camera_worker.update_calibration(self._calibration)

    def _on_dwell_changed(self, value: int) -> None:
        self._calibration.dwell_time_ms = int(value)
        if self._camera_worker is not None:
            self._camera_worker.update_calibration(self._calibration)

    def _capture_current_pinch(self) -> None:
        if self._latest_pinch_px is None:
            QMessageBox.information(
                self,
                "Calibración de pellizco",
                "Muestra tu mano haciendo un pellizco frente a la cámara antes de capturar.",
            )
            return
        # Margen de +25% sobre la distancia medida para garantizar detección cómoda sin falsos positivos
        calibrated = int(max(18.0, min(68.0, self._latest_pinch_px * 1.25 + 4.0)))
        if self._latest_hand_scale_px is not None:
            self._calibration.hand_scale_ref_px = self._latest_hand_scale_px
        self.slider_pinch.setValue(calibrated)

    def _update_start_button_state(self) -> None:
        sim = self.chk_simulation.isChecked()
        ready = sim or (self._camera_ok and self._mic_ok)
        self.btn_start.setEnabled(ready)
        if sim:
            self.lbl_readiness.setText("Modo simulación activo: listo para iniciar WF-02.")
        elif ready:
            self.lbl_readiness.setText("Periféricos verificados: listo para iniciar sesión.")
        else:
            self.lbl_readiness.setText(
                "Esperando cámara y micrófono (o activa modo simulación para probar sin hardware)."
            )

    def _selected_condition(self) -> ConditionLevel:
        for rb, level in self._condition_radios:
            if rb.isChecked():
                return level
        return ConditionLevel.MEDIUM

    def _on_start_clicked(self) -> None:
        code = self.input_participant.text().strip()
        if len(code) < 4:
            QMessageBox.warning(
                self,
                "Código de participante inválido",
                "Ingresa un código anónimo válido (ej. PART-2026-014).",
            )
            return

        self._calibration.simulation_mode = self.chk_simulation.isChecked()
        if self._calibration.simulation_mode:
            self._calibration.asr_engine = "simulated"

        self._session_config = SessionConfig(
            participant_code=code,
            condition=self._selected_condition(),
            calibration=self._calibration,
            output_dir=Path("output"),
        )
        self._stop_hardware_checks()
        self.accept()

    def closeEvent(self, event) -> None:
        self._stop_hardware_checks()
        super().closeEvent(event)
