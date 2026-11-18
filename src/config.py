from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ConditionLevel(str, Enum):
    """Condiciones experimentales / niveles de complejidad de WF-01."""

    FAMILIARIZATION = "0. Familiarización"
    LOW = "1. Baja (Guiada)"
    MEDIUM = "2. Media (Parcial)"
    HIGH = "3. Alta (Abierta)"


TASK_INSTRUCTIONS: dict[ConditionLevel, str] = {
    ConditionLevel.FAMILIARIZATION: (
        "Fase 0 (Familiarización): Practica decir 'Crear nodo [concepto]', "
        "confirmar con mano abierta, mover con puño y enlazar con 'Conectar'."
    ),
    ConditionLevel.LOW: (
        "Fase 1 (Baja - Guiada): Crea los nodos 'Entrada', 'Proceso' y 'Salida', "
        "organízalos en secuencia y conéctalos explícitamente."
    ),
    ConditionLevel.MEDIUM: (
        "Fase 2 (Media - Parcial): Construye un diagrama de al menos 4 conceptos "
        "relacionados sobre tu rutina diaria de estudio y enlaza sus dependencias."
    ),
    ConditionLevel.HIGH: (
        "Fase 3 (Alta - Abierta): Diseña libremente un mapa conceptual completo "
        "sobre un proyecto propio usando voz y gestos."
    ),
}

# Paleta normativa para el cursor virtual circular cinemático
CURSOR_COLORS: dict[str, str] = {
    "MOVE": "#2563EB",          # Azul: dedo índice extendido -> Mover cursor
    "PINCH_SELECT": "#7C3AED",  # Morado: pellizco < umbral 250 ms -> Seleccionar / clic
    "FIST_DRAG": "#EA580C",     # Naranja: puño cerrado -> Arrastrar nodo
    "OPEN_DROP": "#16A34A",     # Verde: mano abierta -> Soltar / confirmar drop
    "IDLE": "#64748B",          # Gris neutro: postura transicional
    "HAND_LOST": "#94A3B8",     # Gris claro: mano fuera de cámara
}

CURSOR_LEGEND: list[tuple[str, str, str]] = [
    ("Azul", "#2563EB", "Dedo índice extendido — Mover cursor virtual"),
    ("Morado", "#7C3AED", "Pellizco pulgar-índice (< umbral, 250 ms) — Seleccionar / clic"),
    ("Naranja", "#EA580C", "Puño cerrado (250 ms) — Arrastrar nodo seleccionado"),
    ("Verde", "#16A34A", "Mano abierta (250 ms) — Soltar / confirmar acción (drop)"),
]


@dataclass
class CalibrationConfig:
    """Parámetros calibrables en WF-01."""

    pinch_threshold_px: float = 30.0
    hand_scale_ref_px: float = 110.0
    dwell_time_ms: int = 250
    pinch_hysteresis_factor: float = 1.35
    camera_index: int = 0
    mic_device_index: int | None = None
    ambient_noise_db: float = -45.0
    simulation_mode: bool = False
    asr_engine: str = "vosk"  # "vosk" | "faster-whisper" | "simulated"
    vosk_model_path: str = "models/vosk-model-small-es-0.42"
    hand_landmarker_model_path: str = "models/hand_landmarker.task"

    def to_dict(self) -> dict:
        return {
            "pinch_threshold_px": round(self.pinch_threshold_px, 2),
            "hand_scale_ref_px": round(self.hand_scale_ref_px, 2),
            "dwell_time_ms": self.dwell_time_ms,
            "pinch_hysteresis_factor": self.pinch_hysteresis_factor,
            "camera_index": self.camera_index,
            "mic_device_index": self.mic_device_index,
            "ambient_noise_db": round(self.ambient_noise_db, 2),
            "simulation_mode": self.simulation_mode,
            "asr_engine": self.asr_engine,
        }


@dataclass
class FusionConfig:
    """Ventanas temporales y umbrales de coordinación multimodal."""

    cursor_recent_window_ms: int = 1500
    create_confirm_window_ms: int = 6000
    connect_target_window_ms: int = 10000
    hand_grace_period_ms: int = 300
    hand_lost_timeout_ms: int = 1000
    inactivity_threshold_ms: int = 3000
    voice_highlight_duration_ms: int = 1800
    require_gesture_confirm_on_create: bool = True


@dataclass
class SessionConfig:
    """Configuración completa de una sesión experimental."""

    participant_code: str = "PART-2026-001"
    condition: ConditionLevel = ConditionLevel.MEDIUM
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    canvas_width: int = 1280
    canvas_height: int = 720
