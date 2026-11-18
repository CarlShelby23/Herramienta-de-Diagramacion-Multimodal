from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GestureState(str, Enum):
    """Estados del cursor virtual circular cinemático."""

    HAND_LOST = "HAND_LOST"
    IDLE = "IDLE"
    MOVE = "MOVE"                  # Azul: índice extendido
    PINCH_SELECT = "PINCH_SELECT"  # Morado: pellizco < umbral
    FIST_DRAG = "FIST_DRAG"        # Naranja: puño cerrado
    OPEN_DROP = "OPEN_DROP"        # Verde: mano abierta


class GestureEventType(str, Enum):
    """Eventos discretos de flanco único tras superar Dwell Time (250 ms)."""

    NONE = "NONE"
    SELECT_CONFIRMED = "SELECT_CONFIRMED"
    DRAG_START = "DRAG_START"
    DROP_CONFIRMED = "DROP_CONFIRMED"
    HAND_LOST = "HAND_LOST"
    HAND_RECOVERED = "HAND_RECOVERED"


@dataclass
class GestureFrame:
    """Instantánea de visión y estado gestual emitida por el motor de visión."""

    timestamp_ms: int
    raw_gesture: GestureState = GestureState.HAND_LOST
    stable_gesture: GestureState = GestureState.HAND_LOST
    event: GestureEventType = GestureEventType.NONE
    dwell_progress: float = 0.0  # 0.0 .. 1.0
    cursor_norm_x: float | None = None
    cursor_norm_y: float | None = None
    cursor_x: float | None = None
    cursor_y: float | None = None
    pinch_distance_px: float | None = None
    effective_threshold_px: float | None = None
    hand_scale_px: float | None = None
    hand_detected: bool = False
    landmarks_count: int = 0
    fps: float = 0.0
    camera_ok: bool = True


class VoiceIntent(str, Enum):
    """Catálogo acotado de intenciones de voz (ASR)."""

    CREATE_NODE = "CREATE_NODE"
    CONNECT = "CONNECT"
    DELETE = "DELETE"
    UNDO = "UNDO"
    UNRECOGNIZED = "UNRECOGNIZED"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class AsrEvent:
    """Resultado estructurado de reconocimiento de voz con separación de latencias."""

    raw_text: str
    intent: VoiceIntent
    concept: str | None = None
    t_speech_start_ms: int = 0
    t_speech_end_ms: int = 0
    t_asr_ready_ms: int = 0
    system_latency_ms: int = 0
    confidence: float = 1.0
    error_message: str | None = None


class InterruptionCategory(str, Enum):
    """Clasificación estricta de desenlaces de secuencias (Indicador 2)."""

    VALID_ACTION = "VALID_ACTION"
    INTENTIONAL_CANCEL = "INTENTIONAL_CANCEL"
    TIMEOUT_ABANDON = "TIMEOUT_ABANDON"
    RECOGNITION_ERROR = "RECOGNITION_ERROR"


@dataclass
class TelemetryEvent:
    """Evento atómico con timestamp registrado durante la sesión."""

    timestamp_ms: int
    modality: str          # "MULTIMODAL" | "VOZ" | "GESTO" | "UI" | "SISTEMA"
    event_type: str        # "NODE_CREATE", "EDGE_CREATE", "NODE_MOVE", etc.
    detail: str = ""
    is_structural: bool = False
    sequence_id: str | None = None
    x: float | None = None
    y: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "modalidad": self.modality,
            "evento": self.event_type,
            "detalle": self.detail,
            "es_estructural": self.is_structural,
            "sequence_id": self.sequence_id,
            "coordenada_x": round(self.x, 1) if self.x is not None else None,
            "coordenada_y": round(self.y, 1) if self.y is not None else None,
            "metadata": self.metadata,
        }
