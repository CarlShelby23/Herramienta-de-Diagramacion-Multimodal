from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models.events import InterruptionCategory, TelemetryEvent


@dataclass
class InactivitySegment:
    """Tramo continuo de inactividad durante la tarea activa (Indicador 1)."""

    start_ms: int
    end_ms: int
    duration_ms: int
    camera_ok: bool
    hand_detected: bool
    mic_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "inicio_ms": self.start_ms,
            "fin_ms": self.end_ms,
            "duracion_ms": self.duration_ms,
            "camara_activa": self.camera_ok,
            "mano_detectada": self.hand_detected,
            "microfono_activo": self.mic_ok,
        }


@dataclass
class SequenceRecord:
    """Ciclo de vida de una secuencia multimodal para clasificar interrupciones (Indicador 2)."""

    sequence_id: str
    intent: str
    started_at_ms: int
    ended_at_ms: int | None = None
    outcome: InterruptionCategory | None = None
    reason: str = ""
    t_speech_end_ms: int | None = None
    t_gesture_confirm_ms: int | None = None
    system_asr_latency_ms: int | None = None
    user_response_latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "intencion": self.intent,
            "inicio_ms": self.started_at_ms,
            "fin_ms": self.ended_at_ms,
            "resultado": self.outcome.value if self.outcome else None,
            "motivo": self.reason,
            "t_speech_end_ms": self.t_speech_end_ms,
            "t_gesture_confirm_ms": self.t_gesture_confirm_ms,
            "latencia_sistema_asr_ms": self.system_asr_latency_ms,
            "latencia_respuesta_usuario_ms": self.user_response_latency_ms,
        }


@dataclass
class BehavioralKPIs:
    """Resumen consolidado de los 4 indicadores conductuales operacionales (REQ-04)."""

    # 1. Tiempo de inactividad (ms)
    inactivity_total_ms: int = 0
    inactivity_segments_count: int = 0
    camera_loss_events: int = 0
    mic_loss_events: int = 0
    hand_loss_events: int = 0
    inactivity_segments: list[InactivitySegment] = field(default_factory=list)

    # 2. Frecuencia de interrupciones (conteo discreto con desglose)
    interruptions_total: int = 0
    intentional_cancels: int = 0
    timeout_abandons: int = 0
    recognition_errors: int = 0
    valid_actions_total: int = 0

    # 3. Latencia de respuesta del usuario vs sistema (ms)
    user_latency_mean_ms: float | None = None
    user_latency_median_ms: float | None = None
    user_latency_min_ms: int | None = None
    user_latency_max_ms: int | None = None
    user_latency_samples_ms: list[int] = field(default_factory=list)
    system_asr_latency_mean_ms: float | None = None
    system_asr_latency_samples_ms: list[int] = field(default_factory=list)

    # 4. Modificaciones estructurales (conteo discreto)
    structural_modifications_total: int = 0
    nodes_created_count: int = 0
    nodes_deleted_count: int = 0
    edges_created_count: int = 0
    edges_deleted_count: int = 0
    non_structural_moves_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tiempo_inactividad_ms": {
                "total_ms": self.inactivity_total_ms,
                "numero_tramos": self.inactivity_segments_count,
                "perdidas_camara": self.camera_loss_events,
                "perdidas_microfono": self.mic_loss_events,
                "perdidas_mano": self.hand_loss_events,
                "tramos": [s.to_dict() for s in self.inactivity_segments],
            },
            "frecuencia_interrupciones": {
                "total_interrupciones": self.interruptions_total,
                "cancelaciones_intencionales": self.intentional_cancels,
                "abandonos_timeout": self.timeout_abandons,
                "errores_reconocimiento": self.recognition_errors,
                "acciones_validas": self.valid_actions_total,
            },
            "latencia_respuesta_usuario_ms": {
                "promedio_usuario_ms": self.user_latency_mean_ms,
                "mediana_usuario_ms": self.user_latency_median_ms,
                "min_usuario_ms": self.user_latency_min_ms,
                "max_usuario_ms": self.user_latency_max_ms,
                "muestras_usuario_ms": self.user_latency_samples_ms,
                "promedio_sistema_asr_ms": self.system_asr_latency_mean_ms,
                "muestras_sistema_asr_ms": self.system_asr_latency_samples_ms,
            },
            "modificaciones_estructurales": {
                "total": self.structural_modifications_total,
                "nodos_creados": self.nodes_created_count,
                "nodos_eliminados": self.nodes_deleted_count,
                "enlaces_creados": self.edges_created_count,
                "enlaces_eliminados": self.edges_deleted_count,
                "movimientos_no_estructurales": self.non_structural_moves_count,
            },
        }


@dataclass
class SessionDocument:
    """Documento único consolidado REQ-04 (Topología + Telemetría + 4 KPIs)."""

    schema_version: str
    session_filename: str
    participant_code: str
    condition: str
    started_at_iso: str
    ended_at_iso: str
    duration_ms: int
    calibration: dict[str, Any]
    topology: dict[str, Any]
    kpis: BehavioralKPIs
    sequences: list[SequenceRecord]
    events: list[TelemetryEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "archivo": self.session_filename,
            "sesion": {
                "codigo_participante": self.participant_code,
                "condicion": self.condition,
                "inicio_iso": self.started_at_iso,
                "fin_iso": self.ended_at_iso,
                "duracion_ms": self.duration_ms,
            },
            "calibracion": self.calibration,
            "topologia": self.topology,
            "indicadores_conductuales": self.kpis.to_dict(),
            "secuencias": [s.to_dict() for s in self.sequences],
            "eventos_telemetria": [e.to_dict() for e in self.events],
        }
