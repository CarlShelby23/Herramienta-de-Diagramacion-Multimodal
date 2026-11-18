from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any

from src.config import SessionConfig
from src.models.diagram import DiagramState
from src.models.events import InterruptionCategory, TelemetryEvent
from src.models.session import (
    BehavioralKPIs,
    InactivitySegment,
    SequenceRecord,
    SessionDocument,
)


class TelemetryTracker:
    """Calcula en segundo plano los 4 indicadores conductuales operacionales (REQ-04).

    1. Tiempo de inactividad (ms) con banderas de salud de cámara, mano y micrófono.
    2. Frecuencia de interrupciones (conteo discreto por categoría).
    3. Latencia de respuesta del usuario (ms) separada de la latencia del sistema ASR.
    4. Modificaciones estructurales (conteo discreto excluyendo movimientos puros).
    """

    SCHEMA_VERSION = "2.0.0-REQ04"

    def __init__(self, config: SessionConfig, start_time_ms: int | None = None) -> None:
        self.config = config
        now_ms = start_time_ms if start_time_ms is not None else int(time.time() * 1000)
        self.started_at_ms: int = now_ms
        self.started_at_iso: str = datetime.fromtimestamp(
            now_ms / 1000.0, tz=timezone.utc
        ).astimezone().isoformat()
        self.ended_at_ms: int | None = None
        self.ended_at_iso: str | None = None
        self.task_active: bool = True

        # Estado de periféricos para contextualizar inactividad
        self.camera_ok: bool = True
        self.hand_detected: bool = False
        self.mic_ok: bool = True

        self._camera_loss_events: int = 0
        self._mic_loss_events: int = 0
        self._hand_loss_events: int = 0

        # Estado de inactividad (Indicador 1)
        self._inactivity_threshold_ms: int = config.fusion.inactivity_threshold_ms
        self._last_activity_ms: int = now_ms
        self._inactivity_open_start_ms: int | None = None
        self._inactivity_open_hw: tuple[bool, bool, bool] = (True, False, True)
        self._inactivity_segments: list[InactivitySegment] = []

        # Secuencias y eventos (Indicadores 2, 3, 4)
        self._seq_counter: int = 0
        self._sequences: dict[str, SequenceRecord] = {}
        self._sequence_order: list[str] = []
        self.events: list[TelemetryEvent] = []

        self._system_asr_latencies: list[int] = []
        self.latest_system_latency_ms: int | None = None
        self.latest_user_latency_ms: int | None = None

        self._nodes_created: int = 0
        self._nodes_deleted: int = 0
        self._edges_created: int = 0
        self._edges_deleted: int = 0
        self._non_structural_moves: int = 0

        self.log_event(
            timestamp_ms=now_ms,
            modality="SISTEMA",
            event_type="SESSION_START",
            detail=f"Sesión iniciada ({config.participant_code} | {config.condition.value})",
            is_structural=False,
        )

    # --- Salud de periféricos y actividad (Indicador 1) ---

    def update_hardware_health(
        self,
        now_ms: int,
        camera_ok: bool | None = None,
        hand_detected: bool | None = None,
        mic_ok: bool | None = None,
    ) -> None:
        if camera_ok is not None and camera_ok != self.camera_ok:
            if not camera_ok:
                self._camera_loss_events += 1
                self.log_event(
                    now_ms, "SISTEMA", "CAMERA_LOST", "Pérdida de señal de cámara"
                )
            else:
                self.log_event(
                    now_ms, "SISTEMA", "CAMERA_RECOVERED", "Señal de cámara recuperada"
                )
            self.camera_ok = camera_ok

        if mic_ok is not None and mic_ok != self.mic_ok:
            if not mic_ok:
                self._mic_loss_events += 1
                self.log_event(
                    now_ms, "SISTEMA", "MIC_LOST", "Pérdida de señal de micrófono/ASR"
                )
            else:
                self.log_event(
                    now_ms, "SISTEMA", "MIC_RECOVERED", "Micrófono/ASR recuperado"
                )
            self.mic_ok = mic_ok

        if hand_detected is not None and hand_detected != self.hand_detected:
            if not hand_detected:
                self._hand_loss_events += 1
                self.log_event(
                    now_ms, "GESTO", "HAND_LOST", "Mano fuera del campo visual"
                )
            else:
                self.log_event(
                    now_ms, "GESTO", "HAND_RECOVERED", "Mano detectada nuevamente"
                )
            self.hand_detected = hand_detected

    def record_activity(self, now_ms: int) -> None:
        """Registra actividad real del usuario y cierra cualquier tramo abierto de inactividad."""
        if not self.task_active:
            return
        if self._inactivity_open_start_ms is not None:
            self._close_inactivity_segment(now_ms)
        elif (now_ms - self._last_activity_ms) >= self._inactivity_threshold_ms:
            # Si hubo un salto temporal mayor al umbral antes de un tick, registrarlo
            self._inactivity_segments.append(
                InactivitySegment(
                    start_ms=self._last_activity_ms,
                    end_ms=now_ms,
                    duration_ms=now_ms - self._last_activity_ms,
                    camera_ok=self.camera_ok,
                    hand_detected=self.hand_detected,
                    mic_ok=self.mic_ok,
                )
            )
        self._last_activity_ms = now_ms

    def tick(self, now_ms: int) -> None:
        """Evaluación periódica para abrir tramos de inactividad mientras la tarea está activa."""
        if not self.task_active:
            return
        if (
            self._inactivity_open_start_ms is None
            and (now_ms - self._last_activity_ms) >= self._inactivity_threshold_ms
        ):
            self._inactivity_open_start_ms = self._last_activity_ms
            self._inactivity_open_hw = (
                self.camera_ok,
                self.hand_detected,
                self.mic_ok,
            )

    def _close_inactivity_segment(self, end_ms: int) -> None:
        if self._inactivity_open_start_ms is None:
            return
        start_ms = self._inactivity_open_start_ms
        duration = max(0, end_ms - start_ms)
        if duration >= self._inactivity_threshold_ms:
            cam_ok, hand_ok, mic_ok = self._inactivity_open_hw
            self._inactivity_segments.append(
                InactivitySegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration,
                    camera_ok=cam_ok and self.camera_ok,
                    hand_detected=hand_ok and self.hand_detected,
                    mic_ok=mic_ok and self.mic_ok,
                )
            )
        self._inactivity_open_start_ms = None

    # --- Secuencias e Interrupciones (Indicadores 2 y 3) ---

    def start_sequence(
        self,
        intent: str,
        now_ms: int,
        t_speech_end_ms: int | None = None,
        system_asr_latency_ms: int | None = None,
    ) -> str:
        self._seq_counter += 1
        seq_id = f"SEQ-{self._seq_counter:04d}"
        record = SequenceRecord(
            sequence_id=seq_id,
            intent=intent,
            started_at_ms=now_ms,
            t_speech_end_ms=t_speech_end_ms,
            system_asr_latency_ms=system_asr_latency_ms,
        )
        self._sequences[seq_id] = record
        self._sequence_order.append(seq_id)

        if system_asr_latency_ms is not None:
            self._system_asr_latencies.append(system_asr_latency_ms)
            self.latest_system_latency_ms = system_asr_latency_ms

        return seq_id

    def complete_sequence(
        self,
        sequence_id: str,
        outcome: InterruptionCategory,
        now_ms: int,
        reason: str = "",
        t_gesture_confirm_ms: int | None = None,
    ) -> SequenceRecord | None:
        record = self._sequences.get(sequence_id)
        if record is None or record.outcome is not None:
            return record

        record.ended_at_ms = now_ms
        record.outcome = outcome
        record.reason = reason

        if (
            outcome == InterruptionCategory.VALID_ACTION
            and record.t_speech_end_ms is not None
            and t_gesture_confirm_ms is not None
        ):
            record.t_gesture_confirm_ms = t_gesture_confirm_ms
            user_lat = max(0, t_gesture_confirm_ms - record.t_speech_end_ms)
            record.user_response_latency_ms = user_lat
            self.latest_user_latency_ms = user_lat

        if outcome != InterruptionCategory.VALID_ACTION:
            self.log_event(
                timestamp_ms=now_ms,
                modality="SISTEMA",
                event_type="SEQUENCE_INTERRUPTION",
                detail=f"{outcome.value}: {reason}",
                is_structural=False,
                sequence_id=sequence_id,
                metadata={"categoria": outcome.value, "intencion": record.intent},
            )
        return record

    def record_instant_error(
        self,
        intent: str,
        now_ms: int,
        reason: str,
        raw_text: str = "",
        system_asr_latency_ms: int | None = None,
    ) -> str:
        seq_id = self.start_sequence(
            intent=intent,
            now_ms=now_ms,
            system_asr_latency_ms=system_asr_latency_ms,
        )
        self.complete_sequence(
            sequence_id=seq_id,
            outcome=InterruptionCategory.RECOGNITION_ERROR,
            now_ms=now_ms,
            reason=reason,
        )
        self.log_event(
            timestamp_ms=now_ms,
            modality="VOZ",
            event_type="RECOGNITION_ERROR",
            detail=reason,
            is_structural=False,
            sequence_id=seq_id,
            metadata={"texto_crudo": raw_text},
        )
        return seq_id

    # --- Registro de eventos y Modificaciones Estructurales (Indicador 4) ---

    def log_event(
        self,
        timestamp_ms: int,
        modality: str,
        event_type: str,
        detail: str = "",
        is_structural: bool = False,
        sequence_id: str | None = None,
        x: float | None = None,
        y: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        evt = TelemetryEvent(
            timestamp_ms=timestamp_ms,
            modality=modality,
            event_type=event_type,
            detail=detail,
            is_structural=is_structural,
            sequence_id=sequence_id,
            x=x,
            y=y,
            metadata=metadata or {},
        )
        self.events.append(evt)

        if event_type == "NODE_CREATE" and is_structural:
            self._nodes_created += 1
        elif event_type == "NODE_DELETE" and is_structural:
            self._nodes_deleted += 1
        elif event_type == "EDGE_CREATE" and is_structural:
            self._edges_created += 1
        elif event_type == "EDGE_DELETE" and is_structural:
            self._edges_deleted += 1
        elif event_type == "NODE_MOVE":
            self._non_structural_moves += 1

        return evt

    # --- Cierre y cálculo de KPIs ---

    def finalize(self, now_ms: int | None = None) -> None:
        if not self.task_active:
            return
        end_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.tick(end_ms)
        self._close_inactivity_segment(end_ms)
        # Cerrar cualquier secuencia abierta al finalizar sesión como abandono
        for seq_id in self._sequence_order:
            seq = self._sequences[seq_id]
            if seq.outcome is None:
                self.complete_sequence(
                    seq_id,
                    InterruptionCategory.TIMEOUT_ABANDON,
                    end_ms,
                    reason="Sesión finalizada con secuencia pendiente",
                )
        self.task_active = False
        self.ended_at_ms = end_ms
        self.ended_at_iso = datetime.fromtimestamp(
            end_ms / 1000.0, tz=timezone.utc
        ).astimezone().isoformat()
        self.log_event(
            end_ms,
            "SISTEMA",
            "SESSION_END",
            "Sesión experimental cerrada",
            is_structural=False,
        )

    def compute_kpis(self, now_ms: int | None = None) -> BehavioralKPIs:
        ref_ms = (
            self.ended_at_ms
            if self.ended_at_ms is not None
            else (now_ms if now_ms is not None else int(time.time() * 1000))
        )

        # Incluir segmento abierto si se consulta en vivo
        segments = list(self._inactivity_segments)
        if self._inactivity_open_start_ms is not None and self.task_active:
            open_dur = max(0, ref_ms - self._inactivity_open_start_ms)
            if open_dur >= self._inactivity_threshold_ms:
                cam_ok, hand_ok, mic_ok = self._inactivity_open_hw
                segments.append(
                    InactivitySegment(
                        start_ms=self._inactivity_open_start_ms,
                        end_ms=ref_ms,
                        duration_ms=open_dur,
                        camera_ok=cam_ok and self.camera_ok,
                        hand_detected=hand_ok and self.hand_detected,
                        mic_ok=mic_ok and self.mic_ok,
                    )
                )

        inactivity_total_ms = sum(s.duration_ms for s in segments)

        # Indicador 2 y 3
        intentional = 0
        timeouts = 0
        rec_errors = 0
        valid_actions = 0
        user_latencies: list[int] = []

        for seq_id in self._sequence_order:
            seq = self._sequences[seq_id]
            if seq.outcome == InterruptionCategory.VALID_ACTION:
                valid_actions += 1
                if seq.user_response_latency_ms is not None:
                    user_latencies.append(seq.user_response_latency_ms)
            elif seq.outcome == InterruptionCategory.INTENTIONAL_CANCEL:
                intentional += 1
            elif seq.outcome == InterruptionCategory.TIMEOUT_ABANDON:
                timeouts += 1
            elif seq.outcome == InterruptionCategory.RECOGNITION_ERROR:
                rec_errors += 1

        interruptions_total = intentional + timeouts + rec_errors

        user_mean = round(statistics.mean(user_latencies), 2) if user_latencies else None
        user_median = round(statistics.median(user_latencies), 2) if user_latencies else None
        user_min = min(user_latencies) if user_latencies else None
        user_max = max(user_latencies) if user_latencies else None

        sys_mean = (
            round(statistics.mean(self._system_asr_latencies), 2)
            if self._system_asr_latencies
            else None
        )

        # Indicador 4: exclusivamente operaciones estructurales
        structural_total = (
            self._nodes_created
            + self._nodes_deleted
            + self._edges_created
            + self._edges_deleted
        )

        return BehavioralKPIs(
            inactivity_total_ms=inactivity_total_ms,
            inactivity_segments_count=len(segments),
            camera_loss_events=self._camera_loss_events,
            mic_loss_events=self._mic_loss_events,
            hand_loss_events=self._hand_loss_events,
            inactivity_segments=segments,
            interruptions_total=interruptions_total,
            intentional_cancels=intentional,
            timeout_abandons=timeouts,
            recognition_errors=rec_errors,
            valid_actions_total=valid_actions,
            user_latency_mean_ms=user_mean,
            user_latency_median_ms=user_median,
            user_latency_min_ms=user_min,
            user_latency_max_ms=user_max,
            user_latency_samples_ms=list(user_latencies),
            system_asr_latency_mean_ms=sys_mean,
            system_asr_latency_samples_ms=list(self._system_asr_latencies),
            structural_modifications_total=structural_total,
            nodes_created_count=self._nodes_created,
            nodes_deleted_count=self._nodes_deleted,
            edges_created_count=self._edges_created,
            edges_deleted_count=self._edges_deleted,
            non_structural_moves_count=self._non_structural_moves,
        )

    def build_session_document(
        self,
        diagram: DiagramState,
        session_filename: str,
        now_ms: int | None = None,
    ) -> SessionDocument:
        end_ms = (
            self.ended_at_ms
            if self.ended_at_ms is not None
            else (now_ms if now_ms is not None else int(time.time() * 1000))
        )
        end_iso = self.ended_at_iso or datetime.fromtimestamp(
            end_ms / 1000.0, tz=timezone.utc
        ).astimezone().isoformat()

        return SessionDocument(
            schema_version=self.SCHEMA_VERSION,
            session_filename=session_filename,
            participant_code=self.config.participant_code,
            condition=self.config.condition.value,
            started_at_iso=self.started_at_iso,
            ended_at_iso=end_iso,
            duration_ms=max(0, end_ms - self.started_at_ms),
            calibration=self.config.calibration.to_dict(),
            topology=diagram.to_topology_dict(),
            kpis=self.compute_kpis(now_ms=end_ms),
            sequences=[self._sequences[sid] for sid in self._sequence_order],
            events=list(self.events),
        )
