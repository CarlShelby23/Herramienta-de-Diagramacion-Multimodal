from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src.config import FusionConfig
from src.models.diagram import (
    CreateEdgeCommand,
    CreateNodeCommand,
    DeleteEdgeCommand,
    DeleteNodeCommand,
    DiagramState,
    EdgeModel,
    MoveNodeCommand,
    NodeModel,
)
from src.models.events import (
    AsrEvent,
    GestureEventType,
    GestureFrame,
    GestureState,
    InterruptionCategory,
    VoiceIntent,
)
from src.telemetry.tracker import TelemetryTracker


class InteractionMode(str, Enum):
    """Estados operativos de la máquina de fusión multimodal."""

    READY = "READY"
    PENDING_CREATE = "PENDING_CREATE"
    DRAGGING = "DRAGGING"
    CONNECT_AWAITING_TARGET = "CONNECT_AWAITING_TARGET"


@dataclass
class PendingCreateState:
    concept: str
    sequence_id: str
    started_at_ms: int
    t_speech_end_ms: int
    x: float
    y: float


@dataclass
class DraggingState:
    node_id: str
    origin_x: float
    origin_y: float
    sequence_id: str
    started_at_ms: int
    hand_lost_since_ms: int | None = None


@dataclass
class ConnectPendingState:
    source_node_id: str
    sequence_id: str
    started_at_ms: int
    t_speech_end_ms: int | None


@dataclass
class FusionUpdate:
    """Resumen de estado para sincronizar la UI tras procesar gesto/voz/tick."""

    feedback_message: str = ""
    feedback_level: str = "info"  # "info" | "success" | "warning" | "error"
    diagram_changed: bool = False


class MultimodalFusionEngine:
    """Coordinador determinístico Voz + Gestos sin falsos positivos ni acciones a medias.

    Reglas garantizadas:
    - Separación estricta entre movimiento (MOVE), selección (PINCH 250 ms),
      arrastre (FIST 250 ms) y confirmación/drop (OPEN 250 ms).
    - Arrastrar/soltar un nodo reposiciona el nodo y sus aristas, pero NUNCA conecta por accidente.
    - "Conectar" enlaza explícitamente un nodo origen seleccionado con un destino confirmado.
    - "Crear nodo [Concepto]" muestra estado verde segmentado + (X, Y) y confirma en ventana temporal
      midiendo latencia de respuesta del usuario (Indicador 3).
    - "Borrar" / "Eliminar" exige elemento seleccionado explícitamente.
    - Comandos ambiguos o no reconocidos nunca ejecutan cambios estructurales.
    """

    def __init__(
        self,
        diagram: DiagramState,
        telemetry: TelemetryTracker,
        config: FusionConfig,
        canvas_width: int = 1280,
        canvas_height: int = 720,
    ) -> None:
        self.diagram = diagram
        self.telemetry = telemetry
        self.config = config
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.mode: InteractionMode = InteractionMode.READY
        self.selected_node_id: str | None = None
        self.selected_edge_id: str | None = None

        self.pending_create: PendingCreateState | None = None
        self.dragging: DraggingState | None = None
        self.pending_connect: ConnectPendingState | None = None

        # Resaltado verde segmentado temporal tras crear nodo por voz
        self.voice_highlighted_nodes: dict[str, int] = {}  # node_id -> expire_ms

        # Última posición válida del cursor para ventana temporal de fusión
        self.last_cursor_x: float | None = None
        self.last_cursor_y: float | None = None
        self.last_cursor_ts_ms: int = 0
        self._last_activity_cursor_x: float | None = None
        self._last_activity_cursor_y: float | None = None

    def set_canvas_size(self, width: int, height: int) -> None:
        self.canvas_width = max(320, int(width))
        self.canvas_height = max(240, int(height))

    def _clamp_node_pos(self, x: float, y: float) -> tuple[float, float]:
        margin_x = 105.0
        margin_y = 48.0
        cx = max(margin_x, min(self.canvas_width - margin_x, x))
        cy = max(margin_y, min(self.canvas_height - margin_y, y))
        return cx, cy

    def _has_recent_cursor(self, now_ms: int) -> bool:
        return (
            self.last_cursor_x is not None
            and self.last_cursor_y is not None
            and (now_ms - self.last_cursor_ts_ms) <= self.config.cursor_recent_window_ms
        )

    # --- Procesamiento de Gestos (Visión o Simulación) ---

    def process_gesture(self, frame: GestureFrame) -> FusionUpdate:
        now_ms = frame.timestamp_ms
        update = FusionUpdate()

        self.telemetry.update_hardware_health(
            now_ms=now_ms,
            camera_ok=frame.camera_ok,
            hand_detected=frame.hand_detected,
        )

        # Actualizar expiraciones y ventanas de tiempo
        timeout_update = self.tick(now_ms)
        if timeout_update.feedback_message:
            update = timeout_update

        # Manejo de pérdida de mano
        if not frame.hand_detected or frame.stable_gesture == GestureState.HAND_LOST:
            if self.mode == InteractionMode.DRAGGING and self.dragging is not None:
                if self.dragging.hand_lost_since_ms is None:
                    self.dragging.hand_lost_since_ms = now_ms
                elif (now_ms - self.dragging.hand_lost_since_ms) >= self.config.hand_lost_timeout_ms:
                    self._abort_drag(
                        now_ms,
                        InterruptionCategory.TIMEOUT_ABANDON,
                        "Mano perdida durante el arrastre; posición restaurada.",
                    )
                    update.feedback_message = "Mano perdida al arrastrar: nodo restaurado."
                    update.feedback_level = "warning"
                    update.diagram_changed = True
            return update

        # Mano detectada: actualizar coordenadas
        if frame.cursor_x is not None and frame.cursor_y is not None:
            self.last_cursor_x = frame.cursor_x
            self.last_cursor_y = frame.cursor_y
            self.last_cursor_ts_ms = now_ms

            if self._last_activity_cursor_x is None or (
                math.hypot(
                    frame.cursor_x - self._last_activity_cursor_x,
                    frame.cursor_y - self._last_activity_cursor_y,
                )
                >= 8.0
            ):
                self._last_activity_cursor_x = frame.cursor_x
                self._last_activity_cursor_y = frame.cursor_y
                self.telemetry.record_activity(now_ms)

        cx = self.last_cursor_x if self.last_cursor_x is not None else self.canvas_width / 2.0
        cy = self.last_cursor_y if self.last_cursor_y is not None else self.canvas_height / 2.0

        # Si hay creación pendiente por voz, la previsualización sigue suavemente el cursor
        if self.mode == InteractionMode.PENDING_CREATE and self.pending_create is not None:
            px, py = self._clamp_node_pos(cx, cy)
            self.pending_create.x = px
            self.pending_create.y = py

            if frame.event in (
                GestureEventType.DROP_CONFIRMED,
                GestureEventType.SELECT_CONFIRMED,
            ):
                self.telemetry.record_activity(now_ms)
                return self._confirm_pending_create(now_ms)
            return update

        # Si estamos arrastrando un nodo con puño cerrado
        if self.mode == InteractionMode.DRAGGING and self.dragging is not None:
            self.dragging.hand_lost_since_ms = None
            node = self.diagram.nodes.get(self.dragging.node_id)
            if node is None:
                self._abort_drag(now_ms, InterruptionCategory.INTENTIONAL_CANCEL, "Nodo inexistente")
                return update

            if frame.event == GestureEventType.DROP_CONFIRMED:
                self.telemetry.record_activity(now_ms)
                return self._commit_drag(now_ms)

            # Actualizar posición en vivo mientras el puño siga activo o en transición controlada
            if frame.stable_gesture == GestureState.FIST_DRAG:
                nx, ny = self._clamp_node_pos(cx, cy)
                node.x = nx
                node.y = ny
                update.diagram_changed = True
            return update

        # Si estamos esperando seleccionar el nodo destino de "Conectar"
        if (
            self.mode == InteractionMode.CONNECT_AWAITING_TARGET
            and self.pending_connect is not None
        ):
            if frame.event in (
                GestureEventType.SELECT_CONFIRMED,
                GestureEventType.DROP_CONFIRMED,
            ):
                self.telemetry.record_activity(now_ms)
                target_node = self.diagram.node_at(cx, cy)
                return self._resolve_connect_target(target_node, now_ms, cx, cy)
            return update

        # Modo READY: selección (Morado 250 ms) o inicio de arrastre (Naranja 250 ms)
        if frame.event == GestureEventType.SELECT_CONFIRMED:
            self.telemetry.record_activity(now_ms)
            return self._handle_select_at(cx, cy, now_ms)

        if frame.event == GestureEventType.DRAG_START:
            self.telemetry.record_activity(now_ms)
            return self._start_drag_at(cx, cy, now_ms)

        return update

    # --- Procesamiento de Comandos de Voz (ASR) ---

    def process_asr(self, event: AsrEvent) -> FusionUpdate:
        now_ms = event.t_asr_ready_ms or event.t_speech_end_ms
        self.telemetry.record_activity(now_ms)

        # Comandos no reconocidos o ambiguos: nunca ejecutan acción estructural
        if event.intent in (VoiceIntent.UNRECOGNIZED, VoiceIntent.AMBIGUOUS):
            reason = event.error_message or f"Comando no reconocido: '{event.raw_text}'"
            self.telemetry.record_instant_error(
                intent=event.intent.value,
                now_ms=now_ms,
                reason=reason,
                raw_text=event.raw_text,
                system_asr_latency_ms=event.system_latency_ms,
            )
            return FusionUpdate(feedback_message=reason, feedback_level="warning")

        if event.intent == VoiceIntent.CREATE_NODE:
            return self._handle_voice_create(event, now_ms)

        if event.intent == VoiceIntent.CONNECT:
            return self._handle_voice_connect(event, now_ms)

        if event.intent == VoiceIntent.DELETE:
            return self._handle_delete(
                now_ms=now_ms,
                modality="VOZ",
                t_speech_end_ms=event.t_speech_end_ms,
                system_asr_latency_ms=event.system_latency_ms,
            )

        if event.intent == VoiceIntent.UNDO:
            return self._handle_undo(
                now_ms=now_ms,
                modality="VOZ",
                system_asr_latency_ms=event.system_latency_ms,
            )

        return FusionUpdate()

    # --- Acciones disparadas desde la UI (barra contextual / botón Deshacer) ---

    def trigger_connect_from_ui(self, now_ms: int) -> FusionUpdate:
        self.telemetry.record_activity(now_ms)
        return self._enter_connect_mode(
            now_ms=now_ms,
            modality="UI",
            t_speech_end_ms=None,
            system_asr_latency_ms=None,
        )

    def trigger_delete_from_ui(self, now_ms: int) -> FusionUpdate:
        self.telemetry.record_activity(now_ms)
        return self._handle_delete(now_ms=now_ms, modality="UI")

    def trigger_undo_from_ui(self, now_ms: int) -> FusionUpdate:
        self.telemetry.record_activity(now_ms)
        return self._handle_undo(now_ms=now_ms, modality="UI")

    def select_at_point_from_ui(self, x: float, y: float, now_ms: int) -> FusionUpdate:
        """Permite interacción limpia con ratón/clic en modo simulación o contingencia."""
        self.telemetry.record_activity(now_ms)
        self.last_cursor_x = x
        self.last_cursor_y = y
        self.last_cursor_ts_ms = now_ms

        if self.mode == InteractionMode.PENDING_CREATE and self.pending_create is not None:
            px, py = self._clamp_node_pos(x, y)
            self.pending_create.x = px
            self.pending_create.y = py
            return self._confirm_pending_create(now_ms)

        if self.mode == InteractionMode.CONNECT_AWAITING_TARGET:
            target = self.diagram.node_at(x, y)
            return self._resolve_connect_target(target, now_ms, x, y)

        return self._handle_select_at(x, y, now_ms)

    def cancel_active_sequence(self, now_ms: int, reason: str = "Cancelado por el usuario") -> FusionUpdate:
        self.telemetry.record_activity(now_ms)
        if self.mode == InteractionMode.PENDING_CREATE and self.pending_create is not None:
            seq_id = self.pending_create.sequence_id
            self.pending_create = None
            self.mode = InteractionMode.READY
            self.telemetry.complete_sequence(
                seq_id, InterruptionCategory.INTENTIONAL_CANCEL, now_ms, reason=reason
            )
            return FusionUpdate(
                feedback_message="Creación de nodo cancelada.",
                feedback_level="info",
                diagram_changed=True,
            )

        if self.mode == InteractionMode.CONNECT_AWAITING_TARGET and self.pending_connect is not None:
            seq_id = self.pending_connect.sequence_id
            self.pending_connect = None
            self.mode = InteractionMode.READY
            self.telemetry.complete_sequence(
                seq_id, InterruptionCategory.INTENTIONAL_CANCEL, now_ms, reason=reason
            )
            return FusionUpdate(
                feedback_message="Conexión cancelada.",
                feedback_level="info",
                diagram_changed=True,
            )

        if self.mode == InteractionMode.DRAGGING and self.dragging is not None:
            self._abort_drag(now_ms, InterruptionCategory.INTENTIONAL_CANCEL, reason)
            return FusionUpdate(
                feedback_message="Arrastre cancelado: posición original restaurada.",
                feedback_level="info",
                diagram_changed=True,
            )

        return FusionUpdate()

    # --- Temporizadores y Ventanas Temporales ---

    def tick(self, now_ms: int) -> FusionUpdate:
        self.telemetry.tick(now_ms)
        update = FusionUpdate()

        # Expirar resaltados verdes temporales de nodos recién creados
        expired_highlights = [
            nid for nid, exp in self.voice_highlighted_nodes.items() if now_ms >= exp
        ]
        if expired_highlights:
            for nid in expired_highlights:
                self.voice_highlighted_nodes.pop(nid, None)
            update.diagram_changed = True

        # Timeout de creación pendiente por voz
        if self.mode == InteractionMode.PENDING_CREATE and self.pending_create is not None:
            elapsed = now_ms - self.pending_create.started_at_ms
            if elapsed >= self.config.create_confirm_window_ms:
                seq_id = self.pending_create.sequence_id
                concept = self.pending_create.concept
                self.pending_create = None
                self.mode = InteractionMode.READY
                self.telemetry.complete_sequence(
                    seq_id,
                    InterruptionCategory.TIMEOUT_ABANDON,
                    now_ms,
                    reason=f"Tiempo agotado esperando confirmación gestual para '{concept}'",
                )
                update.feedback_message = (
                    f"Creación de '{concept}' expiró sin confirmación gestual."
                )
                update.feedback_level = "warning"
                update.diagram_changed = True

        # Timeout de acople Origen-Destino
        if (
            self.mode == InteractionMode.CONNECT_AWAITING_TARGET
            and self.pending_connect is not None
        ):
            elapsed = now_ms - self.pending_connect.started_at_ms
            if elapsed >= self.config.connect_target_window_ms:
                seq_id = self.pending_connect.sequence_id
                self.pending_connect = None
                self.mode = InteractionMode.READY
                self.telemetry.complete_sequence(
                    seq_id,
                    InterruptionCategory.TIMEOUT_ABANDON,
                    now_ms,
                    reason="Tiempo agotado esperando nodo destino para conectar",
                )
                update.feedback_message = "Conexión cancelada por tiempo de espera agotado."
                update.feedback_level = "warning"
                update.diagram_changed = True

        return update

    # --- Implementaciones Internas ---

    def _handle_select_at(self, x: float, y: float, now_ms: int) -> FusionUpdate:
        node = self.diagram.node_at(x, y)
        if node is not None:
            self.selected_node_id = node.id
            self.selected_edge_id = None
            self.telemetry.log_event(
                now_ms,
                "GESTO",
                "NODE_SELECT",
                f"Nodo seleccionado: '{node.text}' ({node.id})",
                is_structural=False,
                x=x,
                y=y,
                metadata={"node_id": node.id},
            )
            return FusionUpdate(
                feedback_message=f"Nodo seleccionado: '{node.text}'",
                feedback_level="info",
                diagram_changed=True,
            )

        edge = self.diagram.edge_at(x, y)
        if edge is not None:
            self.selected_edge_id = edge.id
            self.selected_node_id = None
            self.telemetry.log_event(
                now_ms,
                "GESTO",
                "EDGE_SELECT",
                f"Enlace seleccionado ({edge.source_id} -> {edge.target_id})",
                is_structural=False,
                x=x,
                y=y,
                metadata={"edge_id": edge.id},
            )
            return FusionUpdate(
                feedback_message="Enlace seleccionado. Di 'Borrar' o usa la barra contextual.",
                feedback_level="info",
                diagram_changed=True,
            )

        # Clic/pellizco en espacio vacío limpia selección
        if self.selected_node_id or self.selected_edge_id:
            self.selected_node_id = None
            self.selected_edge_id = None
            return FusionUpdate(
                feedback_message="Selección limpiada.",
                feedback_level="info",
                diagram_changed=True,
            )
        return FusionUpdate()

    def _start_drag_at(self, x: float, y: float, now_ms: int) -> FusionUpdate:
        # Priorizar el nodo bajo el cursor; si no, el nodo actualmente seleccionado si está cerca
        node = self.diagram.node_at(x, y, margin=24.0)
        if node is None and self.selected_node_id:
            sel = self.diagram.nodes.get(self.selected_node_id)
            if sel is not None and sel.contains(x, y, margin=48.0):
                node = sel

        if node is None:
            return FusionUpdate(
                feedback_message="Puño detectado en espacio vacío: apunta a un nodo para arrastrarlo.",
                feedback_level="info",
            )

        self.selected_node_id = node.id
        self.selected_edge_id = None
        seq_id = self.telemetry.start_sequence("NODE_MOVE", now_ms)
        self.dragging = DraggingState(
            node_id=node.id,
            origin_x=node.x,
            origin_y=node.y,
            sequence_id=seq_id,
            started_at_ms=now_ms,
        )
        self.mode = InteractionMode.DRAGGING
        self.telemetry.log_event(
            now_ms,
            "GESTO",
            "NODE_DRAG_START",
            f"Arrastrando '{node.text}'",
            is_structural=False,
            sequence_id=seq_id,
            x=node.x,
            y=node.y,
            metadata={"node_id": node.id},
        )
        return FusionUpdate(
            feedback_message=f"Arrastrando '{node.text}'. Abre la mano (Verde) para soltar.",
            feedback_level="info",
            diagram_changed=True,
        )

    def _commit_drag(self, now_ms: int) -> FusionUpdate:
        assert self.dragging is not None
        drag = self.dragging
        self.dragging = None
        self.mode = InteractionMode.READY

        node = self.diagram.nodes.get(drag.node_id)
        if node is None:
            return FusionUpdate(diagram_changed=True)

        moved_dist = math.hypot(node.x - drag.origin_x, node.y - drag.origin_y)
        if moved_dist >= 3.0:
            cmd = MoveNodeCommand(
                node_id=node.id,
                old_x=drag.origin_x,
                old_y=drag.origin_y,
                new_x=node.x,
                new_y=node.y,
            )
            self.diagram.execute(cmd)

        self.telemetry.complete_sequence(
            drag.sequence_id,
            InterruptionCategory.VALID_ACTION,
            now_ms,
            reason=f"Nodo '{node.text}' soltado en ({int(node.x)}, {int(node.y)})",
        )
        self.telemetry.log_event(
            now_ms,
            "GESTO",
            "NODE_MOVE",
            f"Nodo '{node.text}' movido a ({int(node.x)}, {int(node.y)})",
            is_structural=False,  # Mover NO cuenta como modificación estructural
            sequence_id=drag.sequence_id,
            x=node.x,
            y=node.y,
            metadata={"node_id": node.id},
        )
        return FusionUpdate(
            feedback_message=f"Nodo '{node.text}' colocado en ({int(node.x)}, {int(node.y)}).",
            feedback_level="success",
            diagram_changed=True,
        )

    def _abort_drag(
        self, now_ms: int, category: InterruptionCategory, reason: str
    ) -> None:
        if self.dragging is None:
            return
        drag = self.dragging
        self.dragging = None
        self.mode = InteractionMode.READY

        node = self.diagram.nodes.get(drag.node_id)
        if node is not None:
            node.x = drag.origin_x
            node.y = drag.origin_y

        self.telemetry.complete_sequence(
            drag.sequence_id, category, now_ms, reason=reason
        )

    def _handle_voice_create(self, event: AsrEvent, now_ms: int) -> FusionUpdate:
        if self.mode != InteractionMode.READY:
            # Cancelar secuencia previa limpia si existía
            self.cancel_active_sequence(now_ms, reason="Reemplazada por nuevo comando de voz")

        concept = (event.concept or "").strip()
        if not concept:
            self.telemetry.record_instant_error(
                "CREATE_NODE",
                now_ms,
                "Concepto vacío al crear nodo",
                raw_text=event.raw_text,
                system_asr_latency_ms=event.system_latency_ms,
            )
            return FusionUpdate(
                feedback_message="Falta el concepto tras 'Crear nodo'.",
                feedback_level="warning",
            )

        has_cursor = self._has_recent_cursor(now_ms)
        if has_cursor and self.last_cursor_x is not None and self.last_cursor_y is not None:
            px, py = self._clamp_node_pos(self.last_cursor_x, self.last_cursor_y)
        else:
            # Si no hay cursor reciente, no instanciar a ciegas; abrir ventana esperando mano
            px, py = self._clamp_node_pos(self.canvas_width / 2.0, self.canvas_height / 2.0)

        seq_id = self.telemetry.start_sequence(
            intent="CREATE_NODE",
            now_ms=now_ms,
            t_speech_end_ms=event.t_speech_end_ms,
            system_asr_latency_ms=event.system_latency_ms,
        )

        # Si se exige confirmación gestual (o no había cursor reciente), abrir estado PENDING_CREATE
        if self.config.require_gesture_confirm_on_create or not has_cursor:
            self.pending_create = PendingCreateState(
                concept=concept,
                sequence_id=seq_id,
                started_at_ms=now_ms,
                t_speech_end_ms=event.t_speech_end_ms,
                x=px,
                y=py,
            )
            self.mode = InteractionMode.PENDING_CREATE
            return FusionUpdate(
                feedback_message=(
                    f"Crear '{concept}' en ({int(px)}, {int(py)}): "
                    "confirma con mano abierta (Verde) o pellizco (250 ms)."
                ),
                feedback_level="info",
                diagram_changed=True,
            )

        # Modo creación inmediata (cuando require_gesture_confirm_on_create=False y hay cursor válido)
        self.pending_create = PendingCreateState(
            concept=concept,
            sequence_id=seq_id,
            started_at_ms=now_ms,
            t_speech_end_ms=event.t_speech_end_ms,
            x=px,
            y=py,
        )
        return self._confirm_pending_create(now_ms)

    def _confirm_pending_create(self, now_ms: int) -> FusionUpdate:
        assert self.pending_create is not None
        pending = self.pending_create
        self.pending_create = None
        self.mode = InteractionMode.READY

        node = NodeModel(
            id=self.diagram.next_node_id(),
            text=pending.concept,
            x=pending.x,
            y=pending.y,
        )
        self.diagram.execute(CreateNodeCommand(node))
        self.selected_node_id = node.id
        self.selected_edge_id = None
        self.voice_highlighted_nodes[node.id] = (
            now_ms + self.config.voice_highlight_duration_ms
        )

        self.telemetry.complete_sequence(
            pending.sequence_id,
            InterruptionCategory.VALID_ACTION,
            now_ms,
            reason=f"Nodo '{node.text}' instanciado en ({int(node.x)}, {int(node.y)})",
            t_gesture_confirm_ms=now_ms,
        )
        self.telemetry.log_event(
            now_ms,
            "MULTIMODAL",
            "NODE_CREATE",
            f"Nodo creado: '{node.text}'",
            is_structural=True,
            sequence_id=pending.sequence_id,
            x=node.x,
            y=node.y,
            metadata={"node_id": node.id, "texto": node.text},
        )
        return FusionUpdate(
            feedback_message=f"Nodo '{node.text}' creado en ({int(node.x)}, {int(node.y)}).",
            feedback_level="success",
            diagram_changed=True,
        )

    def _handle_voice_connect(self, event: AsrEvent, now_ms: int) -> FusionUpdate:
        return self._enter_connect_mode(
            now_ms=now_ms,
            modality="VOZ",
            t_speech_end_ms=event.t_speech_end_ms,
            system_asr_latency_ms=event.system_latency_ms,
        )

    def _enter_connect_mode(
        self,
        now_ms: int,
        modality: str,
        t_speech_end_ms: int | None,
        system_asr_latency_ms: int | None,
    ) -> FusionUpdate:
        if self.mode == InteractionMode.DRAGGING:
            # No mezclar arrastre con conexión accidental
            self._abort_drag(
                now_ms,
                InterruptionCategory.INTENTIONAL_CANCEL,
                "Arrastre detenido antes de iniciar conexión explícita",
            )

        source_node = (
            self.diagram.nodes.get(self.selected_node_id)
            if self.selected_node_id
            else None
        )
        if source_node is None:
            # Relación ambigua: no existe nodo origen seleccionado -> NO crear enlace
            self.telemetry.record_instant_error(
                intent="CONNECT",
                now_ms=now_ms,
                reason="Intento de 'Conectar' sin nodo origen seleccionado",
                system_asr_latency_ms=system_asr_latency_ms,
            )
            return FusionUpdate(
                feedback_message="Selecciona primero un nodo origen (pellizco Morado 250 ms) antes de decir 'Conectar'.",
                feedback_level="warning",
            )

        if len(self.diagram.nodes) < 2:
            self.telemetry.record_instant_error(
                intent="CONNECT",
                now_ms=now_ms,
                reason="Se requieren al menos 2 nodos para crear un enlace",
                system_asr_latency_ms=system_asr_latency_ms,
            )
            return FusionUpdate(
                feedback_message="Crea al menos dos nodos antes de conectar.",
                feedback_level="warning",
            )

        seq_id = self.telemetry.start_sequence(
            intent="CONNECT",
            now_ms=now_ms,
            t_speech_end_ms=t_speech_end_ms,
            system_asr_latency_ms=system_asr_latency_ms,
        )
        self.pending_connect = ConnectPendingState(
            source_node_id=source_node.id,
            sequence_id=seq_id,
            started_at_ms=now_ms,
            t_speech_end_ms=t_speech_end_ms,
        )
        self.mode = InteractionMode.CONNECT_AWAITING_TARGET
        self.telemetry.log_event(
            now_ms,
            modality,
            "EDGE_CONNECT_INIT",
            f"Origen fijado: '{source_node.text}' ({source_node.id})",
            is_structural=False,
            sequence_id=seq_id,
            x=source_node.x,
            y=source_node.y,
            metadata={"source_id": source_node.id},
        )
        return FusionUpdate(
            feedback_message=(
                f"Acople activo desde '{source_node.text}': "
                "selecciona el nodo destino (pellizco Morado o mano abierta 250 ms)."
            ),
            feedback_level="info",
            diagram_changed=True,
        )

    def _resolve_connect_target(
        self, target_node: NodeModel | None, now_ms: int, x: float, y: float
    ) -> FusionUpdate:
        assert self.pending_connect is not None
        pending = self.pending_connect
        self.pending_connect = None
        self.mode = InteractionMode.READY

        source_node = self.diagram.nodes.get(pending.source_node_id)
        if source_node is None or target_node is None:
            self.telemetry.complete_sequence(
                pending.sequence_id,
                InterruptionCategory.INTENTIONAL_CANCEL,
                now_ms,
                reason="Selección en espacio vacío canceló la conexión",
            )
            return FusionUpdate(
                feedback_message="Conexión cancelada (no se seleccionó un nodo destino).",
                feedback_level="warning",
                diagram_changed=True,
            )

        if target_node.id == source_node.id:
            self.telemetry.complete_sequence(
                pending.sequence_id,
                InterruptionCategory.INTENTIONAL_CANCEL,
                now_ms,
                reason="Auto-enlace rechazado (origen == destino)",
            )
            return FusionUpdate(
                feedback_message="El nodo destino debe ser distinto al nodo origen.",
                feedback_level="warning",
                diagram_changed=True,
            )

        if self.diagram.has_edge_between(source_node.id, target_node.id):
            self.telemetry.complete_sequence(
                pending.sequence_id,
                InterruptionCategory.RECOGNITION_ERROR,
                now_ms,
                reason="Enlace duplicado rechazado",
            )
            return FusionUpdate(
                feedback_message=f"Ya existe un enlace entre '{source_node.text}' y '{target_node.text}'.",
                feedback_level="warning",
                diagram_changed=True,
            )

        edge = EdgeModel(
            id=self.diagram.next_edge_id(),
            source_id=source_node.id,
            target_id=target_node.id,
        )
        self.diagram.execute(CreateEdgeCommand(edge))
        self.selected_node_id = target_node.id
        self.selected_edge_id = None

        self.telemetry.complete_sequence(
            pending.sequence_id,
            InterruptionCategory.VALID_ACTION,
            now_ms,
            reason=f"Enlace {source_node.id} -> {target_node.id}",
            t_gesture_confirm_ms=now_ms,
        )
        self.telemetry.log_event(
            now_ms,
            "MULTIMODAL",
            "EDGE_CREATE",
            f"Enlace creado: '{source_node.text}' -> '{target_node.text}'",
            is_structural=True,
            sequence_id=pending.sequence_id,
            x=x,
            y=y,
            metadata={
                "edge_id": edge.id,
                "source_id": source_node.id,
                "target_id": target_node.id,
            },
        )
        return FusionUpdate(
            feedback_message=f"Conectado: '{source_node.text}' — '{target_node.text}'.",
            feedback_level="success",
            diagram_changed=True,
        )

    def _handle_delete(
        self,
        now_ms: int,
        modality: str,
        t_speech_end_ms: int | None = None,
        system_asr_latency_ms: int | None = None,
    ) -> FusionUpdate:
        if self.mode != InteractionMode.READY:
            self.cancel_active_sequence(now_ms, reason="Cancelada antes de borrar")

        if self.selected_node_id and self.selected_node_id in self.diagram.nodes:
            node = self.diagram.nodes[self.selected_node_id]
            seq_id = self.telemetry.start_sequence(
                "DELETE_NODE",
                now_ms,
                t_speech_end_ms=t_speech_end_ms,
                system_asr_latency_ms=system_asr_latency_ms,
            )
            self.diagram.execute(DeleteNodeCommand(node.id))
            self.selected_node_id = None
            self.telemetry.complete_sequence(
                seq_id,
                InterruptionCategory.VALID_ACTION,
                now_ms,
                reason=f"Nodo '{node.text}' eliminado",
            )
            self.telemetry.log_event(
                now_ms,
                modality,
                "NODE_DELETE",
                f"Nodo eliminado: '{node.text}'",
                is_structural=True,
                sequence_id=seq_id,
                x=node.x,
                y=node.y,
                metadata={"node_id": node.id},
            )
            return FusionUpdate(
                feedback_message=f"Nodo '{node.text}' eliminado. Di 'Deshacer' si fue un error.",
                feedback_level="success",
                diagram_changed=True,
            )

        if self.selected_edge_id and self.selected_edge_id in self.diagram.edges:
            edge = self.diagram.edges[self.selected_edge_id]
            seq_id = self.telemetry.start_sequence(
                "DELETE_EDGE",
                now_ms,
                t_speech_end_ms=t_speech_end_ms,
                system_asr_latency_ms=system_asr_latency_ms,
            )
            self.diagram.execute(DeleteEdgeCommand(edge.id))
            self.selected_edge_id = None
            self.telemetry.complete_sequence(
                seq_id,
                InterruptionCategory.VALID_ACTION,
                now_ms,
                reason=f"Enlace '{edge.id}' eliminado",
            )
            self.telemetry.log_event(
                now_ms,
                modality,
                "EDGE_DELETE",
                f"Enlace eliminado ({edge.source_id} -> {edge.target_id})",
                is_structural=True,
                sequence_id=seq_id,
                metadata={"edge_id": edge.id},
            )
            return FusionUpdate(
                feedback_message="Enlace eliminado. Di 'Deshacer' para restaurarlo.",
                feedback_level="success",
                diagram_changed=True,
            )

        self.telemetry.record_instant_error(
            intent="DELETE",
            now_ms=now_ms,
            reason="Intento de borrado sin elemento seleccionado",
            system_asr_latency_ms=system_asr_latency_ms,
        )
        return FusionUpdate(
            feedback_message="Selecciona primero un nodo o enlace para evitar borrados accidentales.",
            feedback_level="warning",
        )

    def _handle_undo(
        self,
        now_ms: int,
        modality: str,
        system_asr_latency_ms: int | None = None,
    ) -> FusionUpdate:
        if self.mode != InteractionMode.READY:
            return self.cancel_active_sequence(now_ms, reason="Cancelación vía Deshacer")

        cmd = self.diagram.undo()
        if cmd is None:
            return FusionUpdate(
                feedback_message="No hay operaciones anteriores para deshacer.",
                feedback_level="info",
            )

        # Limpiar selección si apuntaba a algo ya revertido
        if self.selected_node_id and self.selected_node_id not in self.diagram.nodes:
            self.selected_node_id = None
        if self.selected_edge_id and self.selected_edge_id not in self.diagram.edges:
            self.selected_edge_id = None

        seq_id = self.telemetry.start_sequence(
            "UNDO", now_ms, system_asr_latency_ms=system_asr_latency_ms
        )
        self.telemetry.complete_sequence(
            seq_id,
            InterruptionCategory.VALID_ACTION,
            now_ms,
            reason=f"Revertido: {cmd.action_name}",
        )
        self.telemetry.log_event(
            now_ms,
            modality,
            "UNDO",
            f"Operación revertida: {cmd.action_name}",
            is_structural=False,
            sequence_id=seq_id,
            metadata={"reverted_action": cmd.action_name},
        )
        return FusionUpdate(
            feedback_message=f"Deshecho: {cmd.action_name}.",
            feedback_level="success",
            diagram_changed=True,
        )
