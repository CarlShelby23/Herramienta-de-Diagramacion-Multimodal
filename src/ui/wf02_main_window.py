from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QMouseEvent, QPainter, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.audio.asr_worker import AsrWorker
from src.audio.command_parser import parse_voice_command
from src.config import CURSOR_COLORS, TASK_INSTRUCTIONS, SessionConfig
from src.fusion.fusion_engine import InteractionMode, MultimodalFusionEngine
from src.models.diagram import DiagramState, NodeVisualState
from src.models.events import AsrEvent, GestureEventType, GestureFrame, GestureState
from src.telemetry.persistence import (
    build_session_json_filename,
    export_scene_to_png,
)
from src.telemetry.tracker import TelemetryTracker
from src.ui.canvas_items import DashedEdgeItem, RoundedNodeItem, VirtualCursorItem
from src.ui.wf05_close_dialog import WF05CloseDialog
from src.vision.camera_worker import CameraVisionWorker


class InteractiveDiagramView(QGraphicsView):
    """QGraphicsView con soporte para contingencia/simulación por ratón."""

    canvas_clicked = Signal(float, float)
    canvas_moved = Signal(float, float)

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background-color: #F8FAFC; border: none;")
        self.setMouseTracking(True)
        self._dragging_mouse = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.canvas_clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = self.mapToScene(event.position().toPoint())
        self.canvas_moved.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)


class WF02MainWindow(QMainWindow):
    """WF-02 / WF-03 / WF-04 — Lienzo Principal, Manipulación Directa y Acople."""

    def __init__(
        self,
        config: SessionConfig,
        start_workers: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle(
            f"Herramienta de Diagramación Multimodal | {config.participant_code} [{config.condition.value}]"
        )
        self.resize(config.canvas_width, config.canvas_height + 110)

        # Motores centrales
        self.diagram = DiagramState()
        self.telemetry = TelemetryTracker(config)
        self.fusion = MultimodalFusionEngine(
            diagram=self.diagram,
            telemetry=self.telemetry,
            config=config.fusion,
            canvas_width=config.canvas_width,
            canvas_height=config.canvas_height,
        )

        # Ítems gráficos sincronizados con DiagramState
        self._node_items: dict[str, RoundedNodeItem] = {}
        self._edge_items: dict[str, DashedEdgeItem] = {}
        self._pending_node_item: RoundedNodeItem | None = None
        self._cursor_item = VirtualCursorItem()

        # Hilos de hardware
        self._camera_worker: CameraVisionWorker | None = None
        self._asr_worker: AsrWorker | None = None
        self._session_closed = False

        self._build_ui()
        self._setup_shortcuts()

        # Temporizador de 100 ms para ventanas temporales, reloj e inactividad
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(100)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start()

        if start_workers and not config.calibration.simulation_mode:
            self._start_workers()
        else:
            self._set_asr_status("Modo simulación (consola/ratón)", True)
            self._set_gesture_badge(GestureState.MOVE)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- 1. Barra Superior (Instrucción de Tarea + Cierre) ---
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_lay = QHBoxLayout(top_bar)
        top_lay.setContentsMargins(16, 10, 16, 10)
        top_lay.setSpacing(12)

        instruction = TASK_INSTRUCTIONS.get(self.config.condition, "")
        self.lbl_instruction = QLabel(instruction)
        self.lbl_instruction.setObjectName("taskInstruction")
        self.lbl_instruction.setWordWrap(True)

        self.lbl_session_meta = QLabel(
            f"{self.config.participant_code}  ·  00:00"
        )
        self.lbl_session_meta.setStyleSheet(
            "background-color: #F1F5F9; padding: 5px 10px; border-radius: 6px; "
            "font-family: Consolas, monospace; font-weight: 700; color: #334155;"
        )

        self.btn_finish = QPushButton("Finalizar Sesión (WF-05)")
        self.btn_finish.setObjectName("primaryButton")
        self.btn_finish.clicked.connect(self.open_close_modal)

        top_lay.addWidget(self.lbl_instruction, stretch=1)
        top_lay.addWidget(self.lbl_session_meta)
        top_lay.addWidget(self.btn_finish)
        main_layout.addWidget(top_bar)

        # --- 2. Lienzo Interactivo Principal ---
        self.scene = QGraphicsScene(
            0, 0, self.config.canvas_width, self.config.canvas_height, self
        )
        self.scene.addItem(self._cursor_item)

        self.view = InteractiveDiagramView(self.scene, self)
        self.view.canvas_clicked.connect(self._on_canvas_clicked)
        self.view.canvas_moved.connect(self._on_canvas_mouse_moved)
        main_layout.addWidget(self.view, stretch=1)

        # --- Barra Contextual Flotante (Conectar, Borrar, Deshacer) ---
        self.floating_bar = QFrame(self.view.viewport())
        self.floating_bar.setObjectName("floatingToolbar")
        fb_lay = QHBoxLayout(self.floating_bar)
        fb_lay.setContentsMargins(8, 6, 8, 6)
        fb_lay.setSpacing(6)

        self.btn_ctx_connect = QPushButton("Conectar")
        self.btn_ctx_connect.setObjectName("accentButton")
        self.btn_ctx_connect.clicked.connect(self._on_ctx_connect)

        self.btn_ctx_delete = QPushButton("Borrar")
        self.btn_ctx_delete.setObjectName("dangerButton")
        self.btn_ctx_delete.clicked.connect(self._on_ctx_delete)

        self.btn_ctx_undo = QPushButton("Deshacer")
        self.btn_ctx_undo.clicked.connect(self._on_undo)

        self.btn_ctx_cancel = QPushButton("Cancelar")
        self.btn_ctx_cancel.clicked.connect(self._on_cancel_sequence)
        self.btn_ctx_cancel.setVisible(False)

        fb_lay.addWidget(self.btn_ctx_connect)
        fb_lay.addWidget(self.btn_ctx_delete)
        fb_lay.addWidget(self.btn_ctx_undo)
        fb_lay.addWidget(self.btn_ctx_cancel)
        self.floating_bar.adjustSize()
        self.floating_bar.hide()

        # --- Barra Opcional de Simulación de Voz (solo visible en modo simulación) ---
        self.sim_bar = QFrame()
        self.sim_bar.setStyleSheet(
            "background-color: #EFF6FF; border-top: 1px solid #BFDBFE;"
        )
        sim_lay = QHBoxLayout(self.sim_bar)
        sim_lay.setContentsMargins(14, 6, 14, 6)
        sim_lay.setSpacing(8)
        sim_lay.addWidget(QLabel("Simulador Voz:"))
        self.input_sim_voice = QLineEdit()
        self.input_sim_voice.setPlaceholderText(
            "Escribe un comando (ej. 'Crear nodo Percepción', 'Conectar', 'Borrar', 'Deshacer') y pulsa Enter"
        )
        self.input_sim_voice.returnPressed.connect(self._on_sim_voice_submit)
        btn_sim_send = QPushButton("Enviar Voz")
        btn_sim_send.clicked.connect(self._on_sim_voice_submit)
        btn_sim_confirm = QPushButton("Confirmar Drop (Verde)")
        btn_sim_confirm.clicked.connect(self._on_sim_drop_confirm)
        sim_lay.addWidget(self.input_sim_voice, stretch=1)
        sim_lay.addWidget(btn_sim_send)
        sim_lay.addWidget(btn_sim_confirm)
        self.sim_bar.setVisible(self.config.calibration.simulation_mode)
        main_layout.addWidget(self.sim_bar)

        # --- 3. Barra Inferior (Estado ASR/Gesto, Feedback, Latencia y Deshacer) ---
        bottom_bar = QFrame()
        bottom_bar.setObjectName("bottomBar")
        bot_lay = QHBoxLayout(bottom_bar)
        bot_lay.setContentsMargins(14, 8, 14, 8)
        bot_lay.setSpacing(12)

        self.lbl_asr_badge = QLabel("ASR: Iniciando...")
        self.lbl_asr_badge.setStyleSheet(
            "background-color: #F1F5F9; padding: 4px 10px; border-radius: 6px; font-weight: 600;"
        )

        self.lbl_gesture_badge = QLabel("Gesto: Esperando mano")
        self.lbl_gesture_badge.setStyleSheet(
            "background-color: #F1F5F9; padding: 4px 10px; border-radius: 6px; font-weight: 600;"
        )

        self.lbl_feedback = QLabel("Listo. Di 'Crear nodo [concepto]' o apunta con tu mano.")
        self.lbl_feedback.setStyleSheet("color: #334155; font-weight: 500;")

        self.lbl_latency = QLabel("Latencia — Usuario: -- ms | ASR: -- ms")
        self.lbl_latency.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; color: #475569;"
        )

        self.btn_bottom_undo = QPushButton("Deshacer")
        self.btn_bottom_undo.clicked.connect(self._on_undo)
        self.btn_bottom_undo.setEnabled(False)

        bot_lay.addWidget(self.lbl_asr_badge)
        bot_lay.addWidget(self.lbl_gesture_badge)
        bot_lay.addWidget(self.lbl_feedback, stretch=1)
        bot_lay.addWidget(self.lbl_latency)
        bot_lay.addWidget(self.btn_bottom_undo)
        main_layout.addWidget(bottom_bar)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._on_undo)
        QShortcut(QKeySequence("Escape"), self, activated=self._on_cancel_sequence)

    def _start_workers(self) -> None:
        self._camera_worker = CameraVisionWorker(
            calibration=self.config.calibration,
            canvas_width=int(self.scene.width()),
            canvas_height=int(self.scene.height()),
            emit_preview=False,
            hand_grace_period_ms=self.config.fusion.hand_grace_period_ms,
        )
        self._camera_worker.gesture_updated.connect(self.on_gesture_frame)
        self._camera_worker.status_changed.connect(self._on_camera_status)
        self._camera_worker.start()

        self._asr_worker = AsrWorker(calibration=self.config.calibration)
        self._asr_worker.asr_event.connect(self.on_asr_event)
        self._asr_worker.partial_text.connect(self._on_asr_partial)
        self._asr_worker.status_changed.connect(self._set_asr_status)
        self._asr_worker.start()

    def stop_workers(self) -> None:
        self._tick_timer.stop()
        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None
        if self._asr_worker is not None:
            self._asr_worker.stop()
            self._asr_worker = None

    # --- Slots de Hardware y Simulación ---

    def _on_camera_status(self, msg: str, ok: bool) -> None:
        if not ok:
            self._set_feedback(f"Cámara: {msg}", "warning")

    def _set_asr_status(self, msg: str, ok: bool) -> None:
        self.telemetry.update_hardware_health(
            now_ms=int(time.time() * 1000), mic_ok=ok
        )
        bg = "#DCFCE7" if ok else "#FEF3C7"
        fg = "#166534" if ok else "#92400E"
        self.lbl_asr_badge.setText(f"ASR: {msg}")
        self.lbl_asr_badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 4px 10px; "
            "border-radius: 6px; font-weight: 600;"
        )

    def _on_asr_partial(self, partial: str) -> None:
        if partial:
            self.lbl_asr_badge.setText(f"ASR: «{partial}...»")

    def _set_gesture_badge(self, state: GestureState) -> None:
        labels = {
            GestureState.MOVE: "Gesto: Mover (Azul)",
            GestureState.PINCH_SELECT: "Gesto: Pellizco (Morado)",
            GestureState.FIST_DRAG: "Gesto: Arrastrar (Naranja)",
            GestureState.OPEN_DROP: "Gesto: Soltar/Drop (Verde)",
            GestureState.IDLE: "Gesto: Reposo",
            GestureState.HAND_LOST: "Gesto: Sin mano",
        }
        color = CURSOR_COLORS.get(state.value, "#64748B")
        self.lbl_gesture_badge.setText(labels.get(state, state.value))
        self.lbl_gesture_badge.setStyleSheet(
            f"background-color: {color}; color: #FFFFFF; padding: 4px 10px; "
            "border-radius: 6px; font-weight: 600;"
        )

    def _set_feedback(self, message: str, level: str = "info") -> None:
        if not message:
            return
        palette = {
            "info": "#1E293B",
            "success": "#15803D",
            "warning": "#B45309",
            "error": "#B91C1C",
        }
        self.lbl_feedback.setText(message)
        self.lbl_feedback.setStyleSheet(
            f"color: {palette.get(level, '#1E293B')}; font-weight: 600;"
        )

    def on_gesture_frame(self, frame: GestureFrame) -> None:
        self._set_gesture_badge(frame.stable_gesture)
        self._cursor_item.update_cursor(
            frame.cursor_x, frame.cursor_y, frame.stable_gesture, frame.dwell_progress
        )
        update = self.fusion.process_gesture(frame)
        if update.feedback_message:
            self._set_feedback(update.feedback_message, update.feedback_level)
        if update.diagram_changed or self.fusion.mode == InteractionMode.PENDING_CREATE:
            self.sync_canvas()
        self._refresh_status_metrics()

    def on_asr_event(self, event: AsrEvent) -> None:
        update = self.fusion.process_asr(event)
        if update.feedback_message:
            self._set_feedback(update.feedback_message, update.feedback_level)
        if update.diagram_changed:
            self.sync_canvas()
        self._refresh_status_metrics()

    def _on_canvas_mouse_moved(self, x: float, y: float) -> None:
        if not self.config.calibration.simulation_mode:
            return
        now_ms = int(time.time() * 1000)
        self.on_gesture_frame(
            GestureFrame(
                timestamp_ms=now_ms,
                raw_gesture=GestureState.MOVE,
                stable_gesture=GestureState.MOVE,
                event=GestureEventType.NONE,
                cursor_x=x,
                cursor_y=y,
                hand_detected=True,
                landmarks_count=21,
                camera_ok=True,
            )
        )

    def _on_canvas_clicked(self, x: float, y: float) -> None:
        now_ms = int(time.time() * 1000)
        update = self.fusion.select_at_point_from_ui(x, y, now_ms)
        if update.feedback_message:
            self._set_feedback(update.feedback_message, update.feedback_level)
        if update.diagram_changed:
            self.sync_canvas()
        self._refresh_status_metrics()

    def _on_sim_voice_submit(self) -> None:
        text = self.input_sim_voice.text().strip()
        if not text:
            return
        self.input_sim_voice.clear()
        now_ms = int(time.time() * 1000)
        parsed = parse_voice_command(text)
        evt = AsrEvent(
            raw_text=text,
            intent=parsed.intent,
            concept=parsed.concept,
            t_speech_start_ms=now_ms - 500,
            t_speech_end_ms=now_ms - 40,
            t_asr_ready_ms=now_ms,
            system_latency_ms=40,
            error_message=None
            if parsed.intent.name not in ("UNRECOGNIZED", "AMBIGUOUS")
            else parsed.reason,
        )
        self.on_asr_event(evt)

    def _on_sim_drop_confirm(self) -> None:
        now_ms = int(time.time() * 1000)
        cx = self.fusion.last_cursor_x or (self.scene.width() / 2.0)
        cy = self.fusion.last_cursor_y or (self.scene.height() / 2.0)
        self.on_gesture_frame(
            GestureFrame(
                timestamp_ms=now_ms,
                raw_gesture=GestureState.OPEN_DROP,
                stable_gesture=GestureState.OPEN_DROP,
                event=GestureEventType.DROP_CONFIRMED,
                dwell_progress=1.0,
                cursor_x=cx,
                cursor_y=cy,
                hand_detected=True,
                landmarks_count=21,
                camera_ok=True,
            )
        )

    # --- Acciones de Barra Contextual y Barra Inferior ---

    def _on_ctx_connect(self) -> None:
        update = self.fusion.trigger_connect_from_ui(int(time.time() * 1000))
        self._apply_update(update)

    def _on_ctx_delete(self) -> None:
        update = self.fusion.trigger_delete_from_ui(int(time.time() * 1000))
        self._apply_update(update)

    def _on_undo(self) -> None:
        update = self.fusion.trigger_undo_from_ui(int(time.time() * 1000))
        self._apply_update(update)

    def _on_cancel_sequence(self) -> None:
        update = self.fusion.cancel_active_sequence(int(time.time() * 1000))
        self._apply_update(update)

    def _apply_update(self, update) -> None:
        if update.feedback_message:
            self._set_feedback(update.feedback_message, update.feedback_level)
        if update.diagram_changed:
            self.sync_canvas()
        self._refresh_status_metrics()

    # --- Sincronización del Lienzo y Barra Flotante ---

    def _on_tick(self) -> None:
        now_ms = int(time.time() * 1000)
        update = self.fusion.tick(now_ms)
        if update.feedback_message:
            self._set_feedback(update.feedback_message, update.feedback_level)
        if update.diagram_changed:
            self.sync_canvas()

        elapsed_s = max(0, (now_ms - self.telemetry.started_at_ms) // 1000)
        mins, secs = divmod(elapsed_s, 60)
        self.lbl_session_meta.setText(
            f"{self.config.participant_code}  ·  {mins:02d}:{secs:02d}"
        )

    def _refresh_status_metrics(self) -> None:
        u_lat = (
            f"{self.telemetry.latest_user_latency_ms} ms"
            if self.telemetry.latest_user_latency_ms is not None
            else "-- ms"
        )
        s_lat = (
            f"{self.telemetry.latest_system_latency_ms} ms"
            if self.telemetry.latest_system_latency_ms is not None
            else "-- ms"
        )
        self.lbl_latency.setText(f"Latencia — Usuario: {u_lat} | ASR: {s_lat}")
        self.btn_bottom_undo.setEnabled(
            self.diagram.can_undo or self.fusion.mode != InteractionMode.READY
        )
        self.btn_ctx_undo.setEnabled(self.diagram.can_undo)

    def sync_canvas(self) -> None:
        # 1. Sincronizar nodos existentes
        active_node_ids = set(self.diagram.nodes.keys())
        for stale_id in list(self._node_items.keys()):
            if stale_id not in active_node_ids:
                item = self._node_items.pop(stale_id)
                self.scene.removeItem(item)

        for node_id, model in self.diagram.nodes.items():
            visual = self._visual_state_for_node(node_id)
            if node_id not in self._node_items:
                item = RoundedNodeItem(
                    node_id=node_id,
                    text=model.text,
                    x=model.x,
                    y=model.y,
                    width=model.width,
                    height=model.height,
                    visual_state=visual,
                )
                self.scene.addItem(item)
                self._node_items[node_id] = item
            else:
                self._node_items[node_id].sync_from_model(model, visual)

        # 2. Nodo fantasma de previsualización en creación por voz (verde segmentado + X, Y)
        if (
            self.fusion.mode == InteractionMode.PENDING_CREATE
            and self.fusion.pending_create is not None
        ):
            pc = self.fusion.pending_create
            if self._pending_node_item is None:
                self._pending_node_item = RoundedNodeItem(
                    node_id="__pending__",
                    text=pc.concept,
                    x=pc.x,
                    y=pc.y,
                    visual_state=NodeVisualState.PENDING_VOICE,
                )
                self._pending_node_item.setZValue(25)
                self.scene.addItem(self._pending_node_item)
            else:
                self._pending_node_item.text = pc.concept
                self._pending_node_item.visual_state = NodeVisualState.PENDING_VOICE
                self._pending_node_item.setPos(pc.x, pc.y)
                self._pending_node_item.update()
        elif self._pending_node_item is not None:
            self.scene.removeItem(self._pending_node_item)
            self._pending_node_item = None

        # 3. Sincronizar enlaces rectos segmentados
        active_edge_ids = set(self.diagram.edges.keys())
        for stale_eid in list(self._edge_items.keys()):
            if stale_eid not in active_edge_ids:
                item = self._edge_items.pop(stale_eid)
                self.scene.removeItem(item)

        for edge_id, edge in self.diagram.edges.items():
            src = self.diagram.nodes.get(edge.source_id)
            dst = self.diagram.nodes.get(edge.target_id)
            if src is None or dst is None:
                continue
            if edge_id not in self._edge_items:
                e_item = DashedEdgeItem(edge_id)
                self.scene.addItem(e_item)
                self._edge_items[edge_id] = e_item
            e_item = self._edge_items[edge_id]
            e_item.update_endpoints(src, dst)
            e_item.set_selected_visual(edge_id == self.fusion.selected_edge_id)

        # 4. Posicionar barra contextual flotante
        self._update_floating_toolbar()

    def _visual_state_for_node(self, node_id: str) -> NodeVisualState:
        if (
            self.fusion.mode == InteractionMode.CONNECT_AWAITING_TARGET
            and self.fusion.pending_connect is not None
            and self.fusion.pending_connect.source_node_id == node_id
        ):
            return NodeVisualState.CONNECT_SOURCE
        if node_id in self.fusion.voice_highlighted_nodes:
            return NodeVisualState.VOICE_CREATED
        if self.fusion.selected_node_id == node_id:
            return NodeVisualState.SELECTED
        return NodeVisualState.NORMAL

    def _update_floating_toolbar(self) -> None:
        in_pending = self.fusion.mode in (
            InteractionMode.CONNECT_AWAITING_TARGET,
            InteractionMode.PENDING_CREATE,
        )
        self.btn_ctx_cancel.setVisible(in_pending)

        anchor_scene_pt: QPointF | None = None
        if self.fusion.selected_node_id and self.fusion.selected_node_id in self.diagram.nodes:
            node = self.diagram.nodes[self.fusion.selected_node_id]
            anchor_scene_pt = QPointF(node.x, node.y - node.height / 2.0 - 42.0)
            self.btn_ctx_connect.setVisible(True)
            self.btn_ctx_delete.setVisible(True)
        elif self.fusion.selected_edge_id and self.fusion.selected_edge_id in self.diagram.edges:
            edge = self.diagram.edges[self.fusion.selected_edge_id]
            src = self.diagram.nodes.get(edge.source_id)
            dst = self.diagram.nodes.get(edge.target_id)
            if src and dst:
                anchor_scene_pt = QPointF((src.x + dst.x) / 2.0, (src.y + dst.y) / 2.0 - 36.0)
            self.btn_ctx_connect.setVisible(False)
            self.btn_ctx_delete.setVisible(True)
        elif in_pending:
            anchor_scene_pt = QPointF(self.scene.width() / 2.0, 28.0)
            self.btn_ctx_connect.setVisible(False)
            self.btn_ctx_delete.setVisible(False)

        if anchor_scene_pt is None:
            self.floating_bar.hide()
            return

        self.floating_bar.adjustSize()
        vp_pt = self.view.mapFromScene(anchor_scene_pt)
        bar_w = self.floating_bar.width()
        bar_h = self.floating_bar.height()
        vp_w = self.view.viewport().width()
        vp_h = self.view.viewport().height()
        x = max(8, min(vp_w - bar_w - 8, int(vp_pt.x() - bar_w / 2)))
        y = max(8, min(vp_h - bar_h - 8, int(vp_pt.y() - bar_h / 2)))
        self.floating_bar.move(x, y)
        self.floating_bar.show()
        self.floating_bar.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        vp = self.view.viewport().size()
        if vp.width() > 200 and vp.height() > 200:
            self.scene.setSceneRect(QRectF(0, 0, vp.width(), vp.height()))
            self.fusion.set_canvas_size(vp.width(), vp.height())
            if self._camera_worker is not None:
                self._camera_worker.set_canvas_size(vp.width(), vp.height())
            self._update_floating_toolbar()

    # --- Transición a WF-05 ---

    def export_clean_png(self, target_path: Path) -> Path:
        hidden = [self._cursor_item]
        if self._pending_node_item is not None:
            hidden.append(self._pending_node_item)
        return export_scene_to_png(self.scene, target_path, hidden_items=hidden, scale=2.0)

    def open_close_modal(self) -> None:
        self.stop_workers()
        self.telemetry.finalize()
        filename = build_session_json_filename(self.config.participant_code)
        default_json_path = self.config.output_dir / filename
        doc = self.telemetry.build_session_document(
            diagram=self.diagram,
            session_filename=filename,
        )

        dialog = WF05CloseDialog(
            document=doc,
            default_json_path=default_json_path,
            png_exporter=self.export_clean_png,
            parent=self,
        )
        dialog.exec()
        self._session_closed = True
        self.close()

    def closeEvent(self, event) -> None:
        self.stop_workers()
        if not self._session_closed:
            self.telemetry.finalize()
        super().closeEvent(event)
