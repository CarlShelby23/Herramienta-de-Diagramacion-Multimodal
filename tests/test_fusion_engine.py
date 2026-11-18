from __future__ import annotations

from src.audio.command_parser import parse_voice_command
from src.config import ConditionLevel, FusionConfig, SessionConfig
from src.fusion.fusion_engine import InteractionMode, MultimodalFusionEngine
from src.models.diagram import DiagramState
from src.models.events import AsrEvent, GestureEventType, GestureFrame, GestureState
from src.telemetry.tracker import TelemetryTracker


def _asr(text: str, t_end: int, sys_lat: int = 80) -> AsrEvent:
    parsed = parse_voice_command(text)
    return AsrEvent(
        raw_text=text,
        intent=parsed.intent,
        concept=parsed.concept,
        t_speech_start_ms=t_end - 400,
        t_speech_end_ms=t_end,
        t_asr_ready_ms=t_end + sys_lat,
        system_latency_ms=sys_lat,
        error_message=None if parsed.intent.name not in ("UNRECOGNIZED", "AMBIGUOUS") else parsed.reason,
    )


def _gesture(
    ts: int,
    x: float,
    y: float,
    state: GestureState = GestureState.MOVE,
    event: GestureEventType = GestureEventType.NONE,
    hand_detected: bool = True,
) -> GestureFrame:
    return GestureFrame(
        timestamp_ms=ts,
        raw_gesture=state,
        stable_gesture=state,
        event=event,
        cursor_x=x if hand_detected else None,
        cursor_y=y if hand_detected else None,
        hand_detected=hand_detected,
        landmarks_count=21 if hand_detected else 0,
        camera_ok=True,
    )


def _setup_engine(start_ms: int = 10_000):
    cfg = SessionConfig(
        participant_code="PART-2026-014",
        condition=ConditionLevel.MEDIUM,
        fusion=FusionConfig(
            create_confirm_window_ms=5000,
            connect_target_window_ms=8000,
            inactivity_threshold_ms=3000,
            require_gesture_confirm_on_create=True,
        ),
    )
    diagram = DiagramState()
    tracker = TelemetryTracker(cfg, start_time_ms=start_ms)
    engine = MultimodalFusionEngine(diagram, tracker, cfg.fusion)
    return engine, diagram, tracker


def test_voice_create_confirm_and_user_latency_separation() -> None:
    engine, diagram, tracker = _setup_engine(10_000)

    # Posicionar cursor en (300, 250)
    engine.process_gesture(_gesture(10_500, 300, 250, GestureState.MOVE))

    # Decir "Crear nodo Atención" (termina de hablar en 11_000, ASR listo en 11_090 -> sys=90 ms)
    upd = engine.process_asr(_asr("Crear nodo Atención", t_end=11_000, sys_lat=90))
    assert engine.mode == InteractionMode.PENDING_CREATE
    assert len(diagram.nodes) == 0  # Aún no instanciado hasta confirmar

    # Confirmar con mano abierta (Verde - DROP) en 11_850 -> latencia usuario = 850 ms
    upd = engine.process_gesture(
        _gesture(11_850, 310, 260, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED)
    )
    assert engine.mode == InteractionMode.READY
    assert len(diagram.nodes) == 1
    node = next(iter(diagram.nodes.values()))
    assert node.text == "Atención"
    assert upd.diagram_changed

    kpis = tracker.compute_kpis(now_ms=12_000)
    assert kpis.structural_modifications_total == 1
    assert kpis.nodes_created_count == 1
    assert kpis.user_latency_samples_ms == [850]
    assert kpis.system_asr_latency_samples_ms == [90]
    assert kpis.interruptions_total == 0


def test_voice_create_timeout_counts_as_interruption_not_latency() -> None:
    engine, diagram, tracker = _setup_engine(10_000)
    engine.process_gesture(_gesture(10_200, 400, 300))
    engine.process_asr(_asr("Crear nodo Memoria", t_end=10_500, sys_lat=60))
    assert engine.mode == InteractionMode.PENDING_CREATE

    # Avanzar más allá de la ventana (5000 ms) sin confirmación gestual
    upd = engine.tick(16_000)
    assert engine.mode == InteractionMode.READY
    assert len(diagram.nodes) == 0
    assert "expiró" in upd.feedback_message.lower()

    kpis = tracker.compute_kpis(now_ms=16_000)
    assert kpis.timeout_abandons == 1
    assert kpis.interruptions_total == 1
    assert kpis.user_latency_samples_ms == []  # No contamina latencia usuario


def test_drag_moves_node_never_connects_accidentally_and_is_not_structural() -> None:
    engine, diagram, tracker = _setup_engine(10_000)

    # Crear 2 nodos
    engine.process_gesture(_gesture(10_100, 250, 250))
    engine.process_asr(_asr("Crear nodo Entrada", t_end=10_200, sys_lat=50))
    engine.process_gesture(_gesture(10_600, 250, 250, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))

    engine.process_gesture(_gesture(11_000, 650, 250))
    engine.process_asr(_asr("Crear nodo Salida", t_end=11_100, sys_lat=50))
    engine.process_gesture(_gesture(11_500, 650, 250, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))

    assert len(diagram.nodes) == 2
    n1 = list(diagram.nodes.values())[0]

    # Iniciar arrastre con puño sobre n1 y soltar ENCIMA de n2 (650, 250)
    engine.process_gesture(_gesture(12_000, 250, 250, GestureState.FIST_DRAG, GestureEventType.DRAG_START))
    assert engine.mode == InteractionMode.DRAGGING

    engine.process_gesture(_gesture(12_300, 650, 250, GestureState.FIST_DRAG))
    engine.process_gesture(_gesture(12_700, 650, 250, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))

    assert engine.mode == InteractionMode.READY
    assert (n1.x, n1.y) == (650.0, 250.0)
    assert len(diagram.edges) == 0  # NUNCA debe conectar por arrastrar/soltar

    kpis = tracker.compute_kpis(now_ms=12_800)
    assert kpis.structural_modifications_total == 2  # Solo los 2 nodos creados
    assert kpis.non_structural_moves_count == 1


def test_explicit_connect_delete_and_undo_restores_edges() -> None:
    engine, diagram, tracker = _setup_engine(10_000)

    # Intentar "Conectar" sin nada seleccionado -> error sin acción estructural
    upd_err = engine.process_asr(_asr("Conectar", t_end=10_100, sys_lat=40))
    assert upd_err.feedback_level == "warning"
    assert len(diagram.edges) == 0

    # Crear dos nodos
    engine.process_gesture(_gesture(10_300, 250, 250))
    engine.process_asr(_asr("Crear nodo Origen", t_end=10_400, sys_lat=40))
    engine.process_gesture(_gesture(10_800, 250, 250, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))

    engine.process_gesture(_gesture(11_100, 700, 250))
    engine.process_asr(_asr("Crear nodo Destino", t_end=11_200, sys_lat=40))
    engine.process_gesture(_gesture(11_600, 700, 250, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))

    # Seleccionar Origen (pellizco 250 ms)
    engine.process_gesture(_gesture(12_000, 250, 250, GestureState.PINCH_SELECT, GestureEventType.SELECT_CONFIRMED))
    # Decir "Conectar"
    engine.process_asr(_asr("Conectar", t_end=12_200, sys_lat=50))
    assert engine.mode == InteractionMode.CONNECT_AWAITING_TARGET

    # Confirmar Destino con pellizco en (700, 250) -> latencia usuario = 700 ms
    engine.process_gesture(_gesture(12_900, 700, 250, GestureState.PINCH_SELECT, GestureEventType.SELECT_CONFIRMED))
    assert engine.mode == InteractionMode.READY
    assert len(diagram.edges) == 1

    # Borrar nodo Destino (quedó seleccionado) -> borra nodo y su enlace incidente
    engine.process_asr(_asr("Eliminar", t_end=13_200, sys_lat=40))
    assert len(diagram.nodes) == 1
    assert len(diagram.edges) == 0

    # Deshacer -> restaura tanto el nodo Destino como su enlace
    engine.process_asr(_asr("Deshacer", t_end=13_500, sys_lat=40))
    assert len(diagram.nodes) == 2
    assert len(diagram.edges) == 1

    kpis = tracker.compute_kpis(now_ms=13_600)
    assert kpis.structural_modifications_total == 4  # 2 create node + 1 create edge + 1 delete node
    assert kpis.recognition_errors == 1  # El "Conectar" inicial sin origen


def test_hand_lost_during_drag_reverts_cleanly() -> None:
    engine, diagram, tracker = _setup_engine(10_000)

    engine.process_gesture(_gesture(10_100, 300, 300))
    engine.process_asr(_asr("Crear nodo Sensor", t_end=10_200, sys_lat=40))
    engine.process_gesture(_gesture(10_500, 300, 300, GestureState.OPEN_DROP, GestureEventType.DROP_CONFIRMED))
    node = next(iter(diagram.nodes.values()))

    engine.process_gesture(_gesture(11_000, 300, 300, GestureState.FIST_DRAG, GestureEventType.DRAG_START))
    engine.process_gesture(_gesture(11_200, 500, 400, GestureState.FIST_DRAG))
    assert (node.x, node.y) == (500.0, 400.0)

    # Pérdida de mano por > 1000 ms aborta arrastre y restaura (300, 300)
    engine.process_gesture(_gesture(11_300, 0, 0, GestureState.HAND_LOST, hand_detected=False))
    engine.process_gesture(_gesture(12_500, 0, 0, GestureState.HAND_LOST, hand_detected=False))

    assert engine.mode == InteractionMode.READY
    assert (node.x, node.y) == (300.0, 300.0)
    assert tracker.compute_kpis(now_ms=12_600).timeout_abandons == 1
