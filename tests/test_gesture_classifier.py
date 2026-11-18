from __future__ import annotations

from src.config import CalibrationConfig
from src.models.events import GestureEventType, GestureState
from src.vision.filters import CursorSmoother2D, DwellStateMachine
from src.vision.gesture_classifier import GestureClassifier, Landmark2D


def _make_hand(
    index_ext: bool = False,
    middle_ext: bool = False,
    ring_ext: bool = False,
    pinky_ext: bool = False,
    thumb_open: bool = False,
    pinch_close: bool = False,
    scale: float = 1.0,
) -> list[Landmark2D]:
    """Construye una mano sintética de 21 landmarks con geometría realista."""
    cx, cy = 0.5, 0.72
    s = scale
    pts = [Landmark2D(cx, cy)] * 21

    # Muñeca
    pts[0] = Landmark2D(cx, cy)

    # Dedo índice (5 MCP, 6 PIP, 7 DIP, 8 TIP)
    pts[5] = Landmark2D(cx - 0.06 * s, cy - 0.16 * s)
    pts[6] = Landmark2D(cx - 0.06 * s, cy - 0.23 * s)
    pts[7] = Landmark2D(cx - 0.06 * s, cy - (0.28 if index_ext else 0.19) * s)
    pts[8] = Landmark2D(cx - 0.06 * s, cy - (0.34 if index_ext else 0.15) * s)

    # Dedo medio (9..12)
    pts[9] = Landmark2D(cx - 0.01 * s, cy - 0.18 * s)
    pts[10] = Landmark2D(cx - 0.01 * s, cy - 0.25 * s)
    pts[11] = Landmark2D(cx - 0.01 * s, cy - (0.31 if middle_ext else 0.20) * s)
    pts[12] = Landmark2D(cx - 0.01 * s, cy - (0.37 if middle_ext else 0.16) * s)

    # Dedo anular (13..16)
    pts[13] = Landmark2D(cx + 0.04 * s, cy - 0.17 * s)
    pts[14] = Landmark2D(cx + 0.04 * s, cy - 0.24 * s)
    pts[15] = Landmark2D(cx + 0.04 * s, cy - (0.29 if ring_ext else 0.19) * s)
    pts[16] = Landmark2D(cx + 0.04 * s, cy - (0.35 if ring_ext else 0.15) * s)

    # Dedo meñique (17..20)
    pts[17] = Landmark2D(cx + 0.09 * s, cy - 0.15 * s)
    pts[18] = Landmark2D(cx + 0.09 * s, cy - 0.21 * s)
    pts[19] = Landmark2D(cx + 0.09 * s, cy - (0.26 if pinky_ext else 0.17) * s)
    pts[20] = Landmark2D(cx + 0.09 * s, cy - (0.31 if pinky_ext else 0.13) * s)

    # Pulgar (1..4)
    pts[1] = Landmark2D(cx - 0.07 * s, cy - 0.05 * s)
    pts[2] = Landmark2D(cx - 0.10 * s, cy - 0.10 * s)
    pts[3] = Landmark2D(cx - 0.12 * s, cy - 0.14 * s)
    if pinch_close:
        # Punta del pulgar tocando la punta del índice (distancia ~6 px en 640x480)
        pts[8] = Landmark2D(cx - 0.08 * s, cy - 0.24 * s)
        pts[4] = Landmark2D(cx - 0.085 * s, cy - 0.235 * s)
    elif thumb_open:
        pts[4] = Landmark2D(cx - 0.19 * s, cy - 0.17 * s)
    else:
        pts[4] = Landmark2D(cx - 0.04 * s, cy - 0.16 * s)

    return pts


def test_classify_four_canonical_gestures() -> None:
    cal = CalibrationConfig(pinch_threshold_px=30.0, hand_scale_ref_px=90.0)
    clf = GestureClassifier(cal)

    # 1. Azul: índice extendido -> MOVE
    move_lm = _make_hand(index_ext=True)
    res_move = clf.classify(move_lm, 640, 480)
    assert res_move.raw_gesture == GestureState.MOVE

    # 2. Morado: pellizco -> PINCH_SELECT
    pinch_lm = _make_hand(index_ext=True, middle_ext=True, pinch_close=True)
    res_pinch = clf.classify(pinch_lm, 640, 480)
    assert res_pinch.raw_gesture == GestureState.PINCH_SELECT
    assert res_pinch.pinch_distance_px < res_pinch.effective_threshold_px

    # 3. Naranja: puño cerrado -> FIST_DRAG
    clf.reset()
    fist_lm = _make_hand(index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False, thumb_open=False)
    res_fist = clf.classify(fist_lm, 640, 480)
    assert res_fist.raw_gesture == GestureState.FIST_DRAG

    # 4. Verde: mano abierta -> OPEN_DROP
    open_lm = _make_hand(index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True, thumb_open=True)
    res_open = clf.classify(open_lm, 640, 480)
    assert res_open.raw_gesture == GestureState.OPEN_DROP


def test_dwell_time_250ms_edge_triggered_and_no_false_positives() -> None:
    dm = DwellStateMachine(dwell_time_ms=250, jitter_tolerance_ms=60, hand_grace_period_ms=300)

    # Mano entra en MOVE
    st, ev, prog = dm.update(GestureState.MOVE, 1000)
    assert st == GestureState.MOVE
    assert ev == GestureEventType.HAND_RECOVERED

    # Pellizco breve (< 250 ms) NO debe confirmar selección
    st, ev, prog = dm.update(GestureState.PINCH_SELECT, 1100)
    assert st == GestureState.PINCH_SELECT
    assert ev == GestureEventType.NONE
    assert prog == 0.0

    st, ev, prog = dm.update(GestureState.PINCH_SELECT, 1250)
    assert ev == GestureEventType.NONE
    assert 0.5 < prog < 0.7

    # Soltar antes de 250 ms cancela sin disparar
    dm.update(GestureState.MOVE, 1320)
    st, ev, _ = dm.update(GestureState.MOVE, 1400)
    assert st == GestureState.MOVE
    assert ev == GestureEventType.NONE

    # Nuevo pellizco sostenido >= 250 ms dispara EXACTAMENTE UNA VEZ
    dm.update(GestureState.PINCH_SELECT, 2000)
    _, ev_mid, _ = dm.update(GestureState.PINCH_SELECT, 2200)
    assert ev_mid == GestureEventType.NONE

    st_ok, ev_ok, prog_ok = dm.update(GestureState.PINCH_SELECT, 2260)
    assert st_ok == GestureState.PINCH_SELECT
    assert ev_ok == GestureEventType.SELECT_CONFIRMED
    assert prog_ok == 1.0

    # Frames posteriores mientras sigue el pellizco NO re-disparan
    _, ev_repeat, _ = dm.update(GestureState.PINCH_SELECT, 2320)
    assert ev_repeat == GestureEventType.NONE


def test_dwell_hand_loss_grace_period() -> None:
    dm = DwellStateMachine(dwell_time_ms=250, hand_grace_period_ms=300)
    dm.update(GestureState.MOVE, 1000)

    # Pérdida fugaz de 100 ms (< 300 ms) se absorbe sin emitir HAND_LOST
    st, ev, _ = dm.update(GestureState.HAND_LOST, 1100)
    assert st == GestureState.MOVE
    assert ev == GestureEventType.NONE

    # Pérdida prolongada (>= 300 ms) sí emite HAND_LOST
    st, ev, _ = dm.update(GestureState.HAND_LOST, 1450)
    assert st == GestureState.HAND_LOST
    assert ev == GestureEventType.HAND_LOST


def test_cursor_smoother_converges_without_integer_truncation() -> None:
    smoother = CursorSmoother2D()
    x, y = smoother.filter(100.0, 200.0, 1000)
    assert (x, y) == (100.0, 200.0)
    for step in range(1, 20):
        x, y = smoother.filter(110.0, 200.0, 1000 + step * 33)
    assert abs(x - 110.0) < 0.2
    assert abs(y - 200.0) < 0.01
