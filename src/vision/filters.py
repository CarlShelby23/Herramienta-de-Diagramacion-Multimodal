from __future__ import annotations

import math

from src.models.events import GestureEventType, GestureState


class OneEuroFilter1D:
    """Filtro One-Euro para suavizado adaptativo de cursor sin retardo perceptible."""

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.04,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev_ms: int | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev_ms = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def filter(self, x: float, timestamp_ms: int) -> float:
        if self._x_prev is None or self._t_prev_ms is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev_ms = timestamp_ms
            return x

        dt = max((timestamp_ms - self._t_prev_ms) / 1000.0, 1.0 / 240.0)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev_ms = timestamp_ms
        return x_hat


class CursorSmoother2D:
    """Suavizador 2D con reinicio seguro ante pérdida de mano."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.04) -> None:
        self._fx = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)
        self._fy = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def filter(self, x: float, y: float, timestamp_ms: int) -> tuple[float, float]:
        return (
            self._fx.filter(x, timestamp_ms),
            self._fy.filter(y, timestamp_ms),
        )


class DwellStateMachine:
    """Máquina de estados con Dwell Time (250 ms), disparo de flanco único y tolerancia a jitter.

    Garantiza:
    - Ninguna acción de selección/arrastre/drop se dispara por un gesto fugaz (< dwell_time_ms).
    - Cada gesto confirmado dispara un único evento de flanco (evita 60 clics/s).
    - Microcortes de 1-2 frames (<= jitter_tolerance_ms) no reinician injustamente el dwell.
    - Pérdidas de mano se amortiguan durante hand_grace_period_ms antes de cancelar gestos.
    """

    _DWELL_GESTURES = {
        GestureState.PINCH_SELECT,
        GestureState.FIST_DRAG,
        GestureState.OPEN_DROP,
    }

    def __init__(
        self,
        dwell_time_ms: int = 250,
        jitter_tolerance_ms: int = 65,
        hand_grace_period_ms: int = 300,
    ) -> None:
        self.dwell_time_ms = max(50, int(dwell_time_ms))
        self.jitter_tolerance_ms = max(0, int(jitter_tolerance_ms))
        self.hand_grace_period_ms = max(50, int(hand_grace_period_ms))

        self.stable_gesture: GestureState = GestureState.HAND_LOST
        self._candidate: GestureState | None = None
        self._candidate_start_ms: int = 0
        self._candidate_last_seen_ms: int = 0
        self._confirmed_for_hold: bool = False
        self._hand_lost_since_ms: int | None = None
        self._was_hand_lost: bool = True

    def update_dwell_time(self, dwell_time_ms: int) -> None:
        self.dwell_time_ms = max(50, int(dwell_time_ms))

    def reset(self) -> None:
        self.stable_gesture = GestureState.HAND_LOST
        self._candidate = None
        self._candidate_start_ms = 0
        self._candidate_last_seen_ms = 0
        self._confirmed_for_hold = False
        self._hand_lost_since_ms = None
        self._was_hand_lost = True

    def update(
        self, raw_gesture: GestureState, timestamp_ms: int
    ) -> tuple[GestureState, GestureEventType, float]:
        """Procesa un frame gestual y retorna (estado_estable, evento_flanco, progreso_dwell)."""
        if raw_gesture == GestureState.HAND_LOST:
            if self._hand_lost_since_ms is None:
                self._hand_lost_since_ms = timestamp_ms
            elapsed_lost = timestamp_ms - self._hand_lost_since_ms
            if elapsed_lost < self.hand_grace_period_ms and not self._was_hand_lost:
                # Breve oclusión: mantener estado anterior sin disparar nuevos eventos
                return self.stable_gesture, GestureEventType.NONE, 0.0

            event = (
                GestureEventType.HAND_LOST
                if not self._was_hand_lost
                else GestureEventType.NONE
            )
            self._was_hand_lost = True
            self.stable_gesture = GestureState.HAND_LOST
            self._candidate = None
            self._confirmed_for_hold = False
            return self.stable_gesture, event, 0.0

        # Mano presente
        self._hand_lost_since_ms = None
        if self._was_hand_lost:
            self._was_hand_lost = False
            self._candidate = None
            self._confirmed_for_hold = False
            if raw_gesture in (GestureState.MOVE, GestureState.IDLE):
                self.stable_gesture = raw_gesture
                return self.stable_gesture, GestureEventType.HAND_RECOVERED, 0.0

        # Gestos con Dwell Time (Morado / Naranja / Verde)
        if raw_gesture in self._DWELL_GESTURES:
            if self._candidate == raw_gesture:
                self._candidate_last_seen_ms = timestamp_ms
            else:
                self._candidate = raw_gesture
                self._candidate_start_ms = timestamp_ms
                self._candidate_last_seen_ms = timestamp_ms
                self._confirmed_for_hold = False

            elapsed = timestamp_ms - self._candidate_start_ms
            progress = min(1.0, elapsed / float(self.dwell_time_ms))
            self.stable_gesture = raw_gesture

            if elapsed >= self.dwell_time_ms and not self._confirmed_for_hold:
                self._confirmed_for_hold = True
                return self.stable_gesture, self._event_for_gesture(raw_gesture), 1.0

            return self.stable_gesture, GestureEventType.NONE, progress

        # raw_gesture es MOVE o IDLE
        if self._candidate is not None:
            since_candidate = timestamp_ms - self._candidate_last_seen_ms
            if since_candidate <= self.jitter_tolerance_ms:
                # Tolerar 1-2 frames ruidosos sin romper el gesto en progreso
                progress = (
                    1.0
                    if self._confirmed_for_hold
                    else min(
                        1.0,
                        (timestamp_ms - self._candidate_start_ms)
                        / float(self.dwell_time_ms),
                    )
                )
                return self.stable_gesture, GestureEventType.NONE, progress

            # Se soltó el gesto definitivamente
            self._candidate = None
            self._confirmed_for_hold = False

        self.stable_gesture = raw_gesture
        return self.stable_gesture, GestureEventType.NONE, 0.0

    @staticmethod
    def _event_for_gesture(gesture: GestureState) -> GestureEventType:
        if gesture == GestureState.PINCH_SELECT:
            return GestureEventType.SELECT_CONFIRMED
        if gesture == GestureState.FIST_DRAG:
            return GestureEventType.DRAG_START
        if gesture == GestureState.OPEN_DROP:
            return GestureEventType.DROP_CONFIRMED
        return GestureEventType.NONE
