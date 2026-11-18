from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import CalibrationConfig
from src.models.events import GestureState


@dataclass(frozen=True)
class Landmark2D:
    """Punto de referencia normalizado [0..1] de los 21 landmarks de MediaPipe."""

    x: float
    y: float
    z: float = 0.0


@dataclass
class ClassificationResult:
    """Salida geométrica de un fotograma antes del filtro temporal Dwell."""

    raw_gesture: GestureState
    cursor_norm_x: float
    cursor_norm_y: float
    pinch_distance_px: float
    effective_threshold_px: float
    hand_scale_px: float


class GestureClassifier:
    """Clasificador geométrico de los 4 gestos canónicos a partir de 21 landmarks.

    - Azul (`MOVE`): dedo índice extendido, resto mayormente doblado.
    - Morado (`PINCH_SELECT`): distancia pulgar (4) - índice (8) < umbral adaptado por escala
      con histéresis de liberación para evitar parpadeo.
    - Naranja (`FIST_DRAG`): puño cerrado (los 4 dedos principales doblados).
    - Verde (`OPEN_DROP`): mano abierta (4-5 dedos extendidos).
    """

    # Índices oficiales de MediaPipe Hands
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20

    def __init__(
        self,
        calibration: CalibrationConfig,
        roi_margin: float = 0.08,
    ) -> None:
        self.calibration = calibration
        self.roi_margin = roi_margin
        self._in_pinch_hysteresis: bool = False

    def reset(self) -> None:
        self._in_pinch_hysteresis = False

    def update_calibration(self, calibration: CalibrationConfig) -> None:
        self.calibration = calibration

    @staticmethod
    def _dist_px(a: Landmark2D, b: Landmark2D, width: int, height: int) -> float:
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)

    def _map_roi(self, value: float) -> float:
        """Mapea [roi_margin .. 1 - roi_margin] a [0 .. 1] para alcanzar bordes del lienzo."""
        span = max(1e-4, 1.0 - 2.0 * self.roi_margin)
        mapped = (value - self.roi_margin) / span
        return max(0.0, min(1.0, mapped))

    def _is_finger_extended(
        self,
        lm: list[Landmark2D],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
        width: int,
        height: int,
    ) -> bool:
        """Un dedo está extendido si la punta está claramente más lejos de la muñeca que la articulación PIP."""
        wrist = lm[self.WRIST]
        d_tip = self._dist_px(wrist, lm[tip_idx], width, height)
        d_pip = self._dist_px(wrist, lm[pip_idx], width, height)
        d_mcp = self._dist_px(wrist, lm[mcp_idx], width, height)
        return d_tip > d_pip * 1.08 and d_tip > d_mcp * 1.25

    def _is_finger_curled(
        self,
        lm: list[Landmark2D],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
        width: int,
        height: int,
    ) -> bool:
        wrist = lm[self.WRIST]
        d_tip = self._dist_px(wrist, lm[tip_idx], width, height)
        d_pip = self._dist_px(wrist, lm[pip_idx], width, height)
        return d_tip < d_pip * 1.02

    def _is_thumb_open(self, lm: list[Landmark2D], width: int, height: int) -> bool:
        """Pulgar separado de la palma (distancia punta pulgar a MCP índice/meñique)."""
        d_thumb_index_mcp = self._dist_px(
            lm[self.THUMB_TIP], lm[self.INDEX_MCP], width, height
        )
        d_palm = max(
            1.0,
            self._dist_px(lm[self.INDEX_MCP], lm[self.PINKY_MCP], width, height),
        )
        return (d_thumb_index_mcp / d_palm) > 0.55

    def classify(
        self,
        landmarks: list[Landmark2D],
        frame_width: int,
        frame_height: int,
    ) -> ClassificationResult:
        if len(landmarks) < 21:
            self._in_pinch_hysteresis = False
            return ClassificationResult(
                raw_gesture=GestureState.HAND_LOST,
                cursor_norm_x=0.5,
                cursor_norm_y=0.5,
                pinch_distance_px=999.0,
                effective_threshold_px=self.calibration.pinch_threshold_px,
                hand_scale_px=0.0,
            )

        # Escala de mano (muñeca -> nudillo medio) para compensar distancia a cámara
        hand_scale_px = max(
            20.0,
            self._dist_px(
                landmarks[self.WRIST], landmarks[self.MIDDLE_MCP], frame_width, frame_height
            ),
        )
        ref_scale = max(20.0, self.calibration.hand_scale_ref_px)
        scale_factor = max(0.55, min(1.85, hand_scale_px / ref_scale))
        effective_threshold = self.calibration.pinch_threshold_px * scale_factor

        pinch_dist = self._dist_px(
            landmarks[self.THUMB_TIP], landmarks[self.INDEX_TIP], frame_width, frame_height
        )

        # Estado de cada dedo
        index_ext = self._is_finger_extended(
            landmarks, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP, frame_width, frame_height
        )
        middle_ext = self._is_finger_extended(
            landmarks, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP, frame_width, frame_height
        )
        ring_ext = self._is_finger_extended(
            landmarks, self.RING_MCP, self.RING_PIP, self.RING_TIP, frame_width, frame_height
        )
        pinky_ext = self._is_finger_extended(
            landmarks, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP, frame_width, frame_height
        )

        index_curled = self._is_finger_curled(
            landmarks, self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP, frame_width, frame_height
        )
        middle_curled = self._is_finger_curled(
            landmarks, self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP, frame_width, frame_height
        )
        ring_curled = self._is_finger_curled(
            landmarks, self.RING_MCP, self.RING_PIP, self.RING_TIP, frame_width, frame_height
        )
        pinky_curled = self._is_finger_curled(
            landmarks, self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP, frame_width, frame_height
        )
        thumb_open = self._is_thumb_open(landmarks, frame_width, frame_height)

        extended_four = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        curled_four = sum([index_curled, middle_curled, ring_curled, pinky_curled])

        # Histéresis de pellizco: entrar < umbral, salir > umbral * factor
        pinch_release_threshold = (
            effective_threshold * self.calibration.pinch_hysteresis_factor
        )

        # 1. Puño cerrado (Naranja): los 4 dedos doblados (prioridad antes de pellizco falso en puño)
        if curled_four == 4 and not thumb_open:
            self._in_pinch_hysteresis = False
            raw_gesture = GestureState.FIST_DRAG
            # Usar centro de la palma (LM 9) para estabilidad al arrastrar con puño
            cursor_x = self._map_roi(landmarks[self.MIDDLE_MCP].x)
            cursor_y = self._map_roi(landmarks[self.MIDDLE_MCP].y)

        # 2. Pellizco pulgar-índice (Morado): distancia < umbral (no en puño)
        elif (
            (not self._in_pinch_hysteresis and pinch_dist < effective_threshold)
            or (self._in_pinch_hysteresis and pinch_dist < pinch_release_threshold)
        ):
            self._in_pinch_hysteresis = True
            raw_gesture = GestureState.PINCH_SELECT
            # Punto medio entre pulgar e índice evita salto al cerrar los dedos
            mid_x = 0.5 * (landmarks[self.THUMB_TIP].x + landmarks[self.INDEX_TIP].x)
            mid_y = 0.5 * (landmarks[self.THUMB_TIP].y + landmarks[self.INDEX_TIP].y)
            cursor_x = self._map_roi(mid_x)
            cursor_y = self._map_roi(mid_y)

        else:
            self._in_pinch_hysteresis = False
            # 3. Mano abierta (Verde): 4 dedos extendidos + pulgar abierto
            if extended_four == 4 and thumb_open:
                raw_gesture = GestureState.OPEN_DROP
                cursor_x = self._map_roi(landmarks[self.MIDDLE_MCP].x)
                cursor_y = self._map_roi(landmarks[self.MIDDLE_MCP].y)
            # 4. Dedo índice extendido (Azul): mover cursor
            elif index_ext and not (middle_ext and ring_ext and pinky_ext):
                raw_gesture = GestureState.MOVE
                cursor_x = self._map_roi(landmarks[self.INDEX_TIP].x)
                cursor_y = self._map_roi(landmarks[self.INDEX_TIP].y)
            # 5. Puño parcial/firme con pulgar lateral
            elif curled_four == 4:
                raw_gesture = GestureState.FIST_DRAG
                cursor_x = self._map_roi(landmarks[self.MIDDLE_MCP].x)
                cursor_y = self._map_roi(landmarks[self.MIDDLE_MCP].y)
            else:
                raw_gesture = GestureState.IDLE
                cursor_x = self._map_roi(landmarks[self.INDEX_TIP].x)
                cursor_y = self._map_roi(landmarks[self.INDEX_TIP].y)

        return ClassificationResult(
            raw_gesture=raw_gesture,
            cursor_norm_x=cursor_x,
            cursor_norm_y=cursor_y,
            pinch_distance_px=pinch_dist,
            effective_threshold_px=effective_threshold,
            hand_scale_px=hand_scale_px,
        )
