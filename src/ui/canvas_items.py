from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsObject

from src.config import CURSOR_COLORS
from src.models.diagram import NodeModel, NodeVisualState
from src.models.events import GestureState


class RoundedNodeItem(QGraphicsObject):
    """Nodo exclusivamente rectangular con bordes redondeados y 3 estados visuales claros.

    - NORMAL: borde negro sólido delgado.
    - VOICE_CREATED / PENDING_VOICE: borde verde segmentado resaltado + coordenadas (X, Y).
    - SELECTED / CONNECT_SOURCE: borde azul grueso segmentado (`dashed`).
    """

    def __init__(
        self,
        node_id: str,
        text: str,
        x: float,
        y: float,
        width: float = 190.0,
        height: float = 64.0,
        visual_state: NodeVisualState = NodeVisualState.NORMAL,
    ) -> None:
        super().__init__()
        self.node_id = node_id
        self.text = text
        self.width = width
        self.height = height
        self.visual_state = visual_state
        self.setPos(x, y)
        self.setZValue(10)

    def sync_from_model(self, model: NodeModel, visual_state: NodeVisualState) -> None:
        self.prepareGeometryChange()
        self.text = model.text
        self.width = model.width
        self.height = model.height
        self.visual_state = visual_state
        self.setPos(model.x, model.y)
        self.update()

    def boundingRect(self) -> QRectF:
        # Espacio extra abajo para la etiqueta de coordenadas (X, Y) en creación por voz
        return QRectF(
            -self.width / 2.0 - 8.0,
            -self.height / 2.0 - 20.0,
            self.width + 16.0,
            self.height + 46.0,
        )

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(-self.width / 2.0, -self.height / 2.0, self.width, self.height)
        path = QPainterPath()
        path.addRoundedRect(rect, 14.0, 14.0)

        # Estilo según estado visual normativo
        if self.visual_state in (
            NodeVisualState.PENDING_VOICE,
            NodeVisualState.VOICE_CREATED,
        ):
            fill = QColor("#F0FDF4")
            pen = QPen(QColor("#16A34A"), 2.8, Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])
        elif self.visual_state in (
            NodeVisualState.SELECTED,
            NodeVisualState.CONNECT_SOURCE,
        ):
            fill = QColor("#EFF6FF")
            pen = QPen(QColor("#2563EB"), 3.4, Qt.PenStyle.DashLine)
            pen.setDashPattern([7, 4])
        else:
            fill = QColor("#FFFFFF")
            pen = QPen(QColor("#111827"), 1.8, Qt.PenStyle.SolidLine)

        painter.setBrush(QBrush(fill))
        painter.setPen(pen)
        painter.drawPath(path)

        # Etiqueta textual centrada
        font = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#0F172A"))
        text_rect = rect.adjusted(10, 6, -10, -6)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self.text,
        )

        # Etiqueta (X, Y) durante creación / instanciación por voz
        if self.visual_state in (
            NodeVisualState.PENDING_VOICE,
            NodeVisualState.VOICE_CREATED,
        ):
            coord_text = f"({int(self.x())}, {int(self.y())})"
            if self.visual_state == NodeVisualState.PENDING_VOICE:
                coord_text = f"Confirmar en {coord_text}"
            badge_font = QFont("Consolas", 9, QFont.Weight.Bold)
            painter.setFont(badge_font)
            badge_rect = QRectF(-self.width / 2.0, self.height / 2.0 + 4.0, self.width, 18.0)
            painter.setPen(QColor("#15803D"))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, coord_text)

        # Insignia ORIGEN cuando se está acoplando un enlace
        if self.visual_state == NodeVisualState.CONNECT_SOURCE:
            chip_rect = QRectF(-44.0, -self.height / 2.0 - 18.0, 88.0, 16.0)
            chip_path = QPainterPath()
            chip_path.addRoundedRect(chip_rect, 6.0, 6.0)
            painter.fillPath(chip_path, QColor("#2563EB"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, "ORIGEN")

        painter.restore()


class DashedEdgeItem(QGraphicsLineItem):
    """Enlace visual como línea recta segmentada entre dos nodos."""

    def __init__(self, edge_id: str) -> None:
        super().__init__()
        self.edge_id = edge_id
        self.selected_visual = False
        self.setZValue(2)
        self._apply_pen()

    def set_selected_visual(self, selected: bool) -> None:
        self.selected_visual = selected
        self._apply_pen()

    def _apply_pen(self) -> None:
        if self.selected_visual:
            pen = QPen(QColor("#2563EB"), 3.4, Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 4])
        else:
            pen = QPen(QColor("#334155"), 2.4, Qt.PenStyle.DashLine)
            pen.setDashPattern([7, 5])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

    def update_endpoints(self, src: NodeModel, dst: NodeModel) -> None:
        """Recorta la línea a los bordes de ambos rectángulos para máxima legibilidad."""
        p1 = _rect_edge_anchor(src.x, src.y, src.width, src.height, dst.x, dst.y)
        p2 = _rect_edge_anchor(dst.x, dst.y, dst.width, dst.height, src.x, src.y)
        self.setLine(p1.x(), p1.y(), p2.x(), p2.y())


def _rect_edge_anchor(
    cx: float, cy: float, w: float, h: float, tx: float, ty: float
) -> QPointF:
    dx = tx - cx
    dy = ty - cy
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return QPointF(cx, cy)
    half_w = w / 2.0 + 3.0
    half_h = h / 2.0 + 3.0
    scale_x = half_w / abs(dx) if abs(dx) > 1e-4 else float("inf")
    scale_y = half_h / abs(dy) if abs(dy) > 1e-4 else float("inf")
    t = min(scale_x, scale_y)
    return QPointF(cx + dx * t, cy + dy * t)


class VirtualCursorItem(QGraphicsObject):
    """Cursor virtual circular cinemático con los 4 estados de color y anillo Dwell Time."""

    def __init__(self) -> None:
        super().__init__()
        self._state: GestureState = GestureState.HAND_LOST
        self._dwell_progress: float = 0.0
        self.setZValue(100)
        self.setVisible(False)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def update_cursor(
        self,
        x: float | None,
        y: float | None,
        state: GestureState,
        dwell_progress: float,
    ) -> None:
        self._state = state
        self._dwell_progress = max(0.0, min(1.0, dwell_progress))
        if x is None or y is None or state == GestureState.HAND_LOST:
            self.setVisible(False)
            return
        self.setVisible(True)
        self.setPos(x, y)
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-26.0, -26.0, 52.0, 52.0)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        color_hex = CURSOR_COLORS.get(self._state.value, "#64748B")
        base_color = QColor(color_hex)

        # Halo semitransparente
        halo = QColor(base_color)
        halo.setAlpha(45)
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), 18.0, 18.0)

        # Núcleo circular
        painter.setBrush(QBrush(base_color))
        painter.setPen(QPen(QColor("#FFFFFF"), 2.0))
        painter.drawEllipse(QPointF(0, 0), 10.0, 10.0)

        # Anillo de progreso Dwell Time (250 ms)
        if self._dwell_progress > 0.02:
            ring_pen = QPen(base_color, 3.5)
            ring_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring_rect = QRectF(-21.0, -21.0, 42.0, 42.0)
            span_angle = int(-360.0 * self._dwell_progress * 16.0)
            painter.drawArc(ring_rect, 90 * 16, span_angle)

        painter.restore()
