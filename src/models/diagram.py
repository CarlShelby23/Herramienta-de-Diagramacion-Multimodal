from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class NodeVisualState(str, Enum):
    """Estados visuales estrictos de los nodos (rectángulos con bordes redondeados)."""

    NORMAL = "NORMAL"                # Borde negro sólido delgado
    PENDING_VOICE = "PENDING_VOICE"  # Previsualización por voz: verde segmentado + (X, Y)
    VOICE_CREATED = "VOICE_CREATED"  # Recién instanciado por voz: verde segmentado + (X, Y)
    SELECTED = "SELECTED"            # Seleccionado: azul grueso segmentado
    CONNECT_SOURCE = "CONNECT_SOURCE"  # Origen en acople: azul grueso segmentado + etiqueta


@dataclass
class NodeModel:
    """Representación topológica de un nodo del diagrama."""

    id: str
    text: str
    x: float
    y: float
    width: float = 190.0
    height: float = 64.0

    def contains(self, px: float, py: float, margin: float = 10.0) -> bool:
        half_w = (self.width / 2.0) + margin
        half_h = (self.height / 2.0) + margin
        return abs(self.x - px) <= half_w and abs(self.y - py) <= half_h

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "texto": self.text,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
        }


@dataclass
class EdgeModel:
    """Representación topológica de un enlace entre dos nodos."""

    id: str
    source_id: str
    target_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origen_id": self.source_id,
            "destino_id": self.target_id,
        }


class DiagramCommand(ABC):
    """Comando reversible para soportar 'Deshacer' de forma consistente."""

    is_structural: bool = True
    action_name: str = "COMMAND"

    @abstractmethod
    def execute(self, state: DiagramState) -> None:
        ...

    @abstractmethod
    def undo(self, state: DiagramState) -> None:
        ...


class CreateNodeCommand(DiagramCommand):
    is_structural = True
    action_name = "NODE_CREATE"

    def __init__(self, node: NodeModel) -> None:
        self.node = node

    def execute(self, state: DiagramState) -> None:
        state.nodes[self.node.id] = self.node

    def undo(self, state: DiagramState) -> None:
        state.nodes.pop(self.node.id, None)


class DeleteNodeCommand(DiagramCommand):
    is_structural = True
    action_name = "NODE_DELETE"

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.deleted_node: NodeModel | None = None
        self.deleted_edges: list[EdgeModel] = []

    def execute(self, state: DiagramState) -> None:
        self.deleted_node = state.nodes.pop(self.node_id, None)
        self.deleted_edges = [
            edge
            for edge in list(state.edges.values())
            if edge.source_id == self.node_id or edge.target_id == self.node_id
        ]
        for edge in self.deleted_edges:
            state.edges.pop(edge.id, None)

    def undo(self, state: DiagramState) -> None:
        if self.deleted_node is not None:
            state.nodes[self.deleted_node.id] = self.deleted_node
        for edge in self.deleted_edges:
            state.edges[edge.id] = edge


class CreateEdgeCommand(DiagramCommand):
    is_structural = True
    action_name = "EDGE_CREATE"

    def __init__(self, edge: EdgeModel) -> None:
        self.edge = edge

    def execute(self, state: DiagramState) -> None:
        state.edges[self.edge.id] = self.edge

    def undo(self, state: DiagramState) -> None:
        state.edges.pop(self.edge.id, None)


class DeleteEdgeCommand(DiagramCommand):
    is_structural = True
    action_name = "EDGE_DELETE"

    def __init__(self, edge_id: str) -> None:
        self.edge_id = edge_id
        self.deleted_edge: EdgeModel | None = None

    def execute(self, state: DiagramState) -> None:
        self.deleted_edge = state.edges.pop(self.edge_id, None)

    def undo(self, state: DiagramState) -> None:
        if self.deleted_edge is not None:
            state.edges[self.deleted_edge.id] = self.deleted_edge


class MoveNodeCommand(DiagramCommand):
    """Mover un nodo sin cambiar relaciones NO es modificación estructural (Indicador 4)."""

    is_structural = False
    action_name = "NODE_MOVE"

    def __init__(
        self,
        node_id: str,
        old_x: float,
        old_y: float,
        new_x: float,
        new_y: float,
    ) -> None:
        self.node_id = node_id
        self.old_x = old_x
        self.old_y = old_y
        self.new_x = new_x
        self.new_y = new_y

    def execute(self, state: DiagramState) -> None:
        node = state.nodes.get(self.node_id)
        if node is not None:
            node.x = self.new_x
            node.y = self.new_y

    def undo(self, state: DiagramState) -> None:
        node = state.nodes.get(self.node_id)
        if node is not None:
            node.x = self.old_x
            node.y = self.old_y


class DiagramState:
    """Estado topológico puro del diagrama + pila de deshacer."""

    def __init__(self) -> None:
        self.nodes: dict[str, NodeModel] = {}
        self.edges: dict[str, EdgeModel] = {}
        self._undo_stack: list[DiagramCommand] = []
        self._node_seq: int = 0
        self._edge_seq: int = 0

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def next_node_id(self) -> str:
        self._node_seq += 1
        return f"N-{self._node_seq:03d}"

    def next_edge_id(self) -> str:
        self._edge_seq += 1
        return f"E-{self._edge_seq:03d}"

    def execute(self, command: DiagramCommand) -> None:
        command.execute(self)
        self._undo_stack.append(command)

    def undo(self) -> DiagramCommand | None:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo(self)
        return cmd

    def has_edge_between(self, source_id: str, target_id: str) -> bool:
        return any(
            (e.source_id == source_id and e.target_id == target_id)
            or (e.source_id == target_id and e.target_id == source_id)
            for e in self.edges.values()
        )

    def node_at(self, x: float, y: float, margin: float = 12.0) -> NodeModel | None:
        for node in reversed(list(self.nodes.values())):
            if node.contains(x, y, margin=margin):
                return node
        return None

    def edge_at(self, x: float, y: float, tolerance: float = 14.0) -> EdgeModel | None:
        best_edge: EdgeModel | None = None
        best_dist = tolerance
        for edge in self.edges.values():
            src = self.nodes.get(edge.source_id)
            dst = self.nodes.get(edge.target_id)
            if src is None or dst is None:
                continue
            dist = _point_to_segment_distance(x, y, src.x, src.y, dst.x, dst.y)
            if dist <= best_dist:
                best_dist = dist
                best_edge = edge
        return best_edge

    def to_topology_dict(self) -> dict[str, Any]:
        return {
            "nodos": [n.to_dict() for n in self.nodes.values()],
            "enlaces": [e.to_dict() for e in self.edges.values()],
            "total_nodos": len(self.nodes),
            "total_enlaces": len(self.edges),
        }


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)
