from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

from src.models.session import SessionDocument


def sanitize_filename_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", token.strip())
    return cleaned.strip("_") or "PART"


def build_session_json_filename(
    participant_code: str, dt: datetime | None = None
) -> str:
    """Genera el nombre estándar `session_<PART>_<fecha>.json`."""
    stamp = (dt or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_part = sanitize_filename_token(participant_code)
    return f"session_{safe_part}_{stamp}.json"


def build_diagram_png_filename(
    participant_code: str, dt: datetime | None = None
) -> str:
    stamp = (dt or datetime.now()).strftime("%Y%m%d_%H%M%S")
    safe_part = sanitize_filename_token(participant_code)
    return f"diagram_{safe_part}_{stamp}.png"


def save_session_json_atomic(document: SessionDocument, target_path: Path) -> Path:
    """Guarda el documento JSON REQ-04 de forma atómica (tmp + os.replace)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document.to_dict(), indent=2, ensure_ascii=False)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".tmp_session_", suffix=".json", dir=str(target_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target_path


def verify_session_json(path: Path) -> tuple[bool, str]:
    """Verifica que el archivo JSON guardado existe, es válido y contiene REQ-04."""
    if not path.exists():
        return False, "Pendiente de guardar"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"sesion", "topologia", "indicadores_conductuales", "eventos_telemetria"}
        missing = required - set(data.keys())
        if missing:
            return False, f"JSON incompleto (faltan: {', '.join(sorted(missing))})"
        n_nodes = data["topologia"].get("total_nodos", 0)
        n_edges = data["topologia"].get("total_enlaces", 0)
        n_events = len(data["eventos_telemetria"])
        size_kb = path.stat().st_size / 1024.0
        return (
            True,
            f"Verificado ({size_kb:.1f} KB · {n_nodes} nodos · {n_edges} enlaces · {n_events} eventos)",
        )
    except Exception as exc:
        return False, f"Error verificando JSON: {exc}"


def export_scene_to_png(
    scene,
    target_path: Path,
    hidden_items: list | None = None,
    scale: float = 2.0,
) -> Path:
    """Exporta el QGraphicsScene del diagrama a .PNG de forma independiente a la telemetría."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter

    target_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_items = hidden_items or []
    prev_visibility = [item.isVisible() for item in hidden_items]

    try:
        for item in hidden_items:
            item.setVisible(False)

        rect: QRectF = scene.sceneRect()
        if rect.isEmpty():
            rect = QRectF(0, 0, 1280, 720)

        img_w = max(1, int(rect.width() * scale))
        img_h = max(1, int(rect.height() * scale))
        image = QImage(img_w, img_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor("#FFFFFF"))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        scene.render(painter, QRectF(0, 0, img_w, img_h), rect, Qt.AspectRatioMode.KeepAspectRatio)
        painter.end()

        if not image.save(str(target_path), "PNG"):
            raise IOError(f"No se pudo escribir el archivo PNG en {target_path}")
    finally:
        for item, was_visible in zip(hidden_items, prev_visibility):
            item.setVisible(was_visible)

    return target_path
