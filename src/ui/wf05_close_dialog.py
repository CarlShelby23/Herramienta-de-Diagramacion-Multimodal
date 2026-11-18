from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.config import CURSOR_LEGEND
from src.models.session import SessionDocument
from src.telemetry.persistence import (
    build_diagram_png_filename,
    save_session_json_atomic,
    verify_session_json,
)


class WF05CloseDialog(QDialog):
    """WF-05 — Modal de Cierre, Resumen de los 4 KPIs y Exportación Independiente."""

    def __init__(
        self,
        document: SessionDocument,
        default_json_path: Path,
        png_exporter: Callable[[Path], Path],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("WF-05 | Cierre de Sesión y Verificación REQ-04")
        self.setMinimumSize(780, 560)

        self.document = document
        self.json_path = default_json_path
        self._png_exporter = png_exporter
        self.json_saved: bool = False
        self.png_exported_path: Path | None = None

        self._build_ui()
        self._refresh_verification_status()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        # Encabezado
        title = QLabel("Cierre de Sesión y Verificación de Telemetría (WF-05)")
        title.setObjectName("title")
        sub = QLabel(
            f"Participante: {self.document.participant_code}  ·  "
            f"Condición: {self.document.condition}  ·  "
            f"Duración: {self.document.duration_ms / 1000.0:.1f} s"
        )
        sub.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(sub)

        # --- 4 Indicadores Conductuales ---
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(12)
        kpis = self.document.kpis

        # KPI 1: Tiempo de inactividad
        inact_s = kpis.inactivity_total_ms / 1000.0
        hw_note = (
            f"Tramos: {kpis.inactivity_segments_count} · "
            f"Pérdidas (cám/mic/mano): {kpis.camera_loss_events}/"
            f"{kpis.mic_loss_events}/{kpis.hand_loss_events}"
        )
        kpi_grid.addWidget(
            self._make_kpi_card(
                "1. Tiempo de Inactividad",
                f"{kpis.inactivity_total_ms} ms ({inact_s:.1f} s)",
                hw_note,
            ),
            0,
            0,
        )

        # KPI 2: Frecuencia de interrupciones
        inter_detail = (
            f"Cancelaciones: {kpis.intentional_cancels} · "
            f"Timeouts: {kpis.timeout_abandons} · "
            f"Errores rec.: {kpis.recognition_errors} · "
            f"Válidas: {kpis.valid_actions_total}"
        )
        kpi_grid.addWidget(
            self._make_kpi_card(
                "2. Frecuencia de Interrupciones",
                f"{kpis.interruptions_total}",
                inter_detail,
            ),
            0,
            1,
        )

        # KPI 3: Latencia de respuesta del usuario
        user_lat_str = (
            f"{kpis.user_latency_mean_ms:.0f} ms"
            if kpis.user_latency_mean_ms is not None
            else "Sin muestras"
        )
        sys_lat_str = (
            f"{kpis.system_asr_latency_mean_ms:.0f} ms"
            if kpis.system_asr_latency_mean_ms is not None
            else "--"
        )
        lat_detail = (
            f"Muestras usuario: {len(kpis.user_latency_samples_ms)} · "
            f"Latencia sistema ASR prom.: {sys_lat_str}"
        )
        kpi_grid.addWidget(
            self._make_kpi_card(
                "3. Latencia Respuesta Usuario",
                user_lat_str,
                lat_detail,
            ),
            1,
            0,
        )

        # KPI 4: Modificaciones estructurales
        struct_detail = (
            f"Nodos (+{kpis.nodes_created_count}/-{kpis.nodes_deleted_count}) · "
            f"Enlaces (+{kpis.edges_created_count}/-{kpis.edges_deleted_count}) · "
            f"Movimientos (excluidos): {kpis.non_structural_moves_count}"
        )
        kpi_grid.addWidget(
            self._make_kpi_card(
                "4. Modificaciones Estructurales",
                f"{kpis.structural_modifications_total}",
                struct_detail,
            ),
            1,
            1,
        )
        layout.addLayout(kpi_grid)

        # --- Leyenda de Estados del Cursor ---
        legend_card = QFrame()
        legend_card.setObjectName("card")
        legend_lay = QVBoxLayout(legend_card)
        legend_lay.setContentsMargins(14, 10, 14, 10)
        legend_lay.setSpacing(6)
        lbl_leg = QLabel("Leyenda de Estados del Cursor Virtual")
        lbl_leg.setObjectName("sectionHeader")
        legend_lay.addWidget(lbl_leg)

        leg_grid = QGridLayout()
        for idx, (name, color_hex, desc) in enumerate(CURSOR_LEGEND):
            dot = QLabel(f"● {name}:")
            dot.setStyleSheet(f"color: {color_hex}; font-weight: 700;")
            txt = QLabel(desc)
            txt.setStyleSheet("color: #334155;")
            row_box = QHBoxLayout()
            row_box.addWidget(dot)
            row_box.addWidget(txt, stretch=1)
            container = QFrame()
            container.setLayout(row_box)
            leg_grid.addWidget(container, idx // 2, idx % 2)
        legend_lay.addLayout(leg_grid)
        layout.addWidget(legend_card)

        # --- Verificación de Archivo Generado ---
        verify_card = QFrame()
        verify_card.setObjectName("card")
        v_lay = QVBoxLayout(verify_card)
        v_lay.setContentsMargins(14, 10, 14, 10)
        v_lay.setSpacing(4)

        lbl_v = QLabel("Verificación del Archivo de Telemetría y Topología (REQ-04)")
        lbl_v.setObjectName("sectionHeader")
        self.lbl_filename = QLabel(f"Archivo objetivo: {self.json_path}")
        self.lbl_filename.setStyleSheet("font-family: Consolas, monospace; color: #1E293B;")
        self.lbl_verify_status = QLabel("")
        self.lbl_png_status = QLabel("Exportación .PNG: pendiente (independiente de telemetría)")
        self.lbl_png_status.setObjectName("subtitle")

        v_lay.addWidget(lbl_v)
        v_lay.addWidget(self.lbl_filename)
        v_lay.addWidget(self.lbl_verify_status)
        v_lay.addWidget(self.lbl_png_status)
        layout.addWidget(verify_card)

        # --- Dos Acciones Explícitas e Independientes + Cerrar ---
        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.btn_save_json = QPushButton("Guardar Log JSON + Topología")
        self.btn_save_json.setObjectName("primaryButton")
        self.btn_save_json.clicked.connect(self.save_json_now)

        self.btn_export_png = QPushButton("Exportar Diagrama a .PNG")
        self.btn_export_png.setObjectName("accentButton")
        self.btn_export_png.clicked.connect(self._on_export_png_clicked)

        self.btn_close = QPushButton("Cerrar")
        self.btn_close.clicked.connect(self.accept)

        actions.addWidget(self.btn_save_json)
        actions.addWidget(self.btn_export_png)
        actions.addStretch()
        actions.addWidget(self.btn_close)
        layout.addLayout(actions)

    @staticmethod
    def _make_kpi_card(title: str, value: str, detail: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 12px; font-weight: 700; color: #475569;")
        lbl_v = QLabel(value)
        lbl_v.setStyleSheet("font-size: 20px; font-weight: 700; color: #0F172A;")
        lbl_d = QLabel(detail)
        lbl_d.setWordWrap(True)
        lbl_d.setStyleSheet("font-size: 11px; color: #64748B;")
        lay.addWidget(lbl_t)
        lay.addWidget(lbl_v)
        lay.addWidget(lbl_d)
        return card

    def _refresh_verification_status(self) -> None:
        ok, msg = verify_session_json(self.json_path)
        self.json_saved = ok
        color = "#15803D" if ok else "#B45309"
        prefix = "✔ " if ok else "● "
        self.lbl_verify_status.setText(f"{prefix}{msg}")
        self.lbl_verify_status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def save_json_now(self) -> Path | None:
        try:
            save_session_json_atomic(self.document, self.json_path)
            self._refresh_verification_status()
            return self.json_path
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error al guardar JSON",
                f"No se pudo guardar el archivo de telemetría:\n{exc}",
            )
            return None

    def _on_export_png_clicked(self) -> None:
        default_name = build_diagram_png_filename(self.document.participant_code)
        suggested = str(self.json_path.parent / default_name)
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Diagrama a PNG",
            suggested,
            "Imágenes PNG (*.png)",
        )
        if not chosen:
            return
        target = Path(chosen)
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        try:
            saved = self._png_exporter(target)
            self.png_exported_path = saved
            self.lbl_png_status.setText(f"✔ Diagrama exportado a PNG: {saved}")
            self.lbl_png_status.setStyleSheet("color: #15803D; font-weight: 600;")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error al exportar PNG",
                f"No se pudo exportar el diagrama a PNG:\n{exc}",
            )
