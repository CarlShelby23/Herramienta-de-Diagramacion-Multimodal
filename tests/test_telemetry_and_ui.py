from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.audio.command_parser import parse_voice_command
from src.config import CalibrationConfig, ConditionLevel, FusionConfig, SessionConfig
from src.models.diagram import DiagramState
from src.models.events import AsrEvent, GestureEventType, GestureFrame, GestureState
from src.telemetry.persistence import (
    build_session_json_filename,
    save_session_json_atomic,
    verify_session_json,
)
from src.telemetry.tracker import TelemetryTracker


def test_inactivity_and_hardware_loss_telemetry(tmp_path: Path) -> None:
    cfg = SessionConfig(
        participant_code="PART-2026-099",
        condition=ConditionLevel.LOW,
        fusion=FusionConfig(inactivity_threshold_ms=3000),
        output_dir=tmp_path,
    )
    tracker = TelemetryTracker(cfg, start_time_ms=10_000)
    diagram = DiagramState()

    # Simular pérdida de cámara durante inactividad
    tracker.update_hardware_health(11_000, camera_ok=False, hand_detected=False)
    tracker.tick(14_000)  # Abre tramo desde 10_000
    tracker.record_activity(16_000)  # Cierra tramo de 6000 ms

    tracker.update_hardware_health(16_500, camera_ok=True, mic_ok=False)
    tracker.finalize(now_ms=17_000)

    kpis = tracker.compute_kpis()
    assert kpis.inactivity_total_ms == 6000
    assert kpis.inactivity_segments_count == 1
    assert kpis.inactivity_segments[0].camera_ok is False
    assert kpis.camera_loss_events == 1
    assert kpis.mic_loss_events == 1

    # Verificar guardado atómico JSON REQ-04
    fname = build_session_json_filename(cfg.participant_code)
    assert fname.startswith("session_PART-2026-099_") and fname.endswith(".json")
    doc = tracker.build_session_document(diagram, session_filename=fname)
    saved_path = save_session_json_atomic(doc, tmp_path / fname)

    ok, msg = verify_session_json(saved_path)
    assert ok, msg
    raw = json.loads(saved_path.read_text(encoding="utf-8"))
    assert raw["sesion"]["codigo_participante"] == "PART-2026-099"
    assert "topologia" in raw
    assert "indicadores_conductuales" in raw
    assert "tiempo_inactividad_ms" in raw["indicadores_conductuales"]
    assert "frecuencia_interrupciones" in raw["indicadores_conductuales"]
    assert "latencia_respuesta_usuario_ms" in raw["indicadores_conductuales"]
    assert "modificaciones_estructurales" in raw["indicadores_conductuales"]


def test_pyside6_wf01_wf02_wf05_end_to_end_and_png_export(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from src.ui.wf01_setup_dialog import WF01SetupDialog
    from src.ui.wf02_main_window import WF02MainWindow
    from src.ui.wf05_close_dialog import WF05CloseDialog

    app = QApplication.instance() or QApplication([])

    # 1. WF-01: Configuración y calibración
    wf01 = WF01SetupDialog(start_hardware=False)
    wf01.input_participant.setText("PART-2026-014")
    wf01.slider_pinch.setValue(32)
    wf01.spin_dwell.setValue(250)
    wf01.chk_simulation.setChecked(True)
    assert wf01.btn_start.isEnabled()
    wf01._on_start_clicked()
    assert wf01.session_config is not None
    cfg = wf01.session_config
    cfg.output_dir = tmp_path
    assert cfg.participant_code == "PART-2026-014"
    assert cfg.calibration.pinch_threshold_px == 32.0
    assert cfg.calibration.dwell_time_ms == 250

    # 2. WF-02/03/04: Lienzo principal + creación + conexión + barra contextual
    win = WF02MainWindow(config=cfg, start_workers=False)
    win.show()
    app.processEvents()

    # Crear 2 nodos y conectarlos
    for concept, x, y, t0 in [("Estímulo", 280.0, 260.0, 20_000), ("Respuesta", 760.0, 260.0, 22_000)]:
        win.on_gesture_frame(
            GestureFrame(timestamp_ms=t0, stable_gesture=GestureState.MOVE, cursor_x=x, cursor_y=y, hand_detected=True)
        )
        parsed = parse_voice_command(f"Crear nodo {concept}")
        win.on_asr_event(
            AsrEvent(
                raw_text=f"Crear nodo {concept}",
                intent=parsed.intent,
                concept=parsed.concept,
                t_speech_start_ms=t0 + 50,
                t_speech_end_ms=t0 + 200,
                t_asr_ready_ms=t0 + 260,
                system_latency_ms=60,
            )
        )
        win.on_gesture_frame(
            GestureFrame(
                timestamp_ms=t0 + 700,
                stable_gesture=GestureState.OPEN_DROP,
                event=GestureEventType.DROP_CONFIRMED,
                cursor_x=x,
                cursor_y=y,
                hand_detected=True,
            )
        )

    assert len(win.diagram.nodes) == 2
    assert win.floating_bar.isVisible()

    # Conectar usando la barra contextual (Origen = Respuesta -> Destino = Estímulo)
    win._on_ctx_connect()
    win._on_canvas_clicked(280.0, 260.0)
    assert len(win.diagram.edges) == 1

    # 3. WF-05: Modal de cierre, guardado JSON y exportación independiente a PNG
    win.telemetry.finalize()
    fname = build_session_json_filename(cfg.participant_code)
    doc = win.telemetry.build_session_document(win.diagram, session_filename=fname)
    wf05 = WF05CloseDialog(
        document=doc,
        default_json_path=tmp_path / fname,
        png_exporter=win.export_clean_png,
    )

    saved_json = wf05.save_json_now()
    assert saved_json is not None and saved_json.exists()
    assert wf05.json_saved is True

    png_path = tmp_path / "diagram_export.png"
    exported = win.export_clean_png(png_path)
    assert exported.exists() and exported.stat().st_size > 1000
    loaded = QImage(str(exported))
    assert not loaded.isNull() and loaded.width() > 500

    wf05.close()
    win.close()
