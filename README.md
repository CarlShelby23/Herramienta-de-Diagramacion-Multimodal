# Herramienta de Diagramación Multimodal (Voz + Gestos)

Prototipo funcional de escritorio (**PySide6 + OpenCV + MediaPipe + Vosk ASR offline**) diseñado para el proyecto académico de IHC y neurodiversidad (Universidad Veracruzana), cubriendo el flujo completo **WF-01 a WF-05** y telemetría conductual **REQ-04**.

## Instalación rápida

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Descarga de modelos offline (Vosk español + MediaPipe Tasks)

```bash
python scripts/download_models.py
```

Esto descarga:
- `models/vosk-model-small-es-0.42/` (ASR offline en español de baja latencia).
- `models/hand_landmarker.task` (requerido solo si usas MediaPipe 1.x; con MediaPipe 0.10.x se usa automáticamente `mp.solutions.hands`).

## Ejecución

- **Modo completo (cámara + micrófono + ASR offline):**
  ```bash
  python main.py
  ```
- **Modo simulación (pruebas sin cámara/micrófono con ratón y consola de voz):**
  ```bash
  python main.py --sim
  ```

## Flujo de pantallas (WF-01 a WF-05)

1. **WF-01 (`src/ui/wf01_setup_dialog.py`):** código anónimo de participante, selección de condición (`0. Familiarización` a `3. Alta (Abierta)`), checklist en vivo de cámara (FPS + 21 landmarks) y micrófono (ruido base en dB), calibración dinámica de pellizco (`< 30 px` base) y *Dwell Time* (`250 ms`).
2. **WF-02 / WF-03 / WF-04 (`src/ui/wf02_main_window.py`):** barra superior de consigna, lienzo minimalista con nodos rectangulares redondeados (normal negro sólido, creación por voz verde segmentado + `(X, Y)`, seleccionado/acople azul grueso segmentado), enlaces rectos segmentados, barra contextual flotante (`Conectar`, `Borrar`, `Deshacer`) y barra inferior con estado ASR/gesto, latencia y `Deshacer`.
3. **WF-05 (`src/ui/wf05_close_dialog.py`):** resumen de los 4 indicadores conductuales, leyenda de colores del cursor, verificación del archivo `session_<PART>_<fecha>.json`, y acciones independientes **Guardar Log JSON + Topología** y **Exportar Diagrama a `.PNG`**.

## Estados del cursor virtual y comandos de voz

- **Azul:** dedo índice extendido → Mover cursor (suavizado One-Euro Filter).
- **Morado:** pellizco pulgar-índice `< umbral` mantenido `250 ms` → Seleccionar / clic.
- **Naranja:** puño cerrado mantenido `250 ms` → Arrastrar nodo seleccionado (sin conectar por accidente).
- **Verde:** mano abierta mantenida `250 ms` → Soltar / confirmar drop.
- **Voz (catálogo acotado):** `"Crear nodo [Concepto]"`, `"Conectar"`, `"Borrar"` / `"Eliminar"`, `"Deshacer"`.

## Pruebas automatizadas

```bash
QT_QPA_PLATFORM=offscreen pytest -v
```

> Los archivos originales del primer prototipo se conservan como respaldo en `legacy/`.
