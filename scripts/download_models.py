"""Descarga los modelos offline necesarios (Vosk español y MediaPipe HandLandmarker)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

MODELS_DIR = Path("models")
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
VOSK_DIR = MODELS_DIR / "vosk-model-small-es-0.42"
HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_TASK_PATH = MODELS_DIR / "hand_landmarker.task"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {url} -> {destination} ...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(destination, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                if chunk:
                    fh.write(chunk)
    print(f"Listo: {destination}")


def ensure_vosk_model() -> None:
    if VOSK_DIR.exists():
        print(f"Modelo Vosk ya existe en {VOSK_DIR}")
        return
    zip_path = MODELS_DIR / "vosk-model-small-es-0.42.zip"
    _download_file(VOSK_URL, zip_path)
    print(f"Extrayendo {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODELS_DIR)
    zip_path.unlink(missing_ok=True)
    print(f"Modelo Vosk listo en {VOSK_DIR}")


def ensure_hand_landmarker_model() -> None:
    if HAND_TASK_PATH.exists():
        print(f"Modelo MediaPipe HandLandmarker ya existe en {HAND_TASK_PATH}")
        return
    _download_file(HAND_TASK_URL, HAND_TASK_PATH)


def main() -> None:
    ensure_vosk_model()
    ensure_hand_landmarker_model()
    print("\nTodos los modelos offline están listos en ./models/")


if __name__ == "__main__":
    main()
