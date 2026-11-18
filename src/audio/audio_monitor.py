from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QThread, Signal


def compute_rms_db(samples_int16: np.ndarray) -> float:
    """Calcula nivel RMS en dBFS (-80.0 a 0.0 dB) desde audio PCM int16."""
    if samples_int16.size == 0:
        return -80.0
    normalized = samples_int16.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))
    if rms <= 1e-6:
        return -80.0
    db = 20.0 * math.log10(rms)
    return max(-80.0, min(0.0, db))


class AudioLevelMonitor(QThread):
    """Monitor ligero de micrófono para WF-01 (medición de ruido base en dB en vivo)."""

    level_updated = Signal(float, float)  # (db_actual, db_ruido_base_estimado)
    status_changed = Signal(str, bool)    # (mensaje, mic_ok)

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.device_index = device_index
        self.sample_rate = sample_rate
        self._running = False
        self._baseline_samples: deque[float] = deque(maxlen=40)

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def run(self) -> None:
        self._running = True
        try:
            import sounddevice as sd
        except Exception as exc:
            self.status_changed.emit(f"sounddevice no disponible: {exc}", False)
            return

        block_size = int(self.sample_rate * 0.1)  # 100 ms
        try:
            with sd.InputStream(
                device=self.device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
            ) as stream:
                self.status_changed.emit("Micrófono activo", True)
                while self._running:
                    data, overflowed = stream.read(block_size)
                    if overflowed:
                        continue
                    db = compute_rms_db(data[:, 0])
                    self._baseline_samples.append(db)
                    # Percentil 30 estima el piso de ruido ambiente ignorando voz ocasional
                    baseline = float(np.percentile(list(self._baseline_samples), 30))
                    self.level_updated.emit(db, baseline)
        except Exception as exc:
            self.status_changed.emit(f"No se detecta micrófono activo ({exc})", False)
