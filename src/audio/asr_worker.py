from __future__ import annotations

import json
import queue
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from src.audio.audio_monitor import compute_rms_db
from src.audio.command_parser import parse_voice_command
from src.config import CalibrationConfig
from src.models.events import AsrEvent


class BaseASREngine(ABC):
    """Interfaz enchufable para motores ASR offline (Vosk, faster-whisper, etc.)."""

    name: str = "base"

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def accept_pcm16(self, pcm_bytes: bytes) -> str | None:
        """Alimenta un bloque PCM16 mono 16kHz; retorna texto final si cerró frase."""
        ...

    def get_partial(self) -> str:
        return ""

    def reset(self) -> None:
        pass


class VoskASREngine(BaseASREngine):
    """Motor ASR offline de baja latencia basado en Vosk (streaming continuo)."""

    name = "Vosk (Offline)"

    def __init__(self, model_path: str | Path, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.error_message: str | None = None
        self._recognizer = None

        resolved = Path(model_path)
        if not resolved.exists():
            self.error_message = (
                f"Modelo Vosk no encontrado en '{resolved}'. "
                "Ejecuta: python scripts/download_models.py"
            )
            return

        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            model = Model(str(resolved))
            self._recognizer = KaldiRecognizer(model, sample_rate)
            self._recognizer.SetWords(False)
        except Exception as exc:
            self.error_message = f"Error inicializando Vosk: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._recognizer is not None

    def accept_pcm16(self, pcm_bytes: bytes) -> str | None:
        if self._recognizer is None:
            return None
        if self._recognizer.AcceptWaveform(pcm_bytes):
            payload = json.loads(self._recognizer.Result())
            text = (payload.get("text") or "").strip()
            return text if text else None
        return None

    def get_partial(self) -> str:
        if self._recognizer is None:
            return ""
        payload = json.loads(self._recognizer.PartialResult())
        return (payload.get("partial") or "").strip()

    def reset(self) -> None:
        if self._recognizer is not None:
            self._recognizer.Reset()


class FasterWhisperASREngine(BaseASREngine):
    """Motor ASR offline opcional basado en faster-whisper (transcripción por segmento VAD)."""

    name = "faster-whisper (Offline)"

    def __init__(
        self,
        model_size: str = "tiny",
        sample_rate: int = 16000,
        silence_threshold_db: float = -38.0,
        silence_hangover_blocks: int = 7,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_threshold_db = silence_threshold_db
        self.silence_hangover_blocks = silence_hangover_blocks
        self.error_message: str | None = None
        self._model = None
        self._buffer: list[np.ndarray] = []
        self._in_speech = False
        self._silence_count = 0

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as exc:
            self.error_message = f"faster-whisper no disponible: {exc}"

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def accept_pcm16(self, pcm_bytes: bytes) -> str | None:
        if self._model is None:
            return None
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        db = compute_rms_db(arr)
        if db > self.silence_threshold_db:
            self._in_speech = True
            self._silence_count = 0
            self._buffer.append(arr)
            return None

        if self._in_speech:
            self._buffer.append(arr)
            self._silence_count += 1
            if self._silence_count >= self.silence_hangover_blocks:
                audio_f32 = np.concatenate(self._buffer).astype(np.float32) / 32768.0
                self._buffer.clear()
                self._in_speech = False
                self._silence_count = 0
                segments, _ = self._model.transcribe(audio_f32, language="es", beam_size=1)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                return text if text else None
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._in_speech = False
        self._silence_count = 0


def build_asr_engine(calibration: CalibrationConfig) -> tuple[BaseASREngine | None, str]:
    """Construye el motor ASR configurado y devuelve (engine, mensaje_estado)."""
    if calibration.simulation_mode or calibration.asr_engine == "simulated":
        return None, "ASR en modo simulación (consola de comandos)"

    if calibration.asr_engine == "faster-whisper":
        engine = FasterWhisperASREngine()
        if engine.is_ready:
            return engine, f"{engine.name} listo"
        return None, engine.error_message or "faster-whisper no disponible"

    # Por defecto: Vosk offline
    engine = VoskASREngine(calibration.vosk_model_path)
    if engine.is_ready:
        return engine, f"{engine.name} listo"
    return None, engine.error_message or "Modelo Vosk no disponible"


class AsrWorker(QThread):
    """Hilo dedicado de escucha continua de voz con medición precisa de latencia ASR.

    No cierra ni recalibra el micrófono entre comandos, eliminando puntos ciegos.
    """

    asr_event = Signal(object)         # AsrEvent
    partial_text = Signal(str)         # Texto parcial en vivo para feedback
    audio_level_db = Signal(float)     # dBFS actual
    status_changed = Signal(str, bool) # (mensaje, mic_and_asr_ok)

    def __init__(
        self,
        calibration: CalibrationConfig,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.calibration = calibration
        self.sample_rate = sample_rate
        self._running = False
        self._audio_queue: queue.Queue[tuple[bytes, int]] = queue.Queue(maxsize=64)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def create_event_from_text(
        self,
        raw_text: str,
        t_speech_start_ms: int,
        t_speech_end_ms: int,
        t_asr_ready_ms: int,
    ) -> AsrEvent:
        parsed = parse_voice_command(raw_text)
        sys_lat = max(0, t_asr_ready_ms - t_speech_end_ms)
        return AsrEvent(
            raw_text=raw_text,
            intent=parsed.intent,
            concept=parsed.concept,
            t_speech_start_ms=t_speech_start_ms,
            t_speech_end_ms=t_speech_end_ms,
            t_asr_ready_ms=t_asr_ready_ms,
            system_latency_ms=sys_lat,
            error_message=None if parsed.intent.name not in ("UNRECOGNIZED", "AMBIGUOUS") else parsed.reason,
        )

    def run(self) -> None:
        self._running = True
        engine, status_msg = build_asr_engine(self.calibration)
        if engine is None:
            self.status_changed.emit(status_msg, False)
            return

        try:
            import sounddevice as sd
        except Exception as exc:
            self.status_changed.emit(f"sounddevice no disponible: {exc}", False)
            return

        block_size = int(self.sample_rate * 0.1)  # 100 ms por bloque
        speech_threshold_db = max(-48.0, self.calibration.ambient_noise_db + 8.0)

        def audio_callback(indata, frames, time_info, status) -> None:
            if not self._running:
                return
            ts_ms = int(time.time() * 1000)
            try:
                self._audio_queue.put_nowait((bytes(indata), ts_ms))
            except queue.Full:
                pass

        t_speech_start_ms: int | None = None
        t_last_voice_ms: int | None = None
        last_partial = ""

        try:
            with sd.RawInputStream(
                device=self.calibration.mic_device_index,
                samplerate=self.sample_rate,
                blocksize=block_size,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                self.status_changed.emit(f"Escuchando ({engine.name})", True)
                while self._running:
                    try:
                        pcm_bytes, ts_ms = self._audio_queue.get(timeout=0.15)
                    except queue.Empty:
                        continue

                    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                    db = compute_rms_db(samples)
                    self.audio_level_db.emit(db)

                    if db > speech_threshold_db:
                        if t_speech_start_ms is None:
                            t_speech_start_ms = ts_ms
                        t_last_voice_ms = ts_ms

                    final_text = engine.accept_pcm16(pcm_bytes)
                    if final_text:
                        ready_ms = int(time.time() * 1000)
                        end_ms = t_last_voice_ms or ts_ms
                        start_ms = t_speech_start_ms or end_ms
                        event = self.create_event_from_text(
                            raw_text=final_text,
                            t_speech_start_ms=start_ms,
                            t_speech_end_ms=end_ms,
                            t_asr_ready_ms=ready_ms,
                        )
                        self.asr_event.emit(event)
                        self.partial_text.emit("")
                        last_partial = ""
                        t_speech_start_ms = None
                        t_last_voice_ms = None
                    else:
                        partial = engine.get_partial()
                        if partial != last_partial:
                            last_partial = partial
                            self.partial_text.emit(partial)
        except Exception as exc:
            self.status_changed.emit(f"Error de micrófono/ASR: {exc}", False)
