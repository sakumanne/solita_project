"""Custom Holoscan operators for audio capture and Whisper transcription with VAD."""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import webrtcvad
import whisper
from holoscan.core import Operator, OperatorSpec
from speechbrain.inference import EncoderClassifier

# Audio configuration
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.5

# VAD
VAD_FRAME_MS = 20
MIN_SPEECH_RATIO = 0.10  # was 0.30 - make it easier
MIN_SPEECH_SECONDS = 0.10  # was 0.20 - lower threshold
SILENCE_DB_THRESHOLD = -65.0  # was -55.0 - more sensitive

# Utterance loppuu, kun hiljaisuutta on näin monta chunkia
END_SILENCE_CHUNKS = 3
UTTER_MAX_SECONDS = 15.0
MIN_UTTER_TO_TRANSCRIBE_SECONDS = 1.0

# Speaker
MAX_SPEAKERS = 10
SPEAKER_SIM_THRESHOLD = 0.75
LAST_SPEAKER_BONUS = 0.00
NEW_SPEAKER_STREAK = 1
MIN_UTTER_FOR_SPEAKER_SECONDS = 2.5

# Whisper
DOMAIN_PROMPT = "Suomenkielinen puhe. Litteroi tarkasti."

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LATEST_JSON = DATA_DIR / "live_transcript_latest.json"

DEBUG_SPEAKER = False

# Silence warnings
warnings.filterwarnings("ignore")


@contextmanager
def suppress_output(suppress_stdout: bool = True, suppress_stderr: bool = True):
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = old_out
        if suppress_stderr:
            sys.stderr = old_err
        devnull.close()


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -float("inf")
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * math.log10(rms + 1e-9)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)) + 1e-9)
    return (x / peak * 0.9).astype(np.float32)


def postprocess(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return ""
    chars = list(text)
    for i, c in enumerate(chars):
        if c.isalpha():
            chars[i] = c.upper()
            break
    return "".join(chars)


# VAD
vad = webrtcvad.Vad(1)


def chunk_to_pcm16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1, 1) * 32768).astype(np.int16).tobytes()


def vad_stats(pcm16: bytes) -> Tuple[float, float]:
    frame_bytes = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2
    total = 0
    speech = 0
    for i in range(0, len(pcm16) - frame_bytes + 1, frame_bytes):
        if vad.is_speech(pcm16[i:i + frame_bytes], SAMPLE_RATE):
            speech += 1
        total += 1
    if total == 0:
        return 0.0, 0.0
    return speech / total, speech * (VAD_FRAME_MS / 1000.0)


def is_chunk_speech(chunk: np.ndarray) -> bool:
    if rms_db(chunk) < SILENCE_DB_THRESHOLD:
        return False
    ratio, seconds = vad_stats(chunk_to_pcm16(chunk))
    return ratio >= MIN_SPEECH_RATIO and seconds >= MIN_SPEECH_SECONDS


class AudioCaptureOp(Operator):
    """Captures audio from microphone and emits chunks with VAD detection."""

    def __init__(self, *args, chunk_duration: float = CHUNK_SECONDS, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue()
        self.stream = None

    def setup(self, spec: OperatorSpec):
        spec.output("audio_out")

    def start(self):
        """Start audio stream."""
        frames_per_chunk = int(self.chunk_duration * SAMPLE_RATE)

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            self.audio_queue.put(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=frames_per_chunk,
            callback=callback,
        )
        self.stream.start()
        print(f"Audio capture started: {SAMPLE_RATE}Hz, {self.chunk_duration}s chunks")

    def compute(self, op_input, op_output, context):
        """Get audio chunk from queue and emit."""
        try:
            chunk = self.audio_queue.get(timeout=0.1)
            if chunk is not None and chunk.size > 0:
                op_output.emit(chunk.reshape(-1), "audio_out")
        except queue.Empty:
            pass

    def stop(self):
        """Stop audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio capture stopped")


class WhisperTranscribeOp(Operator):
    """Transcribes audio with VAD-based utterance detection and speaker identification."""

    def __init__(
        self,
        *args,
        model_name: str = "small",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.device = device
        self.whisper_model = None
        self.speaker_encoder = None
        self.speaker_centroids: List[np.ndarray] = []
        self.last_speaker: Optional[int] = None

        # Utterance buffering
        self.buf: List[np.ndarray] = []
        self.buf_len = 0.0
        self.silence = 0

    def setup(self, spec: OperatorSpec):
        spec.input("audio_in")
        spec.output("done")  # Add this

    def start(self):
        """Load models."""
        print(f"Loading Whisper model '{self.model_name}'...")
        self.whisper_model = whisper.load_model(self.model_name)

        print("Loading SpeechBrain speaker encoder...")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        print("Models loaded successfully")

    def compute(self, op_input, op_output, context):
        """Process audio chunk with VAD."""
        chunk = op_input.receive("audio_in")
        if chunk is None or chunk.size == 0:
            return

        # DEBUG
        db = rms_db(chunk)
        is_speech = is_chunk_speech(chunk)
        print(f"[DEBUG] chunk: db={db:.1f}, is_speech={is_speech}, buf_len={self.buf_len:.1f}s, silence={self.silence}")

        # VAD detection
        if is_speech:
            self.buf.append(chunk)
            self.buf_len += CHUNK_SECONDS
            self.silence = 0
        else:
            self.silence += 1

        # Check if utterance should end
        end_by_silence = bool(self.buf) and self.silence >= END_SILENCE_CHUNKS
        end_by_maxlen = self.buf_len >= UTTER_MAX_SECONDS

        if end_by_silence or end_by_maxlen:
            print(f"[DEBUG] FLUSHING: end_by_silence={end_by_silence}, end_by_maxlen={end_by_maxlen}")
            self._flush_and_process()
            op_output.emit(True, "done")

    def _flush_and_process(self):
        """Process buffered utterance."""
        if not self.buf:
            return

        utter = normalize_audio(np.concatenate(self.buf))
        utter_seconds = len(utter) / SAMPLE_RATE

        # Skip too short utterances
        if utter_seconds < MIN_UTTER_TO_TRANSCRIBE_SECONDS:
            self.buf = []
            self.buf_len = 0.0
            self.silence = 0
            return

        # Assign speaker
        if utter_seconds >= MIN_UTTER_FOR_SPEAKER_SECONDS:
            emb = self._get_speaker_embedding(utter)
            spk = self._assign_speaker(emb)
        else:
            spk = self.last_speaker if self.last_speaker is not None else 0

        # Transcribe
        text = self._transcribe(utter)

        # Write status and print
        self._write_status(text, spk)
        if text:
            speaker_label = f"SPEAKER_{spk + 1}"
            print(f"{speaker_label}: {text}")

        self.buf = []
        self.buf_len = 0.0
        self.silence = 0

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio."""
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            result = self.whisper_model.transcribe(
                audio,
                language="fi",
                task="transcribe",
                fp16=False,
                initial_prompt=DOMAIN_PROMPT,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                verbose=False,
            )
        return postprocess(result.get("text", ""))

    def _get_speaker_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Calculate speaker embedding."""
        wav = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.speaker_encoder.encode_batch(wav)
        return emb.squeeze().cpu().numpy().astype(np.float32)

    def _assign_speaker(self, emb: np.ndarray, threshold: float = SPEAKER_SIM_THRESHOLD) -> int:
        """Assign speaker ID based on embedding similarity."""
        if not self.speaker_centroids:
            self.speaker_centroids.append(emb.copy())
            self.last_speaker = 0
            return 0

        sims = [self._cosine_similarity(emb, c) for c in self.speaker_centroids]
        best = int(np.argmax(sims))
        best_sim = sims[best]

        if DEBUG_SPEAKER:
            sim_str = ", ".join(f"{i+1}:{s:.2f}" for i, s in enumerate(sims))
            print(f"DEBUG speaker sims -> [{sim_str}] best={best+1} ({best_sim:.2f})")

        if best_sim >= threshold:
            self.speaker_centroids[best] = 0.9 * self.speaker_centroids[best] + 0.1 * emb
            self.last_speaker = best
            return best

        if len(self.speaker_centroids) < MAX_SPEAKERS:
            self.speaker_centroids.append(emb.copy())
            self.last_speaker = len(self.speaker_centroids) - 1
            return self.last_speaker

        self.last_speaker = best
        return best

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    @staticmethod
    def _write_status(transcript: str, speaker_id: int) -> None:
        """Write status to JSON file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "transcript": transcript or "",
            "decibels": None,
            "loud": False,
        }
        with open(LATEST_JSON, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)