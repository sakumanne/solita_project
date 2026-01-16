from __future__ import annotations

import math
import os
import queue
import sys
import warnings
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
import torch


# =========================================================
# ASETUKSET (säädä tarvittaessa)
# =========================================================

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.5

# VAD
VAD_FRAME_MS = 20
MIN_SPEECH_RATIO = 0.30
MIN_SPEECH_SECONDS = 0.20
SILENCE_DB_THRESHOLD = -55.0

# Utterance loppuu, kun hiljaisuutta on näin monta chunkia
END_SILENCE_CHUNKS = 3        # 3 * 0.5s = 1.5s
UTTER_MAX_SECONDS = 10.0      # varmistus

# Speaker
MAX_SPEAKERS = 10
SPEAKER_SIM_THRESHOLD = 0.60
LAST_SPEAKER_BONUS = 0.05
NEW_SPEAKER_STREAK = 2

# Whisper prompt
DOMAIN_PROMPT = (
    "Tämä on suomenkielinen keskustelu. "
    "Käytä selkeää yleiskieltä. Älä lisää mitään mitä ei sanota ääneen."
)

# UI: jos haluat joskus debugata, vaihda True
DEBUG = False


# =========================================================
# Hiljennä kaikki varoitukset & ylimääräinen output
# =========================================================

warnings.filterwarnings("ignore")


def status(msg: str) -> None:
    """Näytetään vain 'odotusvaiheiden' statusrivit."""
    print(f"STATUS: {msg}", flush=True)


@contextmanager
def suppress_output(suppress_stdout: bool = True, suppress_stderr: bool = True):
    """Estää tqdm/progressit ja macOS/tk-tyyppiset stderr-printit."""
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


# =========================================================
# APUTOIMINNOT
# =========================================================

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


# =========================================================
# VAD
# =========================================================

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


# =========================================================
# SPEAKER (ECAPA)
# =========================================================

@dataclass
class SpeakerState:
    encoder: Any
    centroids: List[np.ndarray]
    device: str = "cpu"
    last_speaker: Optional[int] = None
    low_sim_streak: int = 0


def load_speaker_encoder(device: str = "cpu") -> SpeakerState:
    status("Ladataan speaker-encoder (ECAPA)...")
    warnings.filterwarnings("ignore")
    from speechbrain.inference import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="pretrained_ecapa",
    )
    status("Speaker-encoder valmis.")
    return SpeakerState(encoder=encoder, centroids=[], device=device)


def speaker_embedding(state: SpeakerState, audio: np.ndarray) -> np.ndarray:
    wav = torch.from_numpy(audio).float().to(state.device).unsqueeze(0)
    with torch.no_grad():
        emb = state.encoder.encode_batch(wav).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def assign_speaker(state: SpeakerState, emb: np.ndarray) -> int:
    if not state.centroids:
        state.centroids.append(emb.copy())
        state.last_speaker = 0
        state.low_sim_streak = 0
        return 0

    sims = [cosine_similarity(emb, c) for c in state.centroids]

    if state.last_speaker is not None:
        sims[state.last_speaker] += LAST_SPEAKER_BONUS

    best = int(np.argmax(sims))
    best_sim = sims[best]

    if best_sim >= SPEAKER_SIM_THRESHOLD:
        state.centroids[best] = 0.9 * state.centroids[best] + 0.1 * emb
        state.last_speaker = best
        state.low_sim_streak = 0
        return best

    state.low_sim_streak += 1
    if state.low_sim_streak < NEW_SPEAKER_STREAK:
        return state.last_speaker if state.last_speaker is not None else best

    state.low_sim_streak = 0
    if len(state.centroids) < MAX_SPEAKERS:
        state.centroids.append(emb.copy())
        state.last_speaker = len(state.centroids) - 1
        return state.last_speaker

    state.last_speaker = best
    return best


def speaker_label(i: int) -> str:
    return f"SPEAKER_{i + 1}"


# =========================================================
# WHISPER
# =========================================================

def whisper_options() -> dict:
    return dict(
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


def transcribe_audio(model, audio: np.ndarray) -> str:
    # Ei status-rivejä per transkriptio – vain lopputulos tulostetaan
    with suppress_output(suppress_stdout=True, suppress_stderr=True):
        result = model.transcribe(audio, **whisper_options())
    return postprocess(result.get("text", ""))


def load_audio_any(path: str) -> np.ndarray:
    """
    Whisperin oma ffmpeg-loader (ei librosaa) -> ei PySoundFile/audioread warningeja.
    Vaatii että ffmpeg on asennettu.
    """
    status("Ladataan audio (ffmpeg)...")
    try:
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            audio = whisper.audio.load_audio(path)  # float32, 16000 Hz
        status(f"Audio ladattu ({len(audio)/SAMPLE_RATE:.1f} s)")
        return audio.astype(np.float32)
    except Exception as e:
        print(f"ERROR: En saanut ladattua tiedostoa (tarvitset ffmpeg?): {e}")
        return np.array([], dtype=np.float32)


# =========================================================
# LIVE
# =========================================================

def record_chunks(chunk_seconds: float):
    frames = int(SAMPLE_RATE * chunk_seconds)
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status_):  # type: ignore[override]
        q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frames,
        callback=callback,
    ):
        while True:
            yield q.get().reshape(-1)


def live_transcription(model_name: str, device: str = "cpu") -> None:
    status(f"Ladataan Whisper-malli ({model_name})...")
    model = whisper.load_model(model_name)
    status("Whisper-malli valmis.")
    speaker_state = load_speaker_encoder(device=device)

    buf: List[np.ndarray] = []
    buf_len = 0.0
    silence = 0

    print("\nLIVE (Ctrl+C lopettaa)\n")

    try:
        for chunk in record_chunks(CHUNK_SECONDS):
            if is_chunk_speech(chunk):
                buf.append(chunk)
                buf_len += CHUNK_SECONDS
                silence = 0
            else:
                silence += 1

            end_by_silence = bool(buf) and silence >= END_SILENCE_CHUNKS
            end_by_maxlen = buf_len >= UTTER_MAX_SECONDS

            if end_by_silence or end_by_maxlen:
                utter = normalize_audio(np.concatenate(buf))
                emb = speaker_embedding(speaker_state, utter)
                spk = assign_speaker(speaker_state, emb)

                text = transcribe_audio(model, utter)
                if text:
                    print(f"{speaker_label(spk)}: {text}")

                buf = []
                buf_len = 0.0
                silence = 0

    except KeyboardInterrupt:
        print("\nLopetettu.")


# =========================================================
# FILE
# =========================================================

def transcribe_file_with_speakers(model_name: str, device: str = "cpu") -> None:
    # Tkinter file picker (hiljaisena)
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        status("tkinter puuttuu. Anna tiedostopolku komentoriviltä.")
        path = input("Tiedostopolku: ").strip()
    else:
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename()
            root.destroy()

    if not path:
        return

    status(f"Tiedosto valittu: {path}")
    audio = load_audio_any(path)
    if audio.size == 0:
        return

    status(f"Ladataan Whisper-malli ({model_name})...")
    model = whisper.load_model(model_name)
    status("Whisper-malli valmis.")

    speaker_state = load_speaker_encoder(device=device)

    chunk_len = int(SAMPLE_RATE * CHUNK_SECONDS)

    buf: List[np.ndarray] = []
    buf_len = 0.0
    silence = 0

    print("\nFILE\n")

    # Käydään koko audio läpi chunkkeina
    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i + chunk_len]
        if chunk.shape[0] < chunk_len:
            break

        if is_chunk_speech(chunk):
            buf.append(chunk)
            buf_len += CHUNK_SECONDS
            silence = 0
        else:
            silence += 1

        end_by_silence = bool(buf) and silence >= END_SILENCE_CHUNKS
        end_by_maxlen = buf_len >= UTTER_MAX_SECONDS

        if end_by_silence or end_by_maxlen:
            utter = normalize_audio(np.concatenate(buf))
            emb = speaker_embedding(speaker_state, utter)
            spk = assign_speaker(speaker_state, emb)

            text = transcribe_audio(model, utter)
            if text:
                print(f"{speaker_label(spk)}: {text}")

            buf = []
            buf_len = 0.0
            silence = 0


# =========================================================
# CLI
# =========================================================

def choose_mode() -> str:
    while True:
        print("Valitse tila:")
        print("1) Live mikrofoni")
        print("2) Tiedostosta")
        choice = input("Valinta (1/2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("Anna 1 tai 2.\n")


def choose_model() -> str:
    mapping = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large",
    }
    print("Valitse Whisper-malli:")
    print("1) tiny")
    print("2) base")
    print("3) small")
    print("4) medium")
    print("5) large")
    choice = input("Valinta (ENTER = small): ").strip()
    if not choice:
        return "small"
    return mapping.get(choice, "small")


def main():
    mode = choose_mode()
    model_name = choose_model()

    if mode == "2":
        transcribe_file_with_speakers(model_name=model_name, device="cpu")
    else:
        live_transcription(model_name=model_name, device="cpu")


if __name__ == "__main__":
    main()

