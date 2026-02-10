from __future__ import annotations

import json
import math
import os
import queue
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper
import torch


# =========================================================
# ASETUKSET
# =========================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LATEST_JSON = DATA_DIR / "live_transcript_latest.json"

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 0.5

# VAD
VAD_FRAME_MS = 20
MIN_SPEECH_RATIO = 0.30
MIN_SPEECH_SECONDS = 0.20
SILENCE_DB_THRESHOLD = -55.0

# Utterance loppuu, kun hiljaisuutta on näin monta chunkia
END_SILENCE_CHUNKS = 3          # 3 * 0.5s = 1.5s
UTTER_MAX_SECONDS = 15.0        # pidempi pätkä auttaa speaker-embeddingiä

# Älä transkriboi liian lyhyitä pätkiä (roskaa / VAD false positive)
MIN_UTTER_TO_TRANSCRIBE_SECONDS = 1.0

# Speaker (TIUKENNETTU: jotta uudet puhujat syntyy helpommin)
MAX_SPEAKERS = 10
SPEAKER_SIM_THRESHOLD = 0.75    # ↑ oli 0.60 -> liian lepsu, kaikki menee helposti samaan puhujaan
LAST_SPEAKER_BONUS = 0.00       # pois, ettei lukkiudu samaan puhujaan
NEW_SPEAKER_STREAK = 1          # heti uusi puhuja jos sim alle kynnyksen
MIN_UTTER_FOR_SPEAKER_SECONDS = 2.5  # liian lyhyistä pätkistä ei tehdä speaker-päätöstä

# Whisper prompt (neutraali)
DOMAIN_PROMPT = "Suomenkielinen puhe. Litteroi tarkasti."

# Debug: jos haluat nähdä speaker-similariteetit
DEBUG_SPEAKER = False


# =========================================================
# Hiljennä varoituksia
# =========================================================
warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects.*"
)


def status(msg: str) -> None:
    print(f"STATUS: {msg}", flush=True)


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


def write_status(transcript: str, speaker: str) -> None:
    """Write latest transcript status to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    status_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transcript": transcript or "",
        "speaker": speaker,
        "decibels": None,
        "loud": False,
    }
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)


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


def load_speaker_encoder(device: str = "cpu") -> SpeakerState:
    status("Ladataan speaker-encoder (ECAPA)...")
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
    # 1) ensimmäinen puhuja
    if not state.centroids:
        state.centroids.append(emb.copy())
        state.last_speaker = 0
        return 0

    sims = [cosine_similarity(emb, c) for c in state.centroids]
    best = int(np.argmax(sims))
    best_sim = sims[best]

    if DEBUG_SPEAKER:
        sim_str = ", ".join(f"{i+1}:{s:.2f}" for i, s in enumerate(sims))
        print(f"DEBUG speaker sims -> [{sim_str}] best={best+1} ({best_sim:.2f})")

    # 2) jos riittävän samanlainen, päivitä centroidi
    if best_sim >= SPEAKER_SIM_THRESHOLD:
        state.centroids[best] = 0.9 * state.centroids[best] + 0.1 * emb
        state.last_speaker = best
        return best

    # 3) muuten tee uusi puhuja (jos mahtuu)
    if len(state.centroids) < MAX_SPEAKERS:
        state.centroids.append(emb.copy())
        state.last_speaker = len(state.centroids) - 1
        return state.last_speaker

    # 4) jos täynnä, pakota lähin
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


BANNED_OUTPUTS = {
    "Suomenkielinen puhe. Litteroi tarkasti.",
    "Käytä selkeää yleiskieltä. Älä lisää mitään mitä ei sanota ääneen.",
    "Älä lisää mitään mitä ei sanota ääneen.",
}


def transcribe_audio(model, audio: np.ndarray) -> str:
    with suppress_output(suppress_stdout=True, suppress_stderr=True):
        result = model.transcribe(audio, **whisper_options())
    text = postprocess(result.get("text", ""))

    if not text:
        return ""
    if text in BANNED_OUTPUTS:
        return ""
    if text.strip().lower() == DOMAIN_PROMPT.strip().lower():
        return ""
    return text


def load_audio_any(path: str) -> np.ndarray:
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


def _flush_and_print(model, speaker_state: SpeakerState, buf: List[np.ndarray]) -> None:
    if not buf:
        return

    utter = normalize_audio(np.concatenate(buf))
    utter_seconds = len(utter) / SAMPLE_RATE

    # Skip liian lyhyet pätkät kokonaan
    if utter_seconds < MIN_UTTER_TO_TRANSCRIBE_SECONDS:
        return

    # Speaker päätös: tee vain jos pätkä on tarpeeksi pitkä
    if utter_seconds >= MIN_UTTER_FOR_SPEAKER_SECONDS:
        emb = speaker_embedding(speaker_state, utter)
        spk = assign_speaker(speaker_state, emb)
    else:
        # liian lyhyt pätkä -> käytä viimeisintä puhujaa (tai 0)
        spk = speaker_state.last_speaker if speaker_state.last_speaker is not None else 0

    text = transcribe_audio(model, utter)
    if text:
        speaker = speaker_label(spk)
        print(f"{speaker}: {text}")
        write_status(text, speaker)


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
                _flush_and_print(model, speaker_state, buf)
                buf = []
                buf_len = 0.0
                silence = 0

    except KeyboardInterrupt:
        _flush_and_print(model, speaker_state, buf)
        print("\nLopetettu.")


# =========================================================
# FILE
# =========================================================

def transcribe_file_with_speakers(model_name: str, device: str = "cpu") -> None:
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

    for i in range(0, len(audio), chunk_len):
        chunk = audio[i:i + chunk_len]

        # Pad viimeinen vajaa chunk
        if chunk.shape[0] < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - chunk.shape[0]))

        if is_chunk_speech(chunk):
            buf.append(chunk)
            buf_len += CHUNK_SECONDS
            silence = 0
        else:
            silence += 1

        end_by_silence = bool(buf) and silence >= END_SILENCE_CHUNKS
        end_by_maxlen = buf_len >= UTTER_MAX_SECONDS

        if end_by_silence or end_by_maxlen:
            _flush_and_print(model, speaker_state, buf)
            buf = []
            buf_len = 0.0
            silence = 0

    # FINAL FLUSH
    _flush_and_print(model, speaker_state, buf)


# =========================================================
# CLI
# =========================================================




def main():
    # Locked to live microphone + large model.
    live_transcription(model_name="large", device="cpu")


if __name__ == "__main__":
    main()
