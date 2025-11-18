"""Live Finnish speech-to-text helper using OpenAI Whisper.

This script records short chunks from the microphone and transcribes them
as live speech in Finnish. Transcripts containing characters outside the
Finnish/English alphabet and common punctuation are rejected. Loud live
audio is detected and reported when volume exceeds approximately 70 dB,
and every chunk's loudness estimate is appended to a JSON log for external
monitoring.

Example usages:

    # Launch the live transcription helper
    python stt/simple_whisper.py

    # Record and transcribe five-second chunks from the microphone
    python stt/simple_whisper.py --chunk-duration 5

"""
from __future__ import annotations

import argparse
import json
import math
import queue
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16_000
LOUD_DB_THRESHOLD = 70.0
SILENCE_DB_THRESHOLD = -50.0
DATA_DIR = Path(__file__).parent.parent / "data"
LATEST_JSON = DATA_DIR / "live_transcript_latest.json"
SUPPORTED_LANGUAGES: tuple[str, ...] = ("fi",)
DEFAULT_LANGUAGE = "fi"
SUPPORTED_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzåäö"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ"
    "0123456789"
    " .,;:!?\"'()[]{}-/\\@#$%&*+=<>"
)
LOUDNESS_ALERT_THRESHOLD_DB = 70.0

# Kuinka paljon historiaa syötetään initial_promptiin (merkkejä)
MAX_HISTORY_CHARS = 500

INITIAL_PROMPT = (
    "Tämä on suomenkielinen keskustelu. Käytä suomen kieltä ja kirjakieltä."
)


def is_supported_transcript(text: str) -> bool:
    """Return True when the transcript only uses Finnish/English characters."""
    return all((ch in SUPPORTED_CHARACTERS) or ch.isspace() for ch in text)


def rms_to_decibels(rms: float, reference: float = 1.0) -> float:
    """Convert an RMS value to decibels relative to the reference amplitude."""
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms / reference)


def load_model(name: str) -> whisper.Whisper:
    """Load a Whisper model, printing a helpful message the first time."""
    print(f"Loading Whisper model '{name}'...", file=sys.stderr)
    return whisper.load_model(name)


AudioInput = Union[np.ndarray, str, Path]


def postprocess_transcript(text: str) -> str:
    """Lightweight cleanup of the transcript (whitespace, casing, etc.)."""
    text = text.strip()
    if not text:
        return text

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation like " . , ! ? ; :"
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Capitalize first alphabetic character
    first_alpha_idx: Optional[int] = None
    for i, ch in enumerate(text):
        if ch.isalpha():
            first_alpha_idx = i
            break
    if first_alpha_idx is not None:
        text = (
            text[:first_alpha_idx]
            + text[first_alpha_idx].upper()
            + text[first_alpha_idx + 1 :]
        )

    return text


def transcribe_audio(
    model: whisper.Whisper,
    audio: AudioInput,
    history: str = "",
) -> str:
    """Run Whisper on an audio buffer or file path and return the transcript."""
    language = DEFAULT_LANGUAGE

    if isinstance(audio, (np.ndarray, list, tuple)):
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=1)
        audio_input: AudioInput = array
    else:
        audio_input = audio

    # Käytä aiempaa historiaa initial_promptina, jos sitä on
    if history:
        prompt = history[-MAX_HISTORY_CHARS:]
    else:
        prompt = INITIAL_PROMPT

    result = model.transcribe(
        audio_input,
        language=language,
        fp16=False,
        task="transcribe",
        temperature=0.0,
        beam_size=5,
        best_of=5,
        initial_prompt=prompt,
    )
    transcript = result.get("text", "").strip()
    transcript = postprocess_transcript(transcript)

    return transcript if is_supported_transcript(transcript) else ""


def record_chunks(duration: float) -> Iterable[np.ndarray]:
    """Yield successive audio chunks recorded from the microphone."""
    frames_per_chunk = int(duration * SAMPLE_RATE)
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time, status):  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=frames_per_chunk,
        callback=callback,
    ):
        try:
            while True:
                chunk = audio_queue.get()
                yield chunk.reshape(-1)
        except KeyboardInterrupt:
            return


def append_decibel_log_entry(log_file: Path, decibels: float, threshold: float) -> None:
    """Append a JSON line containing the decibel estimate for the chunk."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "decibels": decibels if math.isfinite(decibels) else None,
        "threshold": threshold,
        "is_loud": decibels >= threshold if math.isfinite(decibels) else False,
    }
    with log_file.open("a", encoding="utf-8") as fh:
        json.dump(entry, fh, ensure_ascii=False)
        fh.write("\n")


def _ensure_data_dir() -> None:
    """Create data directory if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_status(timestamp: str, transcript: str, db: float, loud: bool) -> None:
    """Write latest status JSON for monitor to read."""
    _ensure_data_dir()
    status = {
        "timestamp": timestamp,
        "transcript": transcript or "",
        "decibels": None if db == float("-inf") else round(float(db), 2),
        "loud": bool(loud),
    }
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


def live_transcription(
    model_name: str,
    chunk_duration: float,
    decibel_log: Path,
) -> None:
    """Continuously record microphone audio and monitor sound levels."""
    print(
        "Press Ctrl+C to stop. Monitoring sound levels and transcribing speech...\n",
        file=sys.stderr,
    )
    model = load_model(model_name)

    print(f"Recording in {int(chunk_duration)}-second chunks...", file=sys.stderr)

    history = ""  # kerätään aiempia transkriptioita initial_promptia varten

    try:
        for audio_chunk in record_chunks(chunk_duration):
            if not audio_chunk.size:
                continue

            # Compute RMS and convert to dB
            rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
            if rms <= 0.0:
                db = float("-inf")
            else:
                db = 20.0 * math.log10(rms)

            # Log loudness regardless of transcription
            append_decibel_log_entry(decibel_log, db, LOUD_DB_THRESHOLD)

            # Treat silent chunks as silence without hallucinated text
            if (not math.isfinite(db)) or db < SILENCE_DB_THRESHOLD:
                transcript = ""
                loud = False
            else:
                # Check if sound is loud
                loud = db >= LOUD_DB_THRESHOLD

                # Transcribe audio assuming Finnish speech, käyttäen historiaa
                transcript = transcribe_audio(model, audio_chunk, history=history)

            # Päivitä historia vain, jos tuli tekstiä
            if transcript:
                if history:
                    history = (history + " " + transcript).strip()
                else:
                    history = transcript

                # pidä historia riittävän lyhyenä
                if len(history) > MAX_HISTORY_CHARS:
                    history = history[-MAX_HISTORY_CHARS:]

            # Write status for monitor
            timestamp = datetime.utcnow().isoformat() + "Z"
            _write_status(timestamp, transcript, db, loud)

            # Print transcript and warnings
            if transcript:
                print(transcript)
            if loud:
                print(f"WARNING: Loud noise detected ({db:.1f} dB)!", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nStopped monitoring and transcription.", file=sys.stderr)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size to use (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Number of seconds per live recording chunk (default: 5).",
    )
    parser.add_argument(
        "--decibel-log",
        type=Path,
        default=Path("decibel_log.jsonl"),
        help="Path to append JSON loudness samples (default: decibel_log.jsonl).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    live_transcription(args.model, args.chunk_duration, args.decibel_log)


if __name__ == "__main__":
    main()
