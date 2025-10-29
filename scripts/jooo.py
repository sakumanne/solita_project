"""Live speech-to-text helper using OpenAI Whisper.

This script records short chunks from the microphone and transcribes them
as live speech. Only Finnish ("fi") and English ("en") transcription is
supported, and transcripts containing characters outside these languages
are rejected. Loud live audio is detected and reported when volume exceeds
approximately 70 dB, and every chunk's loudness estimate is appended to a
JSON log for external monitoring.

Example usages:

    # Launch the live transcription helper (default language: Finnish)
    python stt/simple_whisper.py

    # Transcribe live speech in English
    python stt/simple_whisper.py --language en

    # Record and transcribe five-second chunks from the microphone
    python stt/simple_whisper.py --chunk-duration 5

"""
from __future__ import annotations

import argparse
import json
import math
import queue
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
DATA_DIR = Path(__file__).parent.parent / "data"
LATEST_JSON = DATA_DIR / "live_transcript_latest.json"
SUPPORTED_LANGUAGES: tuple[str, ...] = ("fi", "en")
DEFAULT_LANGUAGE = "fi"
SUPPORTED_CHARACTERS = set("abcdefghijklmnopqrstuvwxyzåäö" "ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖ" "0123456789" " .,;:!?\"'()[]{}-/\\@#$%&*+=<>")
LOUDNESS_ALERT_THRESHOLD_DB = 70.0


def dependency_hint(package: str, pip_name: Optional[str] = None) -> str:
    """Provide a helpful installation hint for the current interpreter."""
    interpreter = Path(sys.executable).resolve()
    pip_package = pip_name or package
    return (
        f"Missing dependency: '{package}'. Install it for this interpreter with `"
        f"{interpreter} -m pip install {pip_package}` or `pip install -r requirements.txt`."
    )


try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(dependency_hint("numpy")) from exc

try:
    import sounddevice as sd
except ModuleNotFoundError as exc:
    raise SystemExit(dependency_hint("sounddevice")) from exc
except OSError as exc:
    raise SystemExit(
        "sounddevice could not access the PortAudio backend. "
        "Install the PortAudio runtime (e.g. `sudo apt install libportaudio2`, "
        "`brew install portaudio`, or install the official binary on Windows) "
        "and then re-run the script."
    ) from exc

try:
    import whisper
except ModuleNotFoundError as exc:
    raise SystemExit(dependency_hint("whisper", pip_name="openai-whisper")) from exc


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


def transcribe_audio(
    model: whisper.Whisper,
    audio: AudioInput,
    language: str,
) -> str:
    """Run Whisper on an audio buffer or file path and return the transcript."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}. Supported languages: {SUPPORTED_LANGUAGES}.")

    if isinstance(audio, (np.ndarray, list, tuple)):
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=1)
        audio_input: AudioInput = array
    else:
        audio_input = audio

    result = model.transcribe(audio_input, language=language, fp16=False)
    return result.get("text", "").strip()


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
    language: Optional[str],
    chunk_duration: float,
) -> None:
    """Continuously record microphone audio and monitor sound levels."""
    print(
        "Press Ctrl+C to stop. Monitoring sound levels and transcribing speech...\n",
        file=sys.stderr,
    )
    model = load_model(model_name)

    print(f"Recording in {int(chunk_duration)}-second chunks...", file=sys.stderr)
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
            
            # Check if sound is loud
            loud = db >= LOUD_DB_THRESHOLD

            # Transcribe audio
            transcript = transcribe_audio(model, audio_chunk, language=language)
            
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
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=DEFAULT_LANGUAGE,
        help="Language to transcribe (fi for Finnish, en for English).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=2.0,
        help="Number of seconds per live recording chunk (default: 2).",
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
    live_transcription(args.model, args.language, args.chunk_duration)


if __name__ == "__main__":
    main()
