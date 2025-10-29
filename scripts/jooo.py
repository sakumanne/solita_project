"""Simple speech-to-text helpers using OpenAI Whisper.

This script supports:
- Transcribing an existing audio file.
- Recording short chunks from the microphone and transcribing them as live speech.

Example usages:

    # Launch the interactive helper (choose file or live inside the script)
    python stt/simple_whisper.py

    # Transcribe an existing audio file (mp3, wav, m4a, etc.)
    python stt/simple_whisper.py --file path/to/audio.mp3

    # Record and transcribe five-second chunks from the microphone
    python stt/simple_whisper.py --live --chunk-duration 5

"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path
from typing import Iterable, Optional, Union


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

SAMPLE_RATE = 16_000


def load_model(name: str) -> whisper.Whisper:
    """Load a Whisper model, printing a helpful message the first time."""
    print(f"Loading Whisper model '{name}'...", file=sys.stderr)
    return whisper.load_model(name)


AudioInput = Union[np.ndarray, str, Path]


def transcribe_audio(
    model: whisper.Whisper,
    audio: AudioInput,
    language: Optional[str] = None,
) -> str:
    """Run Whisper on an audio buffer or file path and return the transcript."""
    if isinstance(audio, (np.ndarray, list, tuple)):
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=1)
        audio_input: AudioInput = array
    else:
        audio_input = audio

    result = model.transcribe(audio_input, language=language, fp16=False)
    return result.get("text", "").strip()


def transcribe_file(path: Path, model_name: str, language: Optional[str]) -> None:
    """Transcribe an audio file and print the transcript."""
    if not path.exists():
        raise SystemExit(f"Audio file not found: {path}")

    model = load_model(model_name)
    print(f"Transcribing '{path}'...", file=sys.stderr)
    transcript = transcribe_audio(model, str(path), language=language)
    if transcript:
        print(transcript)
    else:
        print("(No speech detected)")


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


def live_transcription(model_name: str, language: Optional[str], chunk_duration: float) -> None:
    """Continuously record microphone audio and print transcripts chunk-by-chunk."""
    print(
        "Press Ctrl+C to stop. Speak after the countdown; transcripts will appear below.\n",
        file=sys.stderr,
    )
    model = load_model(model_name)

    print(f"Recording in {int(chunk_duration)}-second chunks...", file=sys.stderr)
    try:
        for audio_chunk in record_chunks(chunk_duration):
            if not audio_chunk.size:
                continue
            transcript = transcribe_audio(model, audio_chunk, language=language)
            if transcript:
                print(transcript)
    except KeyboardInterrupt:
        print("\nStopped live transcription.", file=sys.stderr)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size to use (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--language",
        help="Hint Whisper about the spoken language (e.g. 'en', 'fi').",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to an audio file to transcribe.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Record from the microphone in short chunks and transcribe live speech.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Number of seconds per live recording chunk (default: 5).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.file and not args.live:
        print(
            "No input option provided. Choose whether to transcribe a file or use the live microphone.",
            file=sys.stderr,
        )

        while True:
            try:
                choice = input("Transcribe a recording or use live microphone? [file/live]: ")
            except EOFError as exc:
                raise SystemExit("No option selected; exiting.") from exc

            choice = choice.strip().lower()
            if choice in {"file", "f"}:
                while True:
                    try:
                        file_input = input("Enter the path to the audio file: ")
                    except EOFError as exc:
                        raise SystemExit("No file provided; exiting.") from exc

                    file_input = file_input.strip()
                    if file_input:
                        args.file = Path(file_input).expanduser()
                        break
                    print("Please provide a file path.", file=sys.stderr)
                break
            if choice in {"live", "l"}:
                args.live = True
                break
            print("Please type 'file' or 'live'.", file=sys.stderr)

        if not args.language:
            try:
                language_hint = input(
                    "Enter the language code (e.g. 'fi' for Finnish) or press Enter to auto-detect: "
                )
            except EOFError:
                language_hint = ""
            language_hint = language_hint.strip()
            if language_hint:
                args.language = language_hint

    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    if args.file:
        transcribe_file(args.file, args.model, args.language)
    if args.live:
        live_transcription(args.model, args.language, args.chunk_duration)


if __name__ == "__main__":
    main()