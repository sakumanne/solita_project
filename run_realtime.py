from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from holomain import main as run_camera_app


def start_whisper(
    model: str,
    chunk_duration: float,
    decibel_log: Path,
    python: str,
) -> subprocess.Popen:
    script = Path(__file__).parent / "scripts" / "jooo.py"
    cmd = [
        python,
        str(script),
        "--model",
        model,
        "--chunk-duration",
        str(chunk_duration),
        "--decibel-log",
        str(decibel_log),
    ]
    return subprocess.Popen(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Holoscan camera feed and Whisper audio transcription together."
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run only the camera feed (skip Whisper).",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model size (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=5.0,
        help="Number of seconds per live recording chunk.",
    )
    parser.add_argument(
        "--decibel-log",
        type=Path,
        default=Path("decibel_log.jsonl"),
        help="Path to append JSON loudness samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    whisper_proc: Optional[subprocess.Popen] = None

    try:
        if not args.no_audio:
            whisper_proc = start_whisper(
                model=args.model,
                chunk_duration=args.chunk_duration,
                decibel_log=args.decibel_log,
                python=sys.executable,
            )

        run_camera_app()
    finally:
        if whisper_proc and whisper_proc.poll() is None:
            whisper_proc.terminate()
            try:
                whisper_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                whisper_proc.kill()


if __name__ == "__main__":
    main()
