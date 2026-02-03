import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
VENV_PYTHON = BASE / ".venv/bin/python"

procs = [
    subprocess.Popen([VENV_PYTHON, "scripts/mjpeg_server.py"], cwd=BASE),
    subprocess.Popen([VENV_PYTHON, "scripts/live_whisper_speaker.py"], cwd=BASE),
]

try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()