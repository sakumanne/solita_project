import math
from collections import defaultdict
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment

# ---------- Config ----------
TEMP_AUDIO_PATH = "temp_recording.wav"  # Temporary file to store recording
SAMPLE_RATE = 16000  # Standard sample rate for speech
RECORDING_DURATION = 10  # Recording duration in seconds
WHISPER_MODEL_SIZE = "small"
HF_MODEL_ID = "pyannote/speaker-diarization"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Record audio from microphone
def record_audio():
    print(f"Recording for {RECORDING_DURATION} seconds...")
    recording = sd.rec(
        int(RECORDING_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    
    # Save as WAV file
    write(TEMP_AUDIO_PATH, SAMPLE_RATE, recording)
    return TEMP_AUDIO_PATH

# Record audio before processing
AUDIO_PATH = record_audio()
WHISPER_MODEL_SIZE = "small"   # "base" | "small" | "medium" | "large"
HF_MODEL_ID = "pyannote/speaker-diarization"  # adjust if you use a specific version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Hugging Face token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "YOUR_HUGGINGFACE_TOKEN")
if HF_TOKEN == "YOUR_HUGGINGFACE_TOKEN":
    raise RuntimeError(
        "Missing Hugging Face token. Set HUGGINGFACE_TOKEN in your environment."
    )

# Disable symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------- Load models ----------
print("Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)

print("Loading Pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    HF_MODEL_ID,
    token=HF_TOKEN,
    revision="main"  # required to avoid ValueError
)
pipeline.to(torch.device(DEVICE))

# ---------- Run diarization ----------
print("Running diarization...")
diarization = pipeline(AUDIO_PATH)

# Build a list of (Segment, speaker_label)
dia_segments = [(turn, speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]

# ---------- Transcribe with Whisper ----------
print("Transcribing with Whisper...")
whisper_result = whisper_model.transcribe(
    AUDIO_PATH,
    fp16=(DEVICE == "cuda"),
    condition_on_previous_text=False,
    verbose=False
)

w_segments = whisper_result.get("segments", [])
if not w_segments:
    print("\nNo transcription segments returned by Whisper.")
    exit(0)

# ---------- Helper: assign speaker by maximum overlap ----------
def assign_speaker(start: float, end: float) -> str:
    seg = Segment(start, end)
    overlap_by_speaker = defaultdict(float)
    for dseg, spk in dia_segments:
        ov = dseg & seg
        if ov:
            overlap_by_speaker[spk] += ov.duration
    if not overlap_by_speaker:
        return "SPK-UNK"
    return max(overlap_by_speaker.items(), key=lambda kv: kv[1])[0]

# ---------- Format timestamps ----------
def fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

# ---------- Emit speaker-attributed transcript ----------
print("\n=== Transcription with Speakers ===\n")
last_spk = None
buffer = []

for seg in w_segments:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    spk = assign_speaker(start, end)
    if spk != last_spk and buffer:
        print(buffer[0])
        for line in buffer[1:]:
            print(line)
        print()
        buffer = []

    if not buffer:
        buffer.append(f"[{spk}] {fmt_ts(start)}–{fmt_ts(end)}")
    buffer.append(f"{text}")
    last_spk = spk

# flush remaining
if buffer:
    print(buffer[0])
    for line in buffer[1:]:
        print(line)

# ---------- Write SRT ----------
def to_srt_timestamp(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - math.floor(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

srt_lines = []
idx = 1
for seg in w_segments:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()
    spk = assign_speaker(start, end)
    srt_lines.append(str(idx))
    srt_lines.append(f"{to_srt_timestamp(start)} --> {to_srt_timestamp(end)}")
    srt_lines.append(f"{spk}: {text}")
    srt_lines.append("")
    idx += 1

with open("transcript_with_speakers.srt", "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))

print("\nSaved SRT: transcript_with_speakers.srt")
