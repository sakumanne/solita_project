"""Live Finnish speech-to-text helper using OpenAI Whisper + simple speaker separation.

Tämä skripti:
- Nauhoittaa mikistä (oletus: 5 sekunnin chunkit)
- Transkriboi puheen suomeksi Whisper small -mallilla
- Käyttää SpeechBrainin speaker-encoderia erottelemaan eri puhujat ilman tokeneita
- Antaa puhujille tunnisteet SPEAKER_00, SPEAKER_01, SPEAKER_02, ...

Huomioita:
- Yhdellä chunkilla (esim. 5 s) on vain yksi puhuja-tunniste. Jos ihmiset puhuvat päällekkäin,
  tämä ei jaa chunkkia kahtia, vaan antaa "dominantin" puhujan chunkille.
- Tämä on "kevyt" online-klusterointi: kun uuden chunkin embedding ei muistuta mitään
  aiempaa puhujaa tarpeeksi hyvin, luodaan uusi SPEAKER_X.

Asennus:
    pip install openai-whisper sounddevice soundfile numpy
    pip install torch
    pip install speechbrain
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
from typing import Iterable, Optional, Union, List, Dict, Any

import numpy as np
import sounddevice as sd
import whisper

import torch
from speechbrain.pretrained import EncoderClassifier

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

# Kuinka paljon historiaa syötetään initial_promptiin (merkkejä)
MAX_HISTORY_CHARS = 500

INITIAL_PROMPT = (
    "Tämä on suomenkielinen keskustelu. Käytä suomen kieltä ja kirjakieltä."
)

AudioInput = Union[np.ndarray, str, Path]


def is_supported_transcript(text: str) -> bool:
    """Return True when the transcript only uses Finnish/English characters."""
    return all((ch in SUPPORTED_CHARACTERS) or ch.isspace() for ch in text)


def rms_to_decibels(rms: float, reference: float = 1.0) -> float:
    """Convert an RMS value to decibels relative to the reference amplitude."""
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms / reference)


def load_whisper_model(name: str) -> whisper.Whisper:
    """Load a Whisper model, printing a helpful message the first time."""
    print(f"Loading Whisper model '{name}'...", file=sys.stderr)
    return whisper.load_model(name)


def load_speaker_encoder(device: str = "cpu") -> EncoderClassifier:
    """Lataa SpeechBrainin speaker-encoderin (ei vaadi tokeneita)."""
    print("Loading SpeechBrain speaker encoder (spkrec-ecapa-voxceleb)...", file=sys.stderr)
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    return encoder


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

    def callback(indata, frames, time_info, status):  # type: ignore[override]
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


# ---------------------------------------------------------------------
# Speaker embedding & online-klusterointi
# ---------------------------------------------------------------------


def get_speaker_embedding(
    encoder: EncoderClassifier,
    audio_chunk: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Laske speaker-embedding yhdelle audiochunkille (mono float32)."""
    # SpeechBrain odottaa tensorin muodossa [batch, time]
    wav = torch.from_numpy(audio_chunk).float().to(device).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(wav)  # shape [1, 1, D] tai [1, D]
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity kahden vektorin välillä."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def assign_speaker(
    emb: np.ndarray,
    speaker_centroids: List[np.ndarray],
    similarity_threshold: float = 0.65,
) -> int:
    """Päätä mihin puhujaan embedding kuuluu, tai luo uusi puhuja.

    - speaker_centroids: lista puhujien "keskivektoreita"
    - palauttaa speaker_id (0, 1, 2, ...)
    - jos mikään centroidi ei ole riittävän lähellä, luodaan uusi puhuja
    """
    if not speaker_centroids:
        speaker_centroids.append(emb.copy())
        return 0

    sims = [cosine_similarity(emb, c) for c in speaker_centroids]
    best_id = int(np.argmax(sims))
    best_sim = sims[best_id]

    if best_sim >= similarity_threshold:
        # Päivitä centroidia vähän uusilla havainnoilla (liukuva keskiarvo)
        speaker_centroids[best_id] = 0.8 * speaker_centroids[best_id] + 0.2 * emb
        return best_id
    else:
        speaker_centroids.append(emb.copy())
        return len(speaker_centroids) - 1


def speaker_label_from_id(speaker_id: int) -> str:
    """Muuta integer-id luettavaan muotoon."""
    return f"SPEAKER_{speaker_id:02d}"


# ---------------------------------------------------------------------
# Päälooppi
# ---------------------------------------------------------------------


def live_transcription(
    model_name: str,
    chunk_duration: float,
    decibel_log: Path,
    device: str = "cpu",
) -> None:
    """Continuously record microphone audio, monitor sound levels and separate speakers."""
    print(
        "Press Ctrl+C to stop. Monitoring sound levels, separating speakers and transcribing speech...\n",
        file=sys.stderr,
    )
    whisper_model = load_whisper_model(model_name)
    speaker_encoder = load_speaker_encoder(device=device)

    print(f"Recording in {int(chunk_duration)}-second chunks...", file=sys.stderr)

    history = ""  # Whisperin tekstihistoria initial_promptia varten

    # Speaker-klusteroinnin tila
    speaker_centroids: List[np.ndarray] = []

    # Aika
    recording_start_time = time.time()
    chunk_index = 0

    try:
        for audio_chunk in record_chunks(chunk_duration):
            if not audio_chunk.size:
                continue

            # Chunkin aika-arvio
            chunk_start_time = recording_start_time + chunk_index * chunk_duration
            chunk_end_time = chunk_start_time + chunk_duration
            chunk_index += 1

            # Laske dB
            rms = float(np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2)))
            if rms <= 0.0:
                db = float("-inf")
            else:
                db = 20.0 * math.log10(rms)

            # Logita dB
            append_decibel_log_entry(decibel_log, db, LOUD_DB_THRESHOLD)

            # Hiljaiset chunkit → ei transkriptiota
            if (not math.isfinite(db)) or db < SILENCE_DB_THRESHOLD:
                transcript = ""
                loud = False
                speaker_id = None
            else:
                loud = db >= LOUD_DB_THRESHOLD

                # Whisper-transkriptio
                transcript = transcribe_audio(
                    whisper_model, audio_chunk, history=history
                )

                # Speaker-embedding ja -id, jos on tekstiä
                if transcript:
                    emb = get_speaker_embedding(speaker_encoder, audio_chunk, device)
                    speaker_id = assign_speaker(emb, speaker_centroids)
                else:
                    speaker_id = None

            # Päivitä historia
            if transcript:
                if history:
                    history = (history + " " + transcript).strip()
                else:
                    history = transcript

                if len(history) > MAX_HISTORY_CHARS:
                    history = history[-MAX_HISTORY_CHARS:]

            # Kirjoita status JSON
            timestamp = datetime.utcnow().isoformat() + "Z"
            _write_status(timestamp, transcript, db, loud)

            # Tulostus
            if transcript:
                if speaker_id is not None:
                    speaker_label = speaker_label_from_id(speaker_id)
                    print(f"{speaker_label}: {transcript}")
                else:
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
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for models (default: cpu, e.g. cuda if GPU available).",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    live_transcription(args.model, args.chunk_duration, args.decibel_log, device=args.device)


if __name__ == "__main__":
    main()
