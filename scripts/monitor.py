"""
Unified monitor that reads and analyzes JSONL logs produced by your
separate audio and video apps (jooo.py and main.py). It no longer runs
Whisper or YOLO itself.

What it expects
- Audio JSONL: one JSON object per line with keys like
  {"time":"ISO8601","transcript":"...","loudness_dbfs":-12.3}
- Video JSONL: one JSON object per line with keys like
  {"time":"ISO8601","labels":[{"name":"unsafe_pose","conf":0.83}]}
  or {"labels":["unsafe_pose","nurse"]}

Usage examples
  python monitor.py \
    --audio-config audio_config.json --video-config video_config.json \
    --alerts-csv alerts.csv

  python monitor.py --audio-json audio_out.jsonl --video-json video_out.jsonl \
    --pose-alert-labels unsafe_pose,wrong_transfer \
    --bad-words abuse,help,stop --loudness-threshold-dbfs -15
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
import json
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set


def _dependency_hint(package: str, pip_name: Optional[str] = None) -> str:
    interpreter = Path(sys.executable).resolve()
    return (
        f"Missing dependency: '{package}'. Install with `"
        f"{interpreter} -m pip install {pip_name or package}`."
    )


def dbfs_from_value(val: Optional[float]) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


@dataclass
class Alert:
    source: str  # "audio" or "video"
    message: str
    level: str = "warning"  # or "info", "critical"


def find_default_weights(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def tail_jsonl(path: Path, stop_event: threading.Event, from_start: bool = False) -> Iterable[dict]:
    while not stop_event.is_set():
        if not path.exists():
            time.sleep(0.2)
            continue
        with path.open("r", encoding="utf-8") as f:
            if not from_start:
                f.seek(0, 2)
            while not stop_event.is_set():
                pos = f.tell()
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    f.seek(pos)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as exc:
                    print(f"Failed to parse JSON line in {path}: {exc}", file=sys.stderr)
                    continue


def parse_csv_set(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {s.strip().lower() for s in value.split(",") if s.strip()}


def read_words_file(path: Optional[Path]) -> Set[str]:
    if not path:
        return set()
    if not path.exists():
        raise SystemExit(f"Bad words file not found: {path}")
    words: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.add(line.lower())
    return words


def audio_json_worker(
    alerts: "queue.Queue[Alert]",
    stop_event: threading.Event,
    audio_json_path: Path,
    bad_words: Set[str],
    loudness_threshold_dbfs: float,
    from_start: bool,
) -> None:
    for obj in tail_jsonl(audio_json_path, stop_event, from_start=from_start):
        if stop_event.is_set():
            break
        text = str(obj.get("transcript", ""))
        if text:
            text_low = text.lower()
            if bad_words and any(w in text_low for w in bad_words):
                matches = sorted({w for w in bad_words if w in text_low})
                alerts.put(Alert("audio", f"Flagged words: {', '.join(matches)}"))
        level = dbfs_from_value(obj.get("loudness_dbfs"))
        if level is not None and level >= loudness_threshold_dbfs:
            alerts.put(Alert("audio", f"Loud speech detected: {level:.1f} dBFS"))


def video_json_worker(
    alerts: "queue.Queue[Alert]",
    stop_event: threading.Event,
    video_json_path: Path,
    pose_alert_labels: Set[str],
    from_start: bool,
) -> None:
    for obj in tail_jsonl(video_json_path, stop_event, from_start=from_start):
        if stop_event.is_set():
            break
        labels_field = obj.get("labels")
        detected: Set[str] = set()
        if isinstance(labels_field, list):
            for item in labels_field:
                if isinstance(item, str):
                    detected.add(item.lower())
                elif isinstance(item, dict) and "name" in item:
                    try:
                        detected.add(str(item.get("name")).lower())
                    except Exception:
                        continue
        if pose_alert_labels and detected:
            hit = sorted(pose_alert_labels.intersection(detected))
            if hit:
                alerts.put(Alert("video", f"Pose alert labels detected: {', '.join(hit)}"))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    # First pass: only read config file paths so we can load JSON defaults
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--audio-config", type=Path)
    p0.add_argument("--video-config", type=Path)
    cfg_args, _ = p0.parse_known_args(list(argv) if argv is not None else None)

    p = argparse.ArgumentParser(description="Audio + Video monitor (JSON log analyzer)", parents=[p0])

    # Audio
    p.add_argument("--no-audio", action="store_true", help="Disable audio monitoring")
    p.add_argument("--audio-json", type=Path, help="Path to audio JSONL log (from jooo.py)")
    p.add_argument(
        "--loudness-threshold-dbfs",
        type=float,
        default=-15.0,
        help="Threshold in dBFS to flag loud/yelling (e.g., -15).",
    )
    p.add_argument("--bad-words", help="Comma-separated list of keywords to flag")
    p.add_argument("--bad-words-file", type=Path, help="File with one keyword per line to flag")

    # Video
    p.add_argument("--no-video", action="store_true", help="Disable video monitoring")
    p.add_argument("--video-json", type=Path, help="Path to video JSONL log (from main.py)")
    p.add_argument(
        "--pose-alert-labels",
        help="Comma-separated class names that should trigger a pose alert",
    )
    p.add_argument("--from-start", action="store_true", help="Read logs from the beginning instead of tailing")

    # Alerts persistence
    p.add_argument("--alerts-csv", type=Path, help="Write alerts to CSV at this path")

    # If JSON configs are provided, set them as parser defaults so CLI overrides them
    def _load_json(path: Optional[Path]) -> dict:
        if not path:
            return {}
        if not path.exists():
            raise SystemExit(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise SystemExit(f"Failed to parse JSON config {path}: {exc}") from exc

    audio_cfg = _load_json(getattr(cfg_args, "audio_config", None))
    video_cfg = _load_json(getattr(cfg_args, "video_config", None))

    defaults: dict = {}
    if audio_cfg:
        enabled = audio_cfg.get("enabled")
        if isinstance(enabled, bool):
            defaults["no_audio"] = not enabled
        if "loudness_threshold_dbfs" in audio_cfg:
            defaults["loudness_threshold_dbfs"] = float(audio_cfg["loudness_threshold_dbfs"])
        if "bad_words" in audio_cfg:
            bw = audio_cfg["bad_words"]
            if isinstance(bw, list):
                defaults["bad_words"] = ",".join(map(str, bw))
            elif isinstance(bw, str):
                defaults["bad_words"] = bw
        if "bad_words_file" in audio_cfg:
            defaults["bad_words_file"] = Path(str(audio_cfg["bad_words_file"]))
        if "json_out" in audio_cfg:
            defaults["audio_json"] = Path(str(audio_cfg["json_out"]))
        if "audio_json" in audio_cfg:
            defaults["audio_json"] = Path(str(audio_cfg["audio_json"]))

    if video_cfg:
        enabled = video_cfg.get("enabled")
        if isinstance(enabled, bool):
            defaults["no_video"] = not enabled
        if "pose_alert_labels" in video_cfg:
            pal = video_cfg["pose_alert_labels"]
            if isinstance(pal, list):
                defaults["pose_alert_labels"] = ",".join(map(str, pal))
            elif isinstance(pal, str):
                defaults["pose_alert_labels"] = pal
        if "json_out" in video_cfg:
            defaults["video_json"] = Path(str(video_cfg["json_out"]))
        if "video_json" in video_cfg:
            defaults["video_json"] = Path(str(video_cfg["video_json"]))

    # Allow either config to specify a global alerts_csv path
    if audio_cfg and "alerts_csv" in audio_cfg:
        defaults["alerts_csv"] = Path(str(audio_cfg["alerts_csv"]))
    if video_cfg and "alerts_csv" in video_cfg:
        defaults["alerts_csv"] = Path(str(video_cfg["alerts_csv"]))

    if defaults:
        p.set_defaults(**defaults)

    args = p.parse_args(list(argv) if argv is not None else None)

    # Resolve default JSON paths if not provided
    if not args.no_audio and not args.audio_json:
        args.audio_json = find_default_weights([Path("audio_out.jsonl"), Path("audio.jsonl")])
    if not args.no_video and not args.video_json:
        args.video_json = find_default_weights([Path("video_out.jsonl"), Path("video.jsonl")])

    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    # Build keyword and pose label sets
    bad_words = parse_csv_set(args.bad_words)
    bad_words |= read_words_file(getattr(args, "bad_words_file", None))
    pose_alert_labels = {s.lower() for s in parse_csv_set(args.pose_alert_labels)}

    alerts: "queue.Queue[Alert]" = queue.Queue()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    try:
        if not args.no_audio:
            if not args.audio_json:
                print("Audio monitoring disabled: no --audio-json provided and no default found.", file=sys.stderr)
            else:
                t_audio = threading.Thread(
                    target=audio_json_worker,
                    name="audio-json-worker",
                    daemon=True,
                    kwargs=dict(
                        alerts=alerts,
                        stop_event=stop_event,
                        audio_json_path=args.audio_json,
                        bad_words=bad_words,
                        loudness_threshold_dbfs=args.loudness_threshold_dbfs,
                        from_start=args.from_start,
                    ),
                )
                t_audio.start()
                threads.append(t_audio)

        if not args.no_video:
            if not args.video_json:
                print("Video monitoring disabled: no --video-json provided and no default found.", file=sys.stderr)
            else:
                t_video = threading.Thread(
                    target=video_json_worker,
                    name="video-json-worker",
                    daemon=True,
                    kwargs=dict(
                        alerts=alerts,
                        stop_event=stop_event,
                        video_json_path=args.video_json,
                        pose_alert_labels=pose_alert_labels,
                        from_start=args.from_start,
                    ),
                )
                t_video.start()
                threads.append(t_video)

        # Prepare CSV logging if requested
        csv_writer = None
        csv_file = None
        if getattr(args, "alerts_csv", None):
            path: Path = args.alerts_csv
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists() or path.stat().st_size == 0
            csv_file = path.open("a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            if new_file:
                csv_writer.writerow(["time", "source", "level", "message"])

        print("Monitoring started. Press Ctrl+C to stop.\n")
        # Main loop: print alerts as they arrive
        while any(t.is_alive() for t in threads):
            try:
                alert = alerts.get(timeout=0.5)
            except queue.Empty:
                continue
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] [{alert.source}] {alert.level.upper()}: {alert.message}")
            if csv_writer is not None:
                csv_writer.writerow([ts, alert.source, alert.level, alert.message])
                csv_file.flush()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        # Close CSV handle if open
        try:
            if csv_file is not None:
                csv_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
