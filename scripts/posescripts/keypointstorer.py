from pathlib import Path
import time
import json
from typing import List, Optional, Any, Union


def extract_keypoints(res: Any) -> Optional[List[List[float]]]:
    """
    Extract pose keypoints (first person) as [[x,y,conf], ...] from an ultralytics result.
    Returns None if no person/keypoints are present.
    """
    k = getattr(res, "keypoints", None)
    if k is None:
        return None

    # Preferred: Keypoints.data -> [N, K, 3]
    try:
        if hasattr(k, "data") and k.data is not None:
            arr = k.data
            if hasattr(arr, "cpu"):
                arr = arr.cpu()
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            if len(arr) == 0:
                return None
            first = arr[0]  # [K, 3] -> (x, y, conf)
            return [[float(x), float(y), float(c)] for (x, y, c) in first.tolist()]
    except Exception:
        pass

    # Fallback: combine xy and conf
    try:
        if hasattr(k, "xy") and k.xy is not None and len(k.xy) > 0:
            xy = k.xy[0]
            if hasattr(xy, "cpu"):
                xy = xy.cpu()
            if hasattr(xy, "numpy"):
                xy = xy.numpy()
            conf = None
            if hasattr(k, "conf") and k.conf is not None and len(k.conf) > 0:
                conf = k.conf[0]
                if hasattr(conf, "cpu"):
                    conf = conf.cpu()
                if hasattr(conf, "numpy"):
                    conf = conf.numpy()
                try:
                    import numpy as _np  # type: ignore
                    if hasattr(conf, "ndim") and conf.ndim > 1:
                        conf = _np.squeeze(conf, -1)
                except Exception:
                    pass
            out = []
            K = xy.shape[0]
            for i in range(K):
                c = float(conf[i]) if conf is not None else 1.0
                out.append([float(xy[i, 0]), float(xy[i, 1]), c])
            return out
    except Exception:
        pass

    return None


class KeypointWriter:
    """
    Append per-frame keypoint records to a JSONL file.
    Usage:
      writer = KeypointWriter()  # defaults to project/data/keypoints.jsonl
      writer.write(frame_idx, keypoints, ts=...)
      writer.close()
    """

    def __init__(self, out_path: Union[str, Path] = None):
        if out_path is None:
            # default: project/data/keypoints.jsonl (project root)
            # resolve project root (two levels up from this script: posescripts -> scripts -> project root)
            base = Path(__file__).resolve().parents[2]
            out_dir = base / "data"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "keypoints.jsonl"
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # open in append mode
        self._f = open(self.path, "a", encoding="utf8")

    def write(self, frame: int, keypoints: List[List[float]], ts: Optional[float] = None):
        rec = {
            "ts": ts if ts is not None else time.time(),
            "frame": int(frame),
            "keypoints": keypoints,
        }
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def record_if_present(results: Any, frame_idx: int, writer: KeypointWriter):
    """
    Convenience: extract keypoints from results[0] and write a JSONL record if keypoints are found.
    Returns True if a record was written.
    """
    try:
        kp = extract_keypoints(results[0])
    except Exception:
        kp = None

    if not kp:
        return False

    writer.write(frame_idx, kp)
    return True