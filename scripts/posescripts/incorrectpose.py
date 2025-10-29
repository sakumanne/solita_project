from typing import List, Optional, Dict, Tuple
import math

# COCO keypoint indices
_IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}


def _pt(kps: List[List[float]], idx: int, conf_thresh: float) -> Optional[Tuple[float, float, float]]:
    try:
        x, y, c = kps[idx]
        if c is None or c < conf_thresh:
            return None
        return (float(x), float(y), float(c))
    except Exception:
        return None


def _mid(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dx1, dy1 = v1; dx2, dy2 = v2
    dot = dx1*dx2 + dy1*dy2
    n1 = math.hypot(dx1, dy1); n2 = math.hypot(dx2, dy2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def neck_tilt_deg(keypoints: List[List[float]], conf_thresh: float = 0.3) -> Optional[float]:
    """
    Estimate neck/head tilt (degrees) using midpoint of shoulders -> nose vector.
    Returns angle between that vector and vertical (0 degrees = perfectly upright).
    Positive angle means head/nose is forward/down relative to shoulders.
    """
    lt = _pt(keypoints, _IDX["left_shoulder"], conf_thresh)
    rt = _pt(keypoints, _IDX["right_shoulder"], conf_thresh)
    nose = _pt(keypoints, _IDX["nose"], conf_thresh)

    # try fallback to eyes if nose not confident
    if nose is None:
        le = _pt(keypoints, _IDX["left_eye"], conf_thresh)
        re = _pt(keypoints, _IDX["right_eye"], conf_thresh)
        if le and re:
            nose = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0, (le[2] + re[2]) / 2.0)

    if not (lt and rt and nose):
        return None

    mid_sh = _mid((lt[0], lt[1]), (rt[0], rt[1]))
    vec = (nose[0] - mid_sh[0], nose[1] - mid_sh[1])  # shoulders -> nose
    # vertical up direction in image coords is (0, -1)
    vertical = (0.0, -1.0)
    return _angle_between(vec, vertical)


def detect_incorrect_neck(
    keypoints: List[List[float]],
    conf_thresh: float = 0.3,
    neck_thresh_deg: float = 25.0
) -> Dict[str, Optional[object]]:
    """
    Simple neck-bend detector.
    - keypoints: list of [x,y,conf] in COCO order
    - conf_thresh: minimum keypoint confidence to use
    - neck_thresh_deg: threshold angle (degrees) above which we flag 'incorrect'
    Returns:
      { "incorrect": bool, "neck_deg": float|None, "used_conf": float|None }
    """
    deg = neck_tilt_deg(keypoints, conf_thresh)
    used_conf = None
    try:
        # approximate used confidence from nose or eyes if available
        nose = keypoints[_IDX["nose"]][2] if len(keypoints) > _IDX["nose"] else None
        le = keypoints[_IDX["left_eye"]][2] if len(keypoints) > _IDX["left_eye"] else None
        re = keypoints[_IDX["right_eye"]][2] if len(keypoints) > _IDX["right_eye"] else None
        # pick max available
        used_conf = max([c for c in (nose, le, re) if c is not None], default=None)
    except Exception:
        used_conf = None

    incorrect = False
    if deg is not None:
        incorrect = deg > neck_thresh_deg

    return {"incorrect": incorrect, "neck_deg": deg, "used_conf": used_conf}


def is_incorrect_neck(keypoints: List[List[float]], **kwargs) -> bool:
    return bool(detect_incorrect_neck(keypoints, **kwargs).get("incorrect"))