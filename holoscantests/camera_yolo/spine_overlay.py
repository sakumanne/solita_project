from pathlib import Path
import numpy as np
from ultralytics import YOLO
import time
import json
import joblib
from holoscantests.camera_yolo import spine_math
import cv2

# ---------------- CONFIG ----------------
# Skeleton model path
YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / "runs" / "weights_coco8" / "best.pt"
if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(f"Weights not found: {YOLO_MODEL_PATH}")

KEYPOINT_SIZE = 3
VIRTUAL_SIZE = 4
SPINE_THICKNESS = 1  # Changed from 2 to 1 for thinner lines
PERSON_COLORS = [
    (0, 255, 0),      # Green
    (255, 255, 0),    # Cyan (was red)
    (0, 165, 255),    # Orange
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
]

# COCO order indices for joints we need
COCO_KEYPOINT_IDX = {
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
}

VIRTUAL_SPINE_NAMES = ["CERVICAL_BASE", "THORACIC_MID", "LUMBAR_BASE", "SACRUM"]

# -------------- MODEL LOAD (ONCE) --------------
_model = YOLO(str(YOLO_MODEL_PATH))

# Posture analyzing model
_ML_MODEL_DIR = Path(__file__).resolve().parents[2] / "holoscantests" / "runs"/ "models"
_spine_clf = None
_neck_clf = None
try:
    _spine_clf = joblib.load(_ML_MODEL_DIR / "spine_model.joblib")
    _neck_clf = joblib.load(_ML_MODEL_DIR / "neck_model.joblib")
except Exception:
    _spine_clf = None
    _neck_clf = None
print(f"ML Models loaded - Spine: {_spine_clf is not None}, Neck: {_neck_clf is not None}")
print(f"Model directory: {_ML_MODEL_DIR}")
print(f"Spine model exists: {(_ML_MODEL_DIR / 'spine_model.joblib').exists()}")
print(f"Neck model exists: {(_ML_MODEL_DIR / 'neck_model.joblib').exists()}")


# -------------- HELPERS --------------

def _get_all_person_keypoints(result, img_w, img_h):
    """
    Convert YOLO result.keypoints -> list of per-person normalized keypoints.
    Each entry: (17, 3) array with (x_norm, y_norm, conf).
    """
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return []

    data = getattr(result.keypoints, "data", None)
    if data is None:
        return []

    data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)

    if data.size == 0:
        return []

    if data.ndim == 3 and data.shape[1] >= 17:
        persons = data[:, :17, :]
    elif data.ndim == 2 and data.shape[0] % 17 == 0:
        persons = data.reshape(data.shape[0] // 17, 17, -1)
    else:
        return []

    keypoints = []
    for person in persons:
        xs = person[:, 0] / max(img_w, 1)
        ys = person[:, 1] / max(img_h, 1)
        confs = person[:, 2] if person.shape[1] >= 3 else np.ones_like(xs)
        keypoints.append(np.stack([xs, ys, confs], axis=1))

    return keypoints


def _midpoint(a, b):
    # a, b are (x, y, conf)
    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
        min(a[2], b[2]),
    )


def _compute_virtual_spine(kpt_norm):
    """
    Given one person's normalized keypoints (17,3),
    compute virtual spine points:
    CERVICAL_BASE, THORACIC_MID, LUMBAR_BASE, SACRUM
    Returns list of 4 (x, y, conf) in normalized coords.
    """
    # Guard if we don't have all the indices
    for idx in COCO_KEYPOINT_IDX.values():
        if idx >= kpt_norm.shape[0]:
            return []

    left_shoulder = kpt_norm[COCO_KEYPOINT_IDX["LEFT_SHOULDER"]]
    right_shoulder = kpt_norm[COCO_KEYPOINT_IDX["RIGHT_SHOULDER"]]
    left_hip = kpt_norm[COCO_KEYPOINT_IDX["LEFT_HIP"]]
    right_hip = kpt_norm[COCO_KEYPOINT_IDX["RIGHT_HIP"]]
    left_knee = kpt_norm[COCO_KEYPOINT_IDX["LEFT_KNEE"]]
    right_knee = kpt_norm[COCO_KEYPOINT_IDX["RIGHT_KNEE"]]

    cervical = _midpoint(left_shoulder, right_shoulder)  # top of spine
    lumbar = _midpoint(left_hip, right_hip)              # bottom of rib cage
    thoracic = (                                             # mid-spine
        (cervical[0] + lumbar[0]) / 2.0,
        (cervical[1] + lumbar[1]) / 2.0,
        min(cervical[2], lumbar[2]),
    )
    knees_mid = _midpoint(left_knee, right_knee)
    sacrum = (                                               # lower spine / pelvis
        (lumbar[0] + knees_mid[0]) / 2.0,
        (lumbar[1] + knees_mid[1]) / 2.0,
        min(lumbar[2], knees_mid[2]),
    )

    return [cervical, thoracic, lumbar, sacrum]


def _draw_square(img, cx, cy, size, color):
    h, w, _ = img.shape
    x0, x1 = max(cx - size, 0), min(cx + size + 1, w)
    y0, y1 = max(cy - size, 0), min(cy + size + 1, h)
    img[y0:y1, x0:x1] = color


def _draw_line(img, x0, y0, x1, y1, color, thickness=2):
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, n).astype(int)
    ys = np.linspace(y0, y1, n).astype(int)
    for xi, yi in zip(xs, ys):
        _draw_square(img, xi, yi, thickness, color)


def _draw_text(img, text, x, y, color=(255, 0, 0), size=1):
    """Draw text using cv2."""
    cv2.putText(img, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                size, color, 1)


# -------------- MAIN PUBLIC FUNCTION --------------

def annotate_spine_rgb(rgb_frame):
    """Takes RGB numpy array, returns RGB with overlays with tracking."""
    h, w = rgb_frame.shape[:2]
    annotated = rgb_frame.copy()

    # Use track() instead of predict() for consistent person IDs
    results = _model.track(rgb_frame, verbose=False, persist=True)
    if len(results) == 0:
        return annotated

    result = results[0]
    
    # Get tracked boxes with IDs
    if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
        return annotated
    
    boxes = result.boxes
    track_ids = boxes.id if hasattr(boxes, "id") and boxes.id is not None else None
    
    persons = _get_all_person_keypoints(result, w, h)
    if not persons:
        return annotated

    # First pass: analyze all people to get posture data
    people_analysis = []
    for idx, kpt_norm in enumerate(persons):
        # Get tracked ID or use index as fallback
        track_id = int(track_ids[idx].item()) if track_ids is not None else idx + 1
        
        # Convert to absolute coords for analysis
        abs_kp = []
        for (xn, yn, conf) in kpt_norm:
            abs_kp.append([float(xn * w), float(yn * h), float(conf)])

        vsp_norm = _compute_virtual_spine(kpt_norm)
        vsp_abs = []
        if vsp_norm:
            for (vx, vy, vc) in vsp_norm:
                vsp_abs.append([float(vx * w), float(vy * h), float(vc)])

        posture = spine_math.analyze_person_posture(abs_kp, vsp_abs) if vsp_abs else None
        people_analysis.append((track_id, kpt_norm, vsp_norm, posture))

    # Second pass: draw with colors based on track ID
    for track_id, kpt_norm, spine_pts, posture in people_analysis:
        # Each tracked person gets their own color based on track ID
        color = PERSON_COLORS[(track_id - 1) % len(PERSON_COLORS)]

        # Draw original YOLO keypoints
        for (xn, yn, conf) in kpt_norm:
            if conf < 0.1:
                continue
            px, py = int(xn * w), int(yn * h)
            _draw_square(annotated, px, py, KEYPOINT_SIZE, color)

        if len(spine_pts) != 4:
            continue

        # Draw spine points
        pix_points = []
        for (vx, vy, _) in spine_pts:
            px, py = int(vx * w), int(vy * h)
            pix_points.append((px, py))
            _draw_square(annotated, px, py, VIRTUAL_SIZE, color)

        # Connect spine with lines - consistent thin thickness
        for i in range(len(pix_points) - 1):
            x0, y0 = pix_points[i]
            x1, y1 = pix_points[i + 1]
            _draw_line(annotated, x0, y0, x1, y1, color, thickness=1)

        # Draw warning indicators if bad posture
        is_bad = False
        if posture:
            is_bad = posture.get("bad_posture_ml", posture.get("bad_posture", False))
        
        if is_bad:
            warnings = posture.get("warnings", [])
            nose_x, nose_y = int(kpt_norm[0][0] * w), int(kpt_norm[0][1] * h)

            # Draw warning text above head
            warning_y = max(nose_y - 60, 20)
            for i, warning in enumerate(warnings):
                text_y = warning_y - (i * 25)
                if text_y > 10:
                    _draw_text(annotated, warning, nose_x - 50, text_y, 
                              color=(0, 0, 255), size=0.4)

        # Draw person ID (tracked ID, not index)
        nose_x, nose_y = int(kpt_norm[0][0] * w), int(kpt_norm[0][1] * h)
        _draw_text(annotated, f"ID:{track_id}", nose_x - 20, max(nose_y - 40, 20), 
                   color=color, size=1)

    # record all detected people (absolute coords) to data/keypoints.jsonl
    global _frame_counter
    try:
        people = _build_people_records_from_result(results[0], w, h) if len(results) > 0 else []
        if people:
            _append_keypoints_record(_frame_counter, people)
    except Exception:
        # silently ignore recording errors to avoid disrupting pipeline
        pass
    _frame_counter += 1

    return annotated


def annotate_spine(bgr_frame):
    """Optional: BGR helper (keeps legacy name)."""
    rgb = bgr_frame[:, :, ::-1]
    annotated_rgb = annotate_spine_rgb(rgb)
    return annotated_rgb[:, :, ::-1]


# --- Keypoint JSONL recording helpers ---
_frame_counter = 0

def _get_project_data_path(filename: str = "keypoints.jsonl"):
    base = Path(__file__).resolve().parents[2]  # Go up 2 levels: camera_yolo -> holoscantests -> solita_project
    out_dir = base / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename

def _build_people_records_from_result(result, img_w: int, img_h: int):
    """
    Return list of per-person dicts:
      [{"id": 0, "keypoints": [[x,y,conf],...], "virtual_spine": [[x,y,conf],...]}, ...]
    Coordinates are absolute pixel coords (not normalized).
    """
    persons_norm = _get_all_person_keypoints(result, img_w, img_h)
    if not persons_norm:
        return []

    people = []
    for pid, kpt_norm in enumerate(persons_norm):
        # absolute keypoints
        abs_kp = []
        for (xn, yn, conf) in kpt_norm:
            abs_kp.append([float(xn * img_w), float(yn * img_h), float(conf)])

        # compute virtual spine (normalized), convert to absolute
        vsp_norm = _compute_virtual_spine(kpt_norm)
        vsp_abs = []
        if vsp_norm:
            for (vx, vy, vc) in vsp_norm:
                vsp_abs.append([float(vx * img_w), float(vy * img_h), float(vc)])

        # ANALYZE POSTURE
        posture = spine_math.analyze_person_posture(abs_kp, vsp_abs) if vsp_abs else None
        
        # Apply ML classifiers if available
        if posture and _spine_clf is not None and _neck_clf is not None:
            try:
                feats = [
                    posture["tilt_deg"],
                    posture["curvature_deg"],
                    posture["head_offset"],
                    posture["neck_flexion_deg"],
                ]
                posture["ml_spine_bad"] = bool(_spine_clf.predict([feats])[0])
                posture["ml_neck_bad"] = bool(_neck_clf.predict([feats])[0])
                posture["bad_posture_ml"] = posture["ml_spine_bad"] or posture["ml_neck_bad"]
            except Exception:
                pass

        people.append({
            "id": int(pid),
            "keypoints": abs_kp,
            "virtual_spine": vsp_abs,
            "posture": posture,  # NEW
        })

    return people

def _append_keypoints_record(frame_idx: int | None, people: list, filename: str = "keypoints.jsonl"):
    """
    Append a JSON line with structure:
      {"ts": <float>, "frame": <int|null>, "people": [...]}
    """
    out_path = _get_project_data_path(filename)
    rec = {
        "ts": time.time(),
        "frame": int(frame_idx) if frame_idx is not None else None,
        "people": people,
    }
    # append-safe write
    with open(out_path, "a", encoding="utf8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()

    # Also save posture data to separate file
    _append_posture_record(frame_idx, people)

def _append_posture_record(frame_idx: int | None, people: list):
    """
    Save only posture analysis to separate file: posture.jsonl
    Format: {"ts": <float>, "frame": <int>, "postures": [{"id": 0, "posture": {...}}, ...]}
    """
    out_path = _get_project_data_path("posture.jsonl")
    
    # Extract ALL posture data (including good posture and None)
    postures = []
    for person in people:
        postures.append({
            "id": person["id"],
            "posture": person.get("posture")  # Include even if None or good posture
        })
    
    # Write if there are any people detected
    if postures:
        rec = {
            "ts": time.time(),
            "frame": int(frame_idx) if frame_idx is not None else None,
            "postures": postures,
        }
        with open(out_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()




