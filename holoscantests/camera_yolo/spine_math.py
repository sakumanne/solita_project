import math
import numpy as np

COCO = {
    "NOSE": 0,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
}

def _shoulder_width_units(kpt_norm_17x3):
    """
    Input: (17,3) where x,y are normalized [0..1] in image space.
    Output: dict of points in shoulder-width units, matching your training data.
    Each point: (x_sw, y_sw, conf)
    """
    k = np.asarray(kpt_norm_17x3, dtype=float)
    def get(idx): return (float(k[idx,0]), float(k[idx,1]), float(k[idx,2]))

    ls = get(COCO["LEFT_SHOULDER"])
    rs = get(COCO["RIGHT_SHOULDER"])
    lh = get(COCO["LEFT_HIP"])
    rh = get(COCO["RIGHT_HIP"])

    mid_hip = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0)
    sw = np.hypot(ls[0]-rs[0], ls[1]-rs[1])
    if sw < 1e-6:
        return None  # skip this person

    def norm(pt):
        x,y,c = pt
        return ((x-mid_hip[0])/sw, (y-mid_hip[1])/sw, c)

    pts = {
        "NOSE": norm(get(COCO["NOSE"])),
        "LEFT_SHOULDER": norm(ls),
        "RIGHT_SHOULDER": norm(rs),
        "LEFT_HIP": norm(lh),
        "RIGHT_HIP": norm(rh),
        "LEFT_KNEE": norm(get(COCO["LEFT_KNEE"])),
        "RIGHT_KNEE": norm(get(COCO["RIGHT_KNEE"])),
    }
    return pts

def _norm_virtual_points_to_sw(virtual_pts_norm01, mid_hip_xy, shoulder_width):
    """Convert virtual spine points from normalized [0..1] to shoulder-width units."""
    out = []
    for (x,y,c) in virtual_pts_norm01:
        out.append(((x-mid_hip_xy[0])/shoulder_width, (y-mid_hip_xy[1])/shoulder_width, c))
    return out

def spine_metrics(points):
    """
    Compute spine tilt and curvature from virtual spine points.
    Args:
        points: dict with keys CERVICAL_BASE, THORACIC_MID, LUMBAR_BASE, SACRUM
                each value is [x, y, conf]
    Returns:
        (tilt_deg, curvature_deg)
    """
    cerv = np.array(points["CERVICAL_BASE"][:2], float)
    thor = np.array(points["THORACIC_MID"][:2], float)
    lumb = np.array(points["LUMBAR_BASE"][:2], float)
    sacr = np.array(points["SACRUM"][:2], float)

    vec = cerv - sacr
    dx, dy = float(vec[0]), float(vec[1])
    tilt_deg = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-6)

    v1 = unit(lumb - sacr)
    v2 = unit(thor - lumb)
    v3 = unit(cerv - thor)

    def angle_between(a, b):
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    curvature_deg = angle_between(v1, v2) + angle_between(v2, v3)
    return tilt_deg, curvature_deg

def neck_metrics(points):
    """
    Compute neck forward head offset and flexion angle.
    Args:
        points: dict with keys NOSE, CERVICAL_BASE, SACRUM
    Returns:
        (head_offset, neck_flexion_deg)
    """
    nose = np.array(points["NOSE"][:2], float)
    cerv = np.array(points["CERVICAL_BASE"][:2], float)
    sacr = np.array(points["SACRUM"][:2], float)

    spine_vec = cerv - sacr
    spine_len = np.linalg.norm(spine_vec) + 1e-6
    spine_unit = spine_vec / spine_len

    neck_vec = nose - cerv

    parallel_len = float(np.dot(neck_vec, spine_unit))
    proj = spine_unit * parallel_len
    perp_vec = neck_vec - proj
    head_offset = float(np.linalg.norm(perp_vec))

    neck_len = np.linalg.norm(neck_vec) + 1e-6
    neck_unit = neck_vec / neck_len
    dot = float(np.clip(np.dot(neck_unit, spine_unit), -1.0, 1.0))
    neck_flexion_deg = math.degrees(math.acos(dot))

    return head_offset, neck_flexion_deg

def classify_posture(tilt_deg, curvature_deg, head_offset, neck_flexion_deg):
    """
    Rule-based posture classification.
    Returns: (is_bad_posture: bool, warnings: list[str])
    """
    warnings = []
    
    # Thresholds (adjust based on your needs)
    if tilt_deg > 15:
        warnings.append("spine_tilt_excessive")
    if curvature_deg > 40:
        warnings.append("spine_curvature_excessive")
    if head_offset > 0.3:  # relative to shoulder width units
        warnings.append("forward_head_posture")
    if neck_flexion_deg > 45:
        warnings.append("neck_flexion_excessive")
    
    is_bad = len(warnings) > 0
    return is_bad, warnings

def analyze_person_posture(keypoints_abs, virtual_spine_abs):
    """
    Main analysis function that takes absolute pixel coordinates.
    Args:
        keypoints_abs: list of [x, y, conf] for 17 COCO keypoints
        virtual_spine_abs: list of [x, y, conf] for 4 virtual spine points
    Returns:
        dict with metrics, classification, and warnings
    """
    try:
        if len(keypoints_abs) < 17 or len(virtual_spine_abs) != 4:
            return None
        
        # Build points dict for metrics functions
        kpt = np.array(keypoints_abs)
        points = {
            "NOSE": keypoints_abs[COCO["NOSE"]],
            "CERVICAL_BASE": virtual_spine_abs[0],
            "THORACIC_MID": virtual_spine_abs[1],
            "LUMBAR_BASE": virtual_spine_abs[2],
            "SACRUM": virtual_spine_abs[3],
        }
        
        # Check confidence thresholds
        min_conf = 0.3
        if any(points[k][2] < min_conf for k in ["NOSE", "CERVICAL_BASE", "SACRUM"]):
            return None
        
        # Compute metrics
        tilt, curvature = spine_metrics(points)
        head_offset_px, neck_flexion = neck_metrics(points)
        
        # Normalize head_offset by shoulder width for consistent thresholds
        ls = keypoints_abs[COCO["LEFT_SHOULDER"]]
        rs = keypoints_abs[COCO["RIGHT_SHOULDER"]]
        shoulder_width = np.hypot(ls[0] - rs[0], ls[1] - rs[1])
        head_offset_normalized = head_offset_px / (shoulder_width + 1e-6)
        
        # Classify
        is_bad, warnings = classify_posture(tilt, curvature, head_offset_normalized, neck_flexion)
        
        return {
            "tilt_deg": round(tilt, 2),
            "curvature_deg": round(curvature, 2),
            "head_offset": round(head_offset_normalized, 3),
            "neck_flexion_deg": round(neck_flexion, 2),
            "bad_posture": is_bad,
            "warnings": warnings,
        }
    except Exception as e:
        # Return None silently to avoid pipeline disruption
        return None