import math
import numpy as np
import pandas as pd
from pathlib import Path

# --- config ---
csv_path = Path.home() / "Desktop" / "pose_data.csv"
out_csv_path = Path.home() / "Desktop" / "pose_data_with_posture.csv"

print(f"Reading: {csv_path}")

df = pd.read_csv(csv_path)


def spine_metrics_from_row(row):
    # coordinates are already normalized (x,y in shoulder-width units)
    cerv = np.array([row["CERVICAL_BASE_x"], row["CERVICAL_BASE_y"]], dtype=float)
    thor = np.array([row["THORACIC_MID_x"], row["THORACIC_MID_y"]], dtype=float)
    lumb = np.array([row["LUMBAR_BASE_x"], row["LUMBAR_BASE_y"]], dtype=float)
    sacr = np.array([row["SACRUM_x"], row["SACRUM_y"]], dtype=float)

    # 1) overall tilt (sacrum -> cervical vs vertical)
    vec = cerv - sacr
    dx, dy = float(vec[0]), float(vec[1])
    tilt_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
    tilt_deg = math.degrees(tilt_rad)

    # 2) curvature (sum of angles between segments)
    def unit(v):
        n = np.linalg.norm(v)
        return v / (n + 1e-6)

    v1 = unit(lumb - sacr)
    v2 = unit(thor - lumb)
    v3 = unit(cerv - thor)

    def angle_between(a, b):
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    ang1 = angle_between(v1, v2)
    ang2 = angle_between(v2, v3)
    curvature_deg = ang1 + ang2

    return tilt_deg, curvature_deg


def neck_metrics_from_row(row):
    # nose & spine points (already normalized)
    nose = np.array([row["NOSE_x"], row["NOSE_y"]], dtype=float)
    cerv = np.array([row["CERVICAL_BASE_x"], row["CERVICAL_BASE_y"]], dtype=float)
    sacr = np.array([row["SACRUM_x"], row["SACRUM_y"]], dtype=float)

    spine_vec = cerv - sacr
    spine_len = np.linalg.norm(spine_vec) + 1e-6
    spine_unit = spine_vec / spine_len

    neck_vec = nose - cerv

    # 1) perpendicular offset of head from spine line
    parallel_len = float(np.dot(neck_vec, spine_unit))
    proj = spine_unit * parallel_len
    perp_vec = neck_vec - proj
    head_offset = float(np.linalg.norm(perp_vec))  # in shoulder-width units

    # 2) neck flexion angle
    neck_len = np.linalg.norm(neck_vec) + 1e-6
    neck_unit = neck_vec / neck_len
    dot = float(np.clip(np.dot(neck_unit, spine_unit), -1.0, 1.0))
    neck_flexion_deg = math.degrees(math.acos(dot))

    return head_offset, neck_flexion_deg


# --- compute metrics for every row ---
spine_tilts = []
spine_curvs = []
head_offsets = []
neck_flexes = []

for _, row in df.iterrows():
    tilt, curv = spine_metrics_from_row(row)
    ho, nf = neck_metrics_from_row(row)

    spine_tilts.append(tilt)
    spine_curvs.append(curv)
    head_offsets.append(ho)
    neck_flexes.append(nf)

df["SPINE_TILT_DEG"] = spine_tilts
df["SPINE_CURVATURE_DEG"] = spine_curvs
df["HEAD_OFFSET"] = head_offsets
df["NECK_FLEXION_DEG"] = neck_flexes

# --- simple OK/BAD flags (tune thresholds) ---
BAD_TILT = 12.0
BAD_CURV = 25.0
BAD_HEAD_OFFSET = 0.40
BAD_NECK_FLEXION = 25.0

df["SPINE_BAD"] = (
    (df["SPINE_TILT_DEG"] > BAD_TILT) |
    (df["SPINE_CURVATURE_DEG"] > BAD_CURV)
).astype(int)

df["NECK_BAD"] = (
    (df["HEAD_OFFSET"] > BAD_HEAD_OFFSET) |
    (df["NECK_FLEXION_DEG"] > BAD_NECK_FLEXION)
).astype(int)

print(df[["image", "SPINE_TILT_DEG", "SPINE_CURVATURE_DEG",
          "HEAD_OFFSET", "NECK_FLEXION_DEG",
          "SPINE_BAD", "NECK_BAD"]].head())

df.to_csv(out_csv_path, index=False)
print(f"Saved analyzed CSV to: {out_csv_path}")
