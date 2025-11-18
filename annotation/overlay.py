import os
from pathlib import Path

import cv2
import pandas as pd

# ---------- paths ----------
desktop = Path.home() / "Desktop"
csv_path = desktop / "pose_data_with_posture.csv"   # written by curvemath.py
annotated_folder = desktop / "test1_annotated"      # written by annotation.py
output_folder = desktop / "test1_posture"           # final images with metrics
output_folder.mkdir(parents=True, exist_ok=True)

print(f"Reading CSV: {csv_path}")
df = pd.read_csv(csv_path)

if "image" not in df.columns:
    raise ValueError("CSV must have an 'image' column (e.g. IMG_6653.jpg_person1).")

# image column example: 'IMG_6653.jpg_person1'
# -> base image file 'IMG_6653.jpg'
df["base_image"] = df["image"].apply(
    lambda s: s.rsplit("_person", 1)[0].replace(".HEIC", ".jpg").replace(".heic", ".jpg")
)
  # adjust if your naming is different


# group all people that belong to the same frame
for base_img, group in df.groupby("base_image"):
    in_path = annotated_folder / base_img
    if not in_path.exists():
        print(f"Annotated image not found for {base_img}, skipping.")
        continue

    img = cv2.imread(str(in_path))
    if img is None:
        print(f"Failed to read {in_path}, skipping.")
        continue

    h, w = img.shape[:2]

    # add text line per person (person index = row order within group)
    for row_idx, row in group.reset_index(drop=True).iterrows():
        y_offset = 40 + 40 * row_idx  # vertical spacing per person

        # read metrics from curvemath.py output
        tilt = row.get("SPINE_TILT_DEG", float("nan"))
        curv = row.get("SPINE_CURVATURE_DEG", float("nan"))
        hoff = row.get("HEAD_OFFSET", float("nan"))
        nflex = row.get("NECK_FLEXION_DEG", float("nan"))
        spine_bad = int(row.get("SPINE_BAD", 0))
        neck_bad = int(row.get("NECK_BAD", 0))

        # build a short text string
        txt = (
            f"P{row_idx+1} "
            f"Tilt={tilt:.1f}° Curv={curv:.1f}° "
            f"Hoff={hoff:.2f} Neck={nflex:.1f}° "
            f"S{'BAD' if spine_bad else 'OK'} "
            f"N{'BAD' if neck_bad else 'OK'}"
        )

        # choose color: red if any BAD, else green
        if spine_bad or neck_bad:
            color = (0, 0, 255)   # red (BGR)
        else:
            color = (0, 255, 0)   # green

        cv2.putText(
            img,
            txt,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path = output_folder / base_img
    cv2.imwrite(str(out_path), img)
    print(f"Wrote: {out_path}")

print("Done. Check:", output_folder)
