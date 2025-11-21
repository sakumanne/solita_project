from pathlib import Path
import pandas as pd

# Input = your automatically generated CSV
csv_in = Path.home() / "Desktop" / "pose_data_with_posture.csv"
# Output = your manually labeled CSV
csv_out = Path.home() / "Desktop" / "pose_data_labeled.csv"

df = pd.read_csv(csv_in)

# Create GT columns if missing
if "GT_SPINE_BAD" not in df.columns:
    df["GT_SPINE_BAD"] = pd.NA
if "GT_NECK_BAD" not in df.columns:
    df["GT_NECK_BAD"] = pd.NA

print("\n📌 Starting manual labeling\n")

for i, row in df.iterrows():
    print(f"\nRow {i}: {row['image']}")
    print("---------------------------------------")
    print(f"  Spine tilt:       {row['SPINE_TILT_DEG']:.2f}")
    print(f"  Spine curvature:  {row['SPINE_CURVATURE_DEG']:.2f}")
    print(f"  Head offset:      {row['HEAD_OFFSET']:.2f}")
    print(f"  Neck flexion:     {row['NECK_FLEXION_DEG']:.2f}")

    # Skip already labeled rows
    if pd.notna(row["GT_SPINE_BAD"]) and pd.notna(row["GT_NECK_BAD"]):
        print("  ✔ Already labeled, skipping.")
        continue

    # Ask user for label input
    spine = input("  Spine BAD? (1=bad, 0=good, Enter=skip): ").strip()
    neck  = input("  Neck BAD?  (1=bad, 0=good, Enter=skip): ").strip()

    # Store values only if provided
    if spine in ("0", "1"):
        df.at[i, "GT_SPINE_BAD"] = int(spine)
    if neck in ("0", "1"):
        df.at[i, "GT_NECK_BAD"] = int(neck)

# Remove rows without labels
df = df.dropna(subset=["GT_SPINE_BAD", "GT_NECK_BAD"])

# Convert labels to integers
df["GT_SPINE_BAD"] = df["GT_SPINE_BAD"].astype(int)
df["GT_NECK_BAD"] = df["GT_NECK_BAD"].astype(int)

# Save clean CSV
df.to_csv(csv_out, index=False)

print(f"\n✅ Labeled CSV saved to:\n{csv_out}")
