import cv2
import csv
import os
import numpy as np
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

# Replace MediaPipe with Ultralytics YOLO Pose
try:
    from ultralytics import YOLO
except Exception as e:
    print("Please install ultralytics: pip install ultralytics")
    raise

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
CUSTOM_WEIGHTS = PROJECT_ROOT / "solita_project" / "runs" / "weights_coco8" / "best.pt"
FULL_YOLO_MODEL = "yolov8x-pose.pt"  # full-capacity YOLOv8 pose checkpoint
if CUSTOM_WEIGHTS.exists():
    YOLO_MODEL = str(CUSTOM_WEIGHTS)
else:
    YOLO_MODEL = FULL_YOLO_MODEL
print(f"Using YOLO pose weights: {YOLO_MODEL}")
image_folder = os.path.join(os.path.expanduser("~"), "Desktop", "test1")
annotated_folder = os.path.join(os.path.expanduser("~"), "Desktop", "test1_annotated")
os.makedirs(annotated_folder, exist_ok=True)
output_csv = os.path.join(os.path.expanduser("~"), "Desktop", "pose_data.csv")
KEYPOINT_RADIUS = 15
VIRTUAL_RADIUS = 12
LABEL_FONT_SCALE = 2
LABEL_THICKNESS = 5
PERSON_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
]

# landmarks we care about (COCO order indices will be used below)
COCO_KEYPOINT_IDX = {
    "NOSE": 0,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7, "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9, "RIGHT_WRIST": 10,
    "LEFT_HIP": 11, "RIGHT_HIP": 12,
    "LEFT_KNEE": 13, "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15, "RIGHT_ANKLE": 16,
}

important_names = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
]

virtual_spine_landmarks = ["CERVICAL_BASE", "THORACIC_MID", "LUMBAR_BASE", "SACRUM"]

if not os.path.isdir(image_folder):
    print(f"Error: image folder not found: {image_folder}")
    sys.exit(1)

interactive = sys.stdin.isatty()

# Load YOLO pose model
model = YOLO(YOLO_MODEL)

def get_all_person_keypoints(result, img_w, img_h):
    """
    Extract every detected person's keypoints from the Ultralytics result object.
    Returns a list with one normalized (17, 3) array per person.
    """
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return []

    data = getattr(result.keypoints, "data", None)
    if data is None:
        return []

    if hasattr(data, "cpu"):
        data = data.cpu().numpy()
    else:
        data = np.asarray(data)

    if data.size == 0:
        return []

    if data.ndim == 3 and data.shape[1] >= 17:
        persons = data[:, :17, :]
    elif data.ndim == 2 and data.shape[0] % 17 == 0:
        num_people = data.shape[0] // 17
        persons = data.reshape(num_people, 17, -1)
    else:
        return []

    keypoints = []
    for person in persons:
        xs = person[:, 0] / max(img_w, 1)
        ys = person[:, 1] / max(img_h, 1)
        confs = person[:, 2] if person.shape[1] >= 3 else np.ones_like(xs)
        kpt_norm = np.stack([xs, ys, confs], axis=1)
        keypoints.append(kpt_norm)

    return keypoints

def load_image(path):
    """
    Try to load an image with OpenCV; fall back to Pillow for HEIC/HEIF files.
    """
    image = cv2.imread(path)
    if image is not None:
        return image

    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        if Image is None:
            print(
                "Error: Pillow is required to read HEIC files. Install via 'pip install pillow'."
            )
            return None
        if pillow_heif is None:
            print(
                "Error: pillow-heif is required for HEIC support. Install via 'pip install pillow-heif'."
            )
            return None
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                rgb = np.array(pil_img)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            print(f"Error: Failed to decode HEIC image {path}: {exc}")
            return None

    return None

def midpoint_kpt(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, min(a[2], b[2]))

with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)

    header = ["image"]
    for name in important_names:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    for name in virtual_spine_landmarks:
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    header.append("label")
    writer.writerow(header)

    supported_exts = (".jpg", ".jpeg", ".png", ".heic", ".heif")
    image_filenames = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(supported_exts)
    ])

    if not image_filenames:
        print(f"No supported images ({', '.join(supported_exts)}) found in {image_folder}")
        sys.exit(1)

    processed_images = 0
    saved_images = 0

    for filename in image_filenames:
        processed_images += 1
        print(f"Processing {filename}...")

        image_path = os.path.join(image_folder, filename)
        image = load_image(image_path)
        if image is None:
            print(f"Error: Could not load {filename}")
            continue

        h, w = image.shape[:2]

        # run YOLO pose inference
        results = model(image)  # returns a Results object list; supports numpy image directly
        if len(results) == 0:
            print(f"No results from YOLO for {filename}")
            continue

        res0 = results[0]
        all_keypoints = get_all_person_keypoints(res0, w, h)
        if not all_keypoints:
            print(f"No pose keypoints detected in {filename}")
            continue

        lower = filename.lower()
        if "patient" in lower:
            label = "patient"
        elif "nurse" in lower:
            label = "nurse"
        elif interactive:
            label = input(f"Enter label for {filename}: ")
        else:
            label = ""

        person_annotations = []
        for person_idx, kpt_norm in enumerate(all_keypoints, start=1):
            # build a dict-like access similar to mediapipe.landmark with x,y,z,visibility
            landmarks = {}
            for name, idx in COCO_KEYPOINT_IDX.items():
                if idx < kpt_norm.shape[0]:
                    x, y, conf = kpt_norm[idx]
                    landmarks[name] = (x, y, 0.0, float(conf))
                else:
                    landmarks[name] = (0.0, 0.0, 0.0, 0.0)

            # compute mid-hip and shoulder width using normalized coords
            left_hip = landmarks["LEFT_HIP"]
            right_hip = landmarks["RIGHT_HIP"]
            left_shoulder = landmarks["LEFT_SHOULDER"]
            right_shoulder = landmarks["RIGHT_SHOULDER"]
            left_knee = landmarks["LEFT_KNEE"]
            right_knee = landmarks["RIGHT_KNEE"]

            mid_hip_x = (left_hip[0] + right_hip[0]) / 2
            mid_hip_y = (left_hip[1] + right_hip[1]) / 2
            mid_hip_z = (left_hip[2] + right_hip[2]) / 2

            shoulder_width = np.linalg.norm([left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1]])
            if shoulder_width < 1e-6:
                print(f"Warning: Small shoulder width detected in {filename} (person {person_idx}). Skipping this person.")
                continue

            row = [f"{filename}_person{person_idx}"]
            for name in important_names:
                x, y, z, vis = landmarks[name]
                norm_x = (x - mid_hip_x) / shoulder_width
                norm_y = (y - mid_hip_y) / shoulder_width
                norm_z = (z - mid_hip_z) / shoulder_width
                row.extend([norm_x, norm_y, norm_z, vis])

            cervical = midpoint_kpt(left_shoulder, right_shoulder)
            lumbar = midpoint_kpt(left_hip, right_hip)
            thoracic = ((cervical[0] + lumbar[0]) / 2, (cervical[1] + lumbar[1]) / 2, min(cervical[2], lumbar[2]))
            knees_mid = midpoint_kpt(left_knee, right_knee)
            sacrum = ((lumbar[0] + knees_mid[0]) / 2, (lumbar[1] + knees_mid[1]) / 2, min(lumbar[2], knees_mid[2]))

            virtual_points = [cervical, thoracic, lumbar, sacrum]
            for (vx, vy, vz) in virtual_points:
                row.extend([
                    (vx - mid_hip_x) / shoulder_width,
                    (vy - mid_hip_y) / shoulder_width,
                    (vz - mid_hip_z) / shoulder_width,
                    1.0,  # visibility for virtual points (heuristic)
                ])

            row.append(label)
            writer.writerow(row)
            person_annotations.append({
                "keypoints": kpt_norm,
                "virtual_points": virtual_points,
                "person_idx": person_idx,
            })

        if not person_annotations:
            print(f"No valid people written for {filename}; all detections skipped.")
            continue

        # annotate image: draw keypoints + virtual spine
        annotated = image.copy()
        for ann in person_annotations:
            color = PERSON_COLORS[(ann["person_idx"] - 1) % len(PERSON_COLORS)]
            for x, y, conf in ann["keypoints"]:
                px = int(x * w)
                py = int(y * h)
                if conf > 0.1:
                    cv2.circle(annotated, (px, py), KEYPOINT_RADIUS, color, -1)

            for name, (vx, vy, vz) in zip(virtual_spine_landmarks, ann["virtual_points"]):
                px = int(vx * w)
                py = int(vy * h)
                cv2.circle(annotated, (px, py), VIRTUAL_RADIUS, color, -1)
                cv2.putText(
                    annotated,
                    f"{name}_{ann['person_idx']}",
                    (px + VIRTUAL_RADIUS, py - VIRTUAL_RADIUS // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    LABEL_FONT_SCALE,
                    color,
                    LABEL_THICKNESS,
                    cv2.LINE_AA,
                )

        base_name, _ = os.path.splitext(filename)
        out_filename = f"{base_name}.jpg"
        out_path = os.path.join(annotated_folder, out_filename)
        if cv2.imwrite(out_path, annotated):
            saved_images += 1
            print(f"Annotated image written to {out_path}")
        else:
            print(f"Error: Failed to write annotated image for {filename}")

        if interactive:
            cv2.imshow("Annotated", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print(f"Pose CSV saved to {output_csv}")
print(f"Processed {processed_images} images.")
print(f"Annotated images saved to {annotated_folder} ({saved_images} files).")
if saved_images == 0:
    print("No annotated images were written; review the warnings above to find the failure point.")
