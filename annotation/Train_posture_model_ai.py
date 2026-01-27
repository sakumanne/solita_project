from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Where models live (inside annotation/)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SPINE_MODEL_PATH = MODEL_DIR / "spine_model.joblib"
NECK_MODEL_PATH  = MODEL_DIR / "neck_model.joblib"

# Source data
csv_path = Path.home() / "Desktop" / "pose_data_labeled.csv"
print("Reading:", csv_path)
df = pd.read_csv(csv_path)
print("Columns:", df.columns.tolist())

# Validate labels
if "GT_SPINE_BAD" not in df.columns or "GT_NECK_BAD" not in df.columns:
    raise ValueError("GT_SPINE_BAD and/or GT_NECK_BAD missing from CSV")

df = df.dropna(subset=["GT_SPINE_BAD", "GT_NECK_BAD"])
df["GT_SPINE_BAD"] = df["GT_SPINE_BAD"].astype(int)
df["GT_NECK_BAD"]  = df["GT_NECK_BAD"].astype(int)

feature_cols = [
    "SPINE_TILT_DEG",
    "SPINE_CURVATURE_DEG",
    "HEAD_OFFSET",
    "NECK_FLEXION_DEG",
]

X = df[feature_cols].values
y_spine = df["GT_SPINE_BAD"].values
y_neck  = df["GT_NECK_BAD"].values
classes = np.array([0, 1])

def _load_or_init(path: Path):
    if path.exists():
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and "scaler" in obj and "clf" in obj:
                return obj["scaler"], obj["clf"]
        except Exception:
            print(f"Warning: failed to load {path}, reinitializing model")
    scaler = StandardScaler()
    clf = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    return scaler, clf

# ---- SPINE MODEL ----
X_train, X_test, ys_train, ys_test = train_test_split(
    X, y_spine, test_size=0.2, random_state=42, stratify=y_spine
)
spine_scaler, spine_clf = _load_or_init(SPINE_MODEL_PATH)
spine_scaler.partial_fit(X_train)
X_train_s = spine_scaler.transform(X_train)
spine_clf.partial_fit(X_train_s, ys_train, classes=classes)
print("SPINE model report:")
print(classification_report(ys_test, spine_clf.predict(spine_scaler.transform(X_test))))
spine_scaler.partial_fit(X)
spine_clf.partial_fit(spine_scaler.transform(X), y_spine, classes=classes)
joblib.dump({"scaler": spine_scaler, "clf": spine_clf}, SPINE_MODEL_PATH)
print(f"Saved spine model to {SPINE_MODEL_PATH}")

# ---- NECK MODEL ----
X_train, X_test, yn_train, yn_test = train_test_split(
    X, y_neck, test_size=0.2, random_state=42, stratify=y_neck
)
neck_scaler, neck_clf = _load_or_init(NECK_MODEL_PATH)
neck_scaler.partial_fit(X_train)
X_train_s = neck_scaler.transform(X_train)
neck_clf.partial_fit(X_train_s, yn_train, classes=classes)
print("NECK model report:")
print(classification_report(yn_test, neck_clf.predict(neck_scaler.transform(X_test))))
neck_scaler.partial_fit(X)
neck_clf.partial_fit(neck_scaler.transform(X), y_neck, classes=classes)
joblib.dump({"scaler": neck_scaler, "clf": neck_clf}, NECK_MODEL_PATH)
print(f"Saved neck model to {NECK_MODEL_PATH}")