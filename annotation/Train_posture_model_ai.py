from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Where models live (inside annotation/)
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SPINE_MODEL_PATH = MODEL_DIR / "spine_model.joblib"
NECK_MODEL_PATH = MODEL_DIR / "neck_model.joblib"

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
df["GT_NECK_BAD"] = df["GT_NECK_BAD"].astype(int)

feature_cols = [
    "SPINE_TILT_DEG",
    "SPINE_CURVATURE_DEG",
    "HEAD_OFFSET",
    "NECK_FLEXION_DEG",
]

X = df[feature_cols].values
y_spine = df["GT_SPINE_BAD"].values
y_neck = df["GT_NECK_BAD"].values
classes = np.array([0, 1])

def _load_or_init_model(path: Path):
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            print(f"Warning: failed to load {path}, reinitializing model")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("sgd", SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)),
    ])

# ---- SPINE MODEL ----
X_train, X_test, ys_train, ys_test = train_test_split(
    X, y_spine, test_size=0.2, random_state=42, stratify=y_spine
)
spine_clf = _load_or_init_model(SPINE_MODEL_PATH)
spine_clf.partial_fit(X_train, ys_train, classes=classes)
print("SPINE model report:")
print(classification_report(ys_test, spine_clf.predict(X_test)))
spine_clf.partial_fit(X, y_spine, classes=classes)  # update with all data
joblib.dump(spine_clf, SPINE_MODEL_PATH)
print(f"Saved spine model to {SPINE_MODEL_PATH}")

# ---- NECK MODEL ----
X_train, X_test, yn_train, yn_test = train_test_split(
    X, y_neck, test_size=0.2, random_state=42, stratify=y_neck
)
neck_clf = _load_or_init_model(NECK_MODEL_PATH)
neck_clf.partial_fit(X_train, yn_train, classes=classes)
print("NECK model report:")
print(classification_report(yn_test, neck_clf.predict(X_test)))
neck_clf.partial_fit(X, y_neck, classes=classes)  # update with all data
joblib.dump(neck_clf, NECK_MODEL_PATH)
print(f"Saved neck model to {NECK_MODEL_PATH}")
