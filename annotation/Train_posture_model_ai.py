from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

desktop = Path.home() / "Desktop"
csv_path = desktop / "pose_data_labeled.csv"

print("Reading:", csv_path)
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())

# Make sure labels exist and are clean
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

# ---- SPINE MODEL ----
X_train, X_test, ys_train, ys_test = train_test_split(
    X, y_spine, test_size=0.2, random_state=42, stratify=y_spine
)

spine_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

spine_clf.fit(X_train, ys_train)
print("SPINE model report:")
print(classification_report(ys_test, spine_clf.predict(X_test)))
joblib.dump(spine_clf, desktop / "spine_model.joblib")
print("Saved spine_model.joblib")

# ---- NECK MODEL ----
X_train, X_test, yn_train, yn_test = train_test_split(
    X, y_neck, test_size=0.2, random_state=42, stratify=y_neck
)

neck_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

neck_clf.fit(X_train, yn_train)
print("NECK model report:")
print(classification_report(yn_test, neck_clf.predict(X_test)))
joblib.dump(neck_clf, desktop / "neck_model.joblib")
print("Saved neck_model.joblib")
