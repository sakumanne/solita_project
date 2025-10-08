"""Minimal helper to load a YOLOv8 pose model using Ultralytics.

Run this on your Ubuntu/Holoscan machine to verify the package install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n-pose.pt"


def load_model(model_path: Optional[str] = None) -> YOLO:
    """Load a YOLO pose model.

    Args:
        model_path: Name or path to a YOLOv8 pose checkpoint. Defaults to the
            lightweight `yolov8n-pose.pt` if nothing is provided.

    Returns:
        The loaded `YOLO` model ready for inference or fine-tuning.
    """

    checkpoint = Path(model_path) if model_path else Path(DEFAULT_MODEL)
    return YOLO(checkpoint.as_posix())


if __name__ == "__main__":
    model = load_model()
    print(f"Loaded YOLO pose model: {model.model.args['model']}")
    print(f"Number of keypoints: {model.model.args.get('kpt_shape', 'unknown')}")
