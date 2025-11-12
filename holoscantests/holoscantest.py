from holoscan.core import Application, Fragment, Operator, OperatorSpec
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from holoscan.operators import V4L2VideoCaptureOp, HolovizOp


class YOLOCameraOp(Operator):
    """Operator, joka lukee kameran ja suorittaa YOLOv8-inferenssin sekä näyttää tulokset."""
    def __init__(self, fragment, device="/dev/video0", weights_path=None, conf=0.25, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.device = device
        self.weights_path = Path(weights_path) if weights_path else None
        self.conf = conf
        self.model = None
        self.capture = None
        self.frame_idx = 0
        self.display = None  # Holoviz-operaattori

    def setup(self, spec: OperatorSpec):
        # Output-portti Holovizille
        spec.output("video_out")

    def start(self):
        # Luo V4L2-video capture
        self.capture = V4L2VideoCaptureOp(self.fragment, device=self.device, width=640, height=480)
        if self.weights_path is None:
            self.weights_path = Path(__file__).resolve().parent / "runs/weights_coco8/best.pt"
        self.model = YOLO(str(self.weights_path))
        print(f"[YOLOCameraOp] Model loaded from {self.weights_path}")
        # Luo Holoviz display-operaattori
        self.display = HolovizOp(self.fragment, name="display")

    def compute(self, op_input, op_output, context):
        frame_msg = self.capture.read()
        if frame_msg is None:
            return

        frame = np.asarray(frame_msg.get())
        if frame.shape[2] == 3:
            frame = frame[:, :, ::-1]  # RGB -> BGR

        # YOLOv8 inference
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB

        self.frame_idx += 1

        # Emittaa Holovizille suoraan
        op_output.emit(annotated_rgb, "video_out")


# ==========================================
# Pipeline
# ==========================================
if __name__ == "__main__":
    # Luo Holoscan-sovellus
    app = Application()
    fragment = Fragment(app, "video_fragment")
    app.add_fragment(fragment)

    # Luo YOLO + kamera operator
    yolo_camera_op = YOLOCameraOp(fragment, device="/dev/video0", weights_path="runs/weights_coco8/best.pt")
    fragment.add_operator(yolo_camera_op)  # Lisätään fragmenttiin

    # Käynnistä pipeline
    app.run()
