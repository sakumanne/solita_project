from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import V4L2VideoCaptureOp, FormatConverterOp, HolovizOp
from holoscan.resources import UnboundedAllocator
from ultralytics import YOLO
from pathlib import Path
import cv2

class YOLOv8PoseOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def __init__(self, *args, weights_path: str, conf: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_path = weights_path
        self.conf = conf

    def start(self):
        wp = Path(self.weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"Missing weights: {wp}")
        self.model = YOLO(str(wp))

    def compute(self, op_input, op_output, context):
        frame_rgba = op_input.receive("in")
        if frame_rgba is None:
            return
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        results = self.model(frame_bgr, conf=self.conf)
        annotated_bgr = results[0].plot()
        annotated_rgba = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGBA)
        op_output.emit(annotated_rgba, "out")

class YOLOPoseApp(Application):
    def compose(self):
        allocator = UnboundedAllocator(self, name="allocator")

        camera = V4L2VideoCaptureOp(
            self,
            name="camera",
            device="/dev/video0",
            width=640,
            height=480,
            pass_through=True,
        )

        fmt = FormatConverterOp(
            self,
            name="format_converter",
            pool=allocator,
            out_dtype="uint8",
        )

        yolo = YOLOv8PoseOp(
            self,
            name="yolo_pose",
            weights_path="runs/weights_coco8/best.pt",
            conf=0.25,
        )

        viz = HolovizOp(
            self,
            name="holoviz",
            tensors=[{"name": "out", "type": "color"}],
            width=640,
            height=480,
            window_title="YOLO Pose",
            use_exclusive_display=False,
        )

        self.add_flow(camera, fmt, {"signal": "source_video"})
        self.add_flow(fmt, yolo, {"tensor": "in"})
        self.add_flow(yolo, viz, {"out": "receivers"})

if __name__ == "__main__":
    app = YOLOPoseApp()
    app.run()