from holoscan.core import Application
from holoscan.operators import V4L2VideoCaptureOp, HolovizOp
import holoscantest  # import operator module

class YOLOPoseApp(Application):
    def compose(self):
        camera = V4L2VideoCaptureOp(
            self,
            name="camera",
            device="/dev/video0",
            width=640,
            height=480,
        )

        yolo = holoscantest.YOLOv8InferenceOp(
            self,
            name="yolo_inference",
            weights_path="runs/weights_coco8/best.pt",
            conf=0.25,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=640,
            height=480,
        )

        # Fixed: use dict mapping for flows
        self.add_flow(camera, yolo, {"signal": "video_in"})
        self.add_flow(yolo, visualizer, {"video_out": "receivers"})

if __name__ == "__main__":
    app = YOLOPoseApp()
    app.run()