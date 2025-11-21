from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator
import cv2, threading, queue
from ultralytics import YOLO
from pathlib import Path

DISPLAY_Q_SIZE = 8
STOP_EVENT = threading.Event()

class OpenCVCaptureOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam /dev/video0 not accessible.")

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()

    def compute(self, op_input, op_output, context):
        if STOP_EVENT.is_set():
            raise SystemExit
        ok, frame = self.cap.read()
        if not ok:
            return
        op_output.emit(frame, "frame")


class YOLOPoseOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def start(self):
        weights = Path(__file__).resolve().parent.parent / "runs" / "weights_coco8" / "best.pt"
        if not weights.exists():
            raise FileNotFoundError(f"Missing weights: {weights}")
        self.model = YOLO(str(weights))

    def compute(self, op_input, op_output, context):
        if STOP_EVENT.is_set():
            raise SystemExit
        frame = op_input.receive("in")
        if frame is None:
            return
        results = self.model(frame)
        annotated = results[0].plot()
        op_output.emit(annotated, "out")


class FrameSinkOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def start(self):
        if not hasattr(self, "display_queue") or self.display_queue is None:
            self.display_queue = queue.Queue(DISPLAY_Q_SIZE)

    def compute(self, op_input, op_output, context):
        if STOP_EVENT.is_set():
            raise SystemExit
        frame = op_input.receive("in")
        if frame is None:
            return
        try:
            self.display_queue.put_nowait(frame)
        except queue.Full:
            pass


def main():
    app = Application()
    fragment = Fragment(app, "video_fragment")
    app.add_fragment(fragment)

    UnboundedAllocator(fragment, name="allocator")

    cam = OpenCVCaptureOp(fragment, name="opencv_camera")
    fragment.add_operator(cam)

    yolo = YOLOPoseOp(fragment, name="yolo_pose")
    fragment.add_operator(yolo)

    sink = FrameSinkOp(fragment, name="sink")
    fragment.add_operator(sink)

    fragment.add_flow(cam, yolo, {("frame", "in")})
    fragment.add_flow(yolo, sink, {("out", "in")})

    dq = queue.Queue(DISPLAY_Q_SIZE)
    sink.display_queue = dq

    def display_loop():
        while not STOP_EVENT.is_set():
            try:
                frame = dq.get(timeout=0.2)
            except queue.Empty:
                continue
            cv2.imshow("Holoscan YOLO Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                STOP_EVENT.set()
                break
        cv2.destroyAllWindows()

    t = threading.Thread(target=display_loop, daemon=True)
    t.start()

    try:
        app.run()
    finally:
        STOP_EVENT.set()
        t.join(timeout=2)

if __name__ == "__main__":
    main()