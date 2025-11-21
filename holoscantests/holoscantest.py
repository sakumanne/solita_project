from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.resources import UnboundedAllocator
import cv2, threading, queue, time, os

DISPLAY_Q_SIZE = 8

class OpenCVCaptureOp(Operator):
    """Holoscan source op: captures frames from /dev/video0 using OpenCV."""
    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError("Webcam /dev/video0 not accessible.")
        print("[OpenCVCaptureOp] opened /dev/video0", flush=True)
        self.idx = 0

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
        print("[OpenCVCaptureOp] released", flush=True)

    def compute(self, op_input, op_output, context):
        ok, frame = self.cap.read()
        if not ok:
            print("[OpenCVCaptureOp] read fail", flush=True)
            return
        self.idx += 1
        if self.idx <= 5:
            cv2.imwrite(f"/tmp/hs_cap_{self.idx}.png", frame)
            print(f"[OpenCVCaptureOp] saved /tmp/hs_cap_{self.idx}.png", flush=True)
        op_output.emit(frame, "frame")


class FrameSinkOp(Operator):
    """Holoscan sink op: shows frames with OpenCV so the graph ticks."""
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def start(self):
        self.frame_count = 0
        print("[FrameSinkOp] start", flush=True)

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("in")
        if frame is None:
            return
        self.frame_count += 1
        try:
            self.display_queue.put_nowait(frame)
        except queue.Full:
            pass


def main():
    app = Application()
    fragment = Fragment(app, "video_fragment")
    app.add_fragment(fragment)

    allocator = UnboundedAllocator(fragment, name="allocator")

    cam = OpenCVCaptureOp(fragment, name="opencv_camera")
    fragment.add_operator(cam)

    sink = FrameSinkOp(fragment, name="sink")
    fragment.add_operator(sink)

    fragment.add_flow(cam, sink, {("frame", "in")})

    # Display infrastructure (main thread)
    dq = queue.Queue(DISPLAY_Q_SIZE)
    sink.display_queue = dq  # inject queue

    stop_flag = False

    def display_loop():
        print("[Display] loop start", flush=True)
        last_ts = time.time()
        while not stop_flag:
            try:
                frame = dq.get(timeout=0.2)
            except queue.Empty:
                continue
            cv2.imshow("Holoscan /dev/video0", frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                print("[Display] quit key", flush=True)
                break
            
        cv2.destroyAllWindows()
        print("[Display] loop exit", flush=True)

    disp_thread = threading.Thread(target=display_loop, daemon=True)
    disp_thread.start()

    try:
        app.run()
    finally:
        stop_flag = True
        disp_thread.join(timeout=2)
        print("[Main] done", flush=True)

if __name__ == "__main__":
    main()