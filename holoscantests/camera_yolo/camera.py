from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.operators.v4l2_camera_passthrough import V4L2CameraPassthroughOp
from holoscan.resources import RMMAllocator
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from .spine_overlay import annotate_spine_rgb


class VideoRecorderOp(Operator):
    """Records video frames to a file."""

    def __init__(self, *args, output_path: str = None, fps: float = 15.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None
        self.frame_count = 0

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def start(self):
        """Initialize video writer."""
        if not self.output_path:
            # Generate default path with timestamp
            recordings_dir = Path(__file__).parent.parent.parent / "recordings"
            recordings_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = str(recordings_dir / f"video_{timestamp}.mp4")
        
        print(f"Video will be saved to: {self.output_path}")

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("in")
        if frame is None:
            op_output.emit(frame, "out")
            return

        # Extract tensor from frame dict
        tensor = None
        if isinstance(frame, dict):
            for key in frame.keys():
                tensor = frame[key]
                break
        else:
            tensor = frame

        if tensor is None:
            op_output.emit(frame, "out")
            return

        # Convert to numpy array
        try:
            try:
                import cupy as cp
                gpu_array = cp.asarray(tensor)
                rgb = cp.asnumpy(gpu_array).astype(np.uint8)
            except (ImportError, AttributeError):
                rgb = np.array(tensor, copy=True).astype(np.uint8)
        except Exception as e:
            op_output.emit(frame, "out")
            return

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            op_output.emit(frame, "out")
            return

        # Initialize video writer on first frame
        if self.video_writer is None:
            height, width = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (width, height)
            )
            print(f"Video recording started: {width}x{height} @ {self.fps}fps")

        # Write frame (OpenCV expects BGR)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr)
        self.frame_count += 1

        # Pass frame through
        op_output.emit(frame, "out")

    def stop(self):
        """Release video writer."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Video recording stopped. Saved {self.frame_count} frames to {self.output_path}")


class SpineOverlayOp(Operator):
    """Applies the YOLO-based spine overlay on RGB frames (NumPy only)."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        frame = op_input.receive("in")
        if frame is None:
            return

        # Frame is a dict - extract the tensor
        tensor = None
        if isinstance(frame, dict):
            # Get the first (and usually only) value
            for key in frame.keys():
                tensor = frame[key]
                break
        else:
            tensor = frame

        if tensor is None:
            op_output.emit(frame, "out")
            return

        # Convert Holoscan Tensor to numpy
        try:
            # Convert Holoscan tensor to numpy array
            # Use CuPy if GPU tensor, otherwise use numpy
            try:
                import cupy as cp
                # Try GPU conversion first
                gpu_array = cp.asarray(tensor)
                rgb = cp.asnumpy(gpu_array).astype(np.uint8)
            except (ImportError, AttributeError):
                # Fallback to direct numpy conversion
                rgb = np.array(tensor, copy=True).astype(np.uint8)
        except Exception as e:
            op_output.emit(frame, "out")
            return

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            op_output.emit(frame, "out")
            return

        try:
            annotated = annotate_spine_rgb(rgb)
            # Convert back to dict format matching input
            op_output.emit({"": annotated}, "out")
        except Exception as e:
            op_output.emit(frame, "out")


class OpenCamera(Application):
    """V4L2 capture → YUYV→RGB conversion → spine overlay → video recording → Holoviz."""
    
    def __init__(self, *args, video_output_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_output_path = video_output_path
    
    def compose(self):
        source_args = self.kwargs("source")
        
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            pass_through=True,
            **source_args,
        )

        # Create format converter for YUYV to RGB conversion
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=RMMAllocator(self, name="rmm-allocator", **self.kwargs("rmm_allocator")),
            in_dtype="yuyv",
            out_dtype="rgb888",
        )

        viz_args = self.kwargs("visualizer")
        if "width" in source_args and "height" in source_args:
            viz_args["width"] = source_args["width"]
            viz_args["height"] = source_args["height"]

        visualizer = HolovizOp(
            self,
            name="visualizer",
            **viz_args,
        )

        passthrough = V4L2CameraPassthroughOp(self, name="passthrough")
        overlay = SpineOverlayOp(self, name="spine_overlay")
        video_recorder = VideoRecorderOp(self, name="video_recorder", output_path=self.video_output_path)

        # Main flow: passthrough -> video_recorder -> visualizer
        self.add_flow(passthrough, video_recorder, {("output", "in")})
        self.add_flow(video_recorder, visualizer, {("out", "receivers")})

        # YUYV path: source -> converter -> overlay -> passthrough
        self.add_flow(source, format_converter, {("signal", "source_video")})
        self.add_flow(format_converter, overlay, {("tensor", "in")})
        self.add_flow(overlay, passthrough, {("out", "input")})

        # Flow for other VideoBuffer formats (NV12, RGB24, etc.) directly compatible with Holoviz
        self.add_flow(source, passthrough, {("signal", "input")})

        def dynamic_flow_callback(op):
            """Route based on V4L2 pixel format metadata.

            YUYV is not supported directly by Holoviz in display drivers >=R550.
            """
            pixel_format = op.metadata.get("V4L2_pixel_format", "")

            if "YUYV" in pixel_format.upper():
                op.add_dynamic_flow("signal", format_converter, "source_video")
            else:
                op.add_dynamic_flow("signal", passthrough, "input")

        self.set_dynamic_flows(source, dynamic_flow_callback)
