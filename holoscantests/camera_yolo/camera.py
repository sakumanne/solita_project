from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.operators.v4l2_camera_passthrough import V4L2CameraPassthroughOp
from holoscan.resources import RMMAllocator
import numpy as np
from .spine_overlay import annotate_spine_rgb


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
    """V4L2 capture → YUYV→RGB conversion → spine overlay → Holoviz."""
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

        # Main flow: passthrough -> visualizer
        self.add_flow(passthrough, visualizer, {("output", "receivers")})

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


