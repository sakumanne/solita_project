from holoscan.core import Operator, OperatorSpec
import numpy as np
import cupy as cp  # Holoscan often uses CuPy arrays
from pathlib import Path
from ultralytics import YOLO


class YOLOv8InferenceOp(Operator):
    """Holoscan operator that runs YOLOv8 pose inference using Ultralytics."""
    
    def __init__(self, fragment, *args, weights_path=None, conf=0.25, **kwargs):
        self.weights_path = Path(weights_path) if weights_path else None
        self.conf = conf
        self.model = None
        self.frame_idx = 0
        super().__init__(fragment, *args, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        # Input port receives frames from camera
        spec.input("video_in")
        # Output port sends annotated frames to display
        spec.output("video_out")
        
    def start(self):
        """Load model once when operator starts."""
        if self.weights_path is None:
            self.weights_path = Path(__file__).resolve().parent.parent / "runs" / "weights_coco8" / "best.pt"
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")
        
        self.model = YOLO(str(self.weights_path))
        print(f"[YOLOv8Op] Model loaded from {self.weights_path}")
    
    def compute(self, op_input, op_output, context):
        """Process each frame."""
        # Receive input message
        in_message = op_input.receive("video_in")
        if in_message is None:
            return
        
        # Extract frame - V4L2VideoCaptureOp outputs a VideoBuffer
        # Convert to numpy array (BGR format expected by YOLOv8)
        video_buffer = in_message
        
        # VideoBuffer structure in Holoscan 3.8:
        # Access via .get() method which returns a numpy-like array
        frame = np.asarray(video_buffer.get())
        
        # If frame is GPU tensor (CuPy), transfer to CPU for Ultralytics
        if isinstance(frame, cp.ndarray):
            frame = cp.asnumpy(frame)
        
        # Ensure BGR format (V4L2 typically outputs RGB, YOLO expects BGR)
        # V4L2VideoCaptureOp default is RGB, so convert:
        if frame.shape[2] == 3:  # RGB -> BGR
            frame = frame[:, :, ::-1]
        
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.conf, verbose=False)
        
        # Annotate frame
        annotated = results[0].plot()
        
        self.frame_idx += 1
        
        # Convert back to RGB for HolovizOp (expects RGB)
        annotated_rgb = annotated[:, :, ::-1]
        
        # Emit annotated frame
        op_output.emit(annotated_rgb, "video_out")
    
    def stop(self):
        """Cleanup when operator stops."""
        print(f"[YOLOv8Op] Processed {self.frame_idx} frames.")