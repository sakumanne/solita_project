from ultralytics import YOLO
import cv2
from pathlib import Path   

weights_path = Path(__file__).resolve().parent / "runs" / "weights_coco8" / "best.pt"
if not weights_path.exists():
    raise FileNotFoundError(f"Weights not found: {weights_path}")
model = YOLO(weights_path)

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame)
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
finally:
    cap.release()
    cv2.destroyAllWindows()