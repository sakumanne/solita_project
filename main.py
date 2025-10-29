from ultralytics import YOLO
import cv2
from pathlib import Path   
from scripts.posescripts.keypointstorer import KeypointWriter, record_if_present

weights_path = Path(__file__).resolve().parent / "runs" / "weights_coco8" / "best.pt"
if not weights_path.exists():
    raise FileNotFoundError(f"Weights not found: {weights_path}")
model = YOLO(str(weights_path))

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible.")
    
writer = KeypointWriter()  # writes to data/keypoints.jsonl by default
frame_idx = 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame)
        annotated = results[0].plot()
        
        # append keypoints for the first detected person (if any)
        record_if_present(results, frame_idx, writer)
        frame_idx += 1

        cv2.imshow("YOLOv8 Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
         
finally:
    writer.close()
    cap.release()
    cv2.destroyAllWindows()