from ultralytics import YOLO
import cv2
from pathlib import Path   
from scripts.posescripts.keypointstorer import KeypointWriter, extract_keypoints, record_if_present
from scripts.posescripts.incorrectpose import detect_incorrect_neck

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
        
        # check for incorrect neck and annotate the frame
        kp = extract_keypoints(results[0])
        if kp:
            out = detect_incorrect_neck(kp, conf_thresh=0.3, neck_thresh_deg=25.0)
            if out["incorrect"]:
                # add red text overlay
                cv2.putText(annotated, f"Incorrect Neck: {out['neck_deg']:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frame_idx += 1

        cv2.imshow("YOLOv8 Pose", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
         
finally:
    writer.close()
    cap.release()
    cv2.destroyAllWindows()