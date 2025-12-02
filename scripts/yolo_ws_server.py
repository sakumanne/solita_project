import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path

import cv2
import websockets
from ultralytics import YOLO

from scripts.posescripts.keypointstorer import extract_keypoints
from scripts.posescripts.incorrectpose import detect_incorrect_neck


def load_model():
    weights_path = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "weights_coco8"
        / "best.pt"
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    print(f"[YOLO WS] Loading model from {weights_path}")
    return YOLO(str(weights_path))


def map_severity(neck_deg: float | None, incorrect: bool) -> str:
    """Muunna niskan kulma + incorrect -> severity."""
    if neck_deg is None:
        return "ok"
    if not incorrect:
        return "ok"
    # yksinkertainen jako:
    if neck_deg < 35:
        return "warn"
    return "critical"


async def stream_yolo(websocket):
    """Lukee kameraa, ajaa YOLOa ja lähettää eventtejä clientille."""
    model = load_model()

    # Windows: käytä DirectShow'ta MSMF:n sijaan
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible.")

    print("[YOLO WS] Webcam opened")

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[YOLO WS] Failed to read frame")
                break

            # YOLO inference
            results = model(frame, verbose=False)

            # niskan kulma keypointeista
            kp = extract_keypoints(results[0])
            neck_deg = None
            incorrect = False
            if kp:
                out = detect_incorrect_neck(
                    kp,
                    conf_thresh=0.3,
                    neck_thresh_deg=25.0,
                )
                neck_deg = float(out.get("neck_deg", 0.0))
                incorrect = bool(out.get("incorrect", False))

            severity = map_severity(neck_deg, incorrect)
            angle = neck_deg if neck_deg is not None else 0.0

            # YOLO annotated frame -> PNG -> Base64
            annotated = results[0].plot()  # BGR
            ok_png, png = cv2.imencode(".png", annotated)
            if not ok_png:
                print("[YOLO WS] Failed to encode frame to PNG")
                continue
            frame_b64 = base64.b64encode(png).decode("utf-8")

            payload = {
                "source": "yolo",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "frame_idx": frame_idx,
                "angle": angle,
                "severity": severity,
                "incorrect": incorrect,
                "frame": frame_b64,  # 🔥 kuva
            }

            await websocket.send(json.dumps(payload))

            frame_idx += 1

            # anna vähän hengähdystaukoa event-loopille
            await asyncio.sleep(0.01)

    except websockets.ConnectionClosed:
        print("[YOLO WS] Client disconnected")
    finally:
        cap.release()
        print("[YOLO WS] Webcam released")


async def handler(websocket):
    print("[YOLO WS] Client connected")
    await stream_yolo(websocket)


async def main():
    print("Starting YOLO WebSocket server on ws://localhost:8765")
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
