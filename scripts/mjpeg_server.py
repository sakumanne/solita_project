# scripts/mjpeg_server.py
import asyncio
import cv2
import numpy as np
from aiohttp import web
from pathlib import Path
import sys

# Add parent directory to path to import holoscantests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from holoscantests.camera_yolo.spine_overlay import annotate_spine_rgb


async def video_feed(request):
    cap = cv2.VideoCapture(0)  # valitse oikea laite
    boundary = b"--frame"

    async def stream(resp):
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.03)
                continue
            
            # Convert BGR to RGB for annotation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                # Apply spine overlay annotation
                annotated = annotate_spine_rgb(rgb_frame)
                # Convert back to BGR for encoding
                frame_to_encode = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Annotation error: {e}")
                frame_to_encode = frame
            
            ok, buf = cv2.imencode(".jpg", frame_to_encode)
            if not ok:
                continue
            await resp.write(boundary + b"\r\n")
            await resp.write(b"Content-Type: image/jpeg\r\n\r\n")
            await resp.write(buf.tobytes() + b"\r\n")
            await asyncio.sleep(0.03)  # ~30 fps

    resp = web.StreamResponse(
        status=200,
        headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame"},
    )
    await resp.prepare(request)
    await stream(resp)
    return resp


app = web.Application()
app.router.add_get("/video", video_feed)

if __name__ == "__main__":
    print("Starting MJPEG server with spine overlay annotation...")
    print("Video stream available at http://0.0.0.0:5000/video")
    web.run_app(app, host="0.0.0.0", port=5000)