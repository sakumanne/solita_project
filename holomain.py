import os
import threading
from holoscantests.camera_yolo.camera import OpenCamera
from holoscantests.speech_rec.audio_app import AudioTranscriptionApp


def run_camera():
    """Run camera capture in separate thread."""
    try:
        app = OpenCamera()
        config_path = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
        if os.path.exists(config_path):
            print(f"Using camera config: {config_path}")
            app.config(config_path)
        else:
            print("No YAML config found. Running camera with defaults.")
        app.run()
    except Exception as e:
        print(f"Camera error: {e}")


def run_audio():
    """Run audio capture in separate thread."""
    try:
        app = AudioTranscriptionApp()
        app.run()
    except Exception as e:
        print(f"Audio error: {e}")


def main():
    print("Starting Holomain - Camera + Audio")
    print("Press Enter to quit\n")
    
    # Start both in daemon threads
    camera_thread = threading.Thread(target=run_camera, daemon=True)
    camera_thread.start()
    
    audio_thread = threading.Thread(target=run_audio, daemon=True)
    audio_thread.start()
    
    # Wait for Enter key
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    print("\nShutting down...")
    print("Application has finished running.")


if __name__ == "__main__":
    main()