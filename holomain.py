import os
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from holoscantests.camera_yolo.camera import OpenCamera
from holoscantests.speech_rec.audio_app import AudioTranscriptionApp


def get_output_paths(timestamp=None):
    """Generate synchronized output paths for video and audio."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    recordings_dir = Path(__file__).parent / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = str(recordings_dir / f"video_{timestamp}.mp4")
    audio_path = str(recordings_dir / f"audio_{timestamp}.wav")
    
    return video_path, audio_path


def run_camera(video_path):
    """Run camera capture in separate thread."""
    try:
        app = OpenCamera(video_output_path=video_path)
        config_path = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
        if os.path.exists(config_path):
            print(f"Using camera config: {config_path}")
            app.config(config_path)
        else:
            print("No YAML config found. Running camera with defaults.")
        app.run()
    except Exception as e:
        print(f"Camera error: {e}")


def run_audio(audio_path):
    """Run audio capture in separate thread."""
    try:
        app = AudioTranscriptionApp(audio_output_path=audio_path)
        app.run()
    except Exception as e:
        print(f"Audio error: {e}")


def merge_video_audio(video_path, audio_path):
    """Merge video and audio files into a single MP4 file."""
    print(f"\nChecking files for merge:")
    print(f"  Video exists: {os.path.exists(video_path)} - {video_path}")
    print(f"  Audio exists: {os.path.exists(audio_path)} - {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found. Skipping merge.")
        return None
    
    if not os.path.exists(video_path):
        print(f"Video file not found. Skipping merge.")
        return None
    
    # Output path with same timestamp but merged
    output_path = video_path.replace(".mp4", "_merged.mp4")
    
    try:
        # FFmpeg command to merge video and audio
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",  # Copy video codec (no re-encoding)
            "-c:a", "aac",   # Audio codec
            "-map", "0:v:0",  # Map video from first input
            "-map", "1:a:0",  # Map audio from second input
            "-shortest",      # End when shortest stream ends
            "-y",             # Overwrite output file
            output_path
        ]
        
        print(f"\nMerging video and audio with FFmpeg...")
        print(f"Command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully created: {output_path}")
            
            # Remove original files
            try:
                os.remove(video_path)
                print(f"✓ Removed original video: {video_path}")
            except Exception as e:
                print(f"Could not remove original video: {e}")
            
            try:
                os.remove(audio_path)
                print(f"✓ Removed original audio: {audio_path}")
            except Exception as e:
                print(f"Could not remove original audio: {e}")
            
            return output_path
        else:
            print(f"✗ FFmpeg error (return code {result.returncode}):")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please install FFmpeg to merge video and audio.")
        print("On Linux: sudo apt install ffmpeg")
        print("On macOS: brew install ffmpeg")
        print(f"Video saved separately at: {video_path}")
        print(f"Audio saved at: {audio_path}")
        return None
    except Exception as e:
        print(f"✗ Error merging files: {e}")
        return None


def main():
    print("Starting Holomain - Camera + Audio with Recording")

    print("Press Enter to quit\n")
    
    # Generate synchronized output paths
    video_path, audio_path = get_output_paths()
    print(f"Recording will be saved to:")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}\n")
    
    # Start both in non-daemon threads for graceful shutdown
    camera_thread = threading.Thread(target=run_camera, args=(video_path,), daemon=False)
    camera_thread.start()
    
    audio_thread = threading.Thread(target=run_audio, args=(audio_path,), daemon=False)
    audio_thread.start()
    
    # Wait for Enter key
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    print("\nShutting down...")
    
    # Give threads a moment to stop gracefully
    camera_thread.join(timeout=5)
    audio_thread.join(timeout=5)
    
    print("Application has finished running.")
    
    # Merge video and audio files
    merged_path = merge_video_audio(video_path, audio_path)
    
    if merged_path:
        print(f"\nFinal recording: {merged_path}")
    else:
        print(f"\nRecordings saved separately:")
        print(f"  Video: {video_path}")
        print(f"  Audio: {audio_path}")


if __name__ == "__main__":
    main()