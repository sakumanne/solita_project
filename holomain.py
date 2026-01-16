import os
from holoscantests.camera_yolo.camera import OpenCamera as oc
#from scripts.WhisperOperator import WhisperOperator as wo


def main():
    app = oc()

    config_path = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
    if os.path.exists(config_path):
        print(f"Using config: {config_path}")
        app.config(config_path)
    else:
        print("No YAML config found. Running with defaults.")

    app.run()
    print("Application has finished running.")


if __name__ == "__main__":
    main()