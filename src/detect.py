from ultralytics import YOLO
import cv2
from pathlib import Path

VIDEO_PATH = r"data\raw-videos\1.mp4"
MODEL_PATH = r"models\yolo26m.pt"
OUTPUT_PATH = r"outputs\1stdetection_m3.mp4"
CONFIDENCE = float(0.3)

def run_detection(
    video_path: str,
    model_path: str,
    output_path: str,
    conf: float,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        annotated = results[0].plot()

        writer.write(annotated)

    cap.release()
    writer.release()

    print(f"Saved the detection video to: {output_path}")

if __name__ == "__main__":
    run_detection(
        video_path=str(VIDEO_PATH),
        model_path=str(MODEL_PATH),
        output_path=str(OUTPUT_PATH),
        conf=CONFIDENCE
    )