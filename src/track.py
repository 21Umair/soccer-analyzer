from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd


VIDEO_PATH = r"data\raw-videos\1.mp4"
MODEL_PATH = r"models\yolo26m.pt"
OUTPUT_VIDEO_PATH = r"outputs\tracked_custom_botsort.mp4"
OUTPUT_CSV_PATH = r"outputs\tracked_custom_botsort.csv"
CONFIDENCE = float(0.3)

def track_video(
    video_path: str,
    model_path: str,
    output_video_path: str,
    output_csv_path: str,
    conf: float,
):
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    custom_names = {
        "person": "player",
        "sports ball": "football",
    }

    results = model.track(
        source= video_path,
        # tracker= r"configs\custom_bytrack.yaml",
        # tracker= r"botsort.yaml",
        tracker= r"configs\custom_botsort.yaml",
        stream= True,
        persist= True,
        conf=conf,
        iou=0.5,
        verbose=False,
    )

    rows = []
    writer= None
    fps= 25

    for frame_idx, result in enumerate(results):
        frame = result.orig_img.copy()

        if writer is None:
            height, width = frame.shape[:2]

            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if original_fps and original_fps > 0:
                fps = original_fps

            writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, class_id, box_conf in zip(
                boxes, track_ids, class_ids, confidences
            ):
                x1, y1, x2, y2 = box.astype(int)

                original_name = model.names[int(class_id)]
                display_name = custom_names.get(original_name, original_name)

                foot_x = (x1 + x2) / 2
                foot_y = y2

                rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": track_id,
                        "class_id": int(class_id),
                        "class_name": original_name,
                        "display_name": display_name,
                        "confidence": float(box_conf),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "foot_x": foot_x,
                        "foot_y": foot_y,
                    }
                )

                label = f"{display_name} #{track_id} {box_conf:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label_y = max(y1 - 10, 20)
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    (int(foot_x), int(foot_y)),
                    4,
                    (0, 0, 255),
                    -1,
                )

        writer.write(frame)

    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    print(f"Saved tracked video to: {output_video_path}")
    print(f"Saved tracking CSV to: {output_csv_path}")
    print(f"Total tracked detections saved: {len(df)}")

if __name__ == "__main__":
    track_video(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_video_path=OUTPUT_VIDEO_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        conf=CONFIDENCE,
    )
