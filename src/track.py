from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd

from src.simple_features import extract_color_feature
from src.reid_memory import GlobalIDManager


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

    id_manager = GlobalIDManager(
        primary_ttl_frames = 250,
        secondary_ttl_frames = 3000,
        match_threshold = 0.72,
    )

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
            confs = result.boxes.conf.cpu().numpy()

            active_tracker_ids = []

            for box, tracker_id, class_id, conf in zip(
                boxes, track_ids, class_ids, confs
            ):
                x1, y1, x2, y2 = box

                bbox = [x1, y1, x2, y2]

                active_tracker_ids.append(int(tracker_id))

                # Extract color feature from this player crop
                feature = extract_color_feature(frame, bbox)

                # Ask our memory system:
                # Is this a new player or an old player coming back?
                global_id = id_manager.find_match(
                    tracker_id=int(tracker_id),
                    frame_idx=frame_idx,
                    bbox=bbox,
                    feature=feature,
                )

                # Update memory with the latest information
                id_manager.update(
                    tracker_id=int(tracker_id),
                    global_id=global_id,
                    frame_idx=frame_idx,
                    bbox=bbox,
                    feature=feature,
                )

                original_name = model.names[int(class_id)]
                display_name = custom_names.get(original_name, original_name)

                foot_x = (x1 + x2) / 2
                foot_y = y2

                rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": int(tracker_id),
                        "global_id": int(global_id),
                        "class_id": int(class_id),
                        "class_name": original_name,
                        "display_name": display_name,
                        "confidence": float(conf),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "foot_x": float(foot_x),
                        "foot_y": float(foot_y),
                    }
                )

                label = f"{display_name} #{tracker_id} {confs:.2f}"

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
