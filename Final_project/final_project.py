import cv2
import torch
from ultralytics import YOLO
from collections import defaultdict

input_filename = "person_dog.mp4"
output_filename = "person_dog_314512065.mp4"
target_classes = [0, 16]  # person, dog
student_id = "314512065"
model_weights = "yolov8x.pt"

CLASS_COLORS = {
    0: (255, 191, 0),   # person
    16: (0, 140, 255)   # dog
}
DEFAULT_COLOR = (0, 255, 0)

IMGSZ = 1280          
CONF = 0.35           
IOU = 0.55           
MIN_BOX_AREA = 900    
TRACKER_YAML = "bytetrack.yaml"  

def process_video_custom():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_weights)
    model.to(device)
    class_names = model.names

    cap = cv2.VideoCapture(input_filename)
    if not cap.isOpened():
        print("Failed to open video:", input_filename)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    my_font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1.0
    text_x = 30
    base_text_y = 60
    line_height = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_YAML,
            classes=target_classes,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        cls_to_ids = defaultdict(set)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue

                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                    cls_to_ids[cls_id].add(track_id)

                color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
                if track_id is not None:
                    label = f"{name} {conf:.2f} ID:{track_id}"
                else:
                    label = f"{name} {conf:.2f}"

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_top = max(0, y1 - 22)
                cv2.rectangle(frame, (x1, y_top), (x1 + w, y_top + 22), color, -1)
                cv2.putText(frame, label, (x1, y_top + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        text_y = base_text_y

        cv2.putText(frame, f"ID: {student_id}", (text_x + 2, text_y + 2),
                    my_font, font_scale, (50, 50, 50), 2)
        cv2.putText(frame, f"ID: {student_id}", (text_x, text_y),
                    my_font, font_scale, (0, 255, 255), 2)
        text_y += line_height

        if len(cls_to_ids) > 0:
            for cls_id in sorted(cls_to_ids.keys()):
                name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
                count = len(cls_to_ids[cls_id])
                obj_color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                text_str = f"{name} : {count}"

                cv2.putText(frame, text_str, (text_x + 2, text_y + 2),
                            my_font, font_scale, (50, 50, 50), 2)
                cv2.putText(frame, text_str, (text_x, text_y),
                            my_font, font_scale, obj_color, 2)
                text_y += line_height

        out.write(frame)

    cap.release()
    out.release()
    print("Saved:", output_filename)

if __name__ == "__main__":
    process_video_custom()
