from pathlib import Path
import csv
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# YOLO 没有输出合格圆心：只有绿框，没有红点。
# YOLO 输出了圆心：Track Model 必须分配旧 ID 或 NEW，并显示红点和 ID。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from coordinate_association_model import (
    CONTEXT_RADIUS,
    assign_ids_from_logits,
    build_temporal_context,
    load_coordinate_model,
    normalize_detection_features,
)


YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "ring_pose"
    / "yolo26n_pose_ring"
    / "weights"
    / "best_ring_class0.pt"
)
TRACK_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "coordinate_tracker"
    / "best_coordinate_association.pt"
)
INPUT_VIDEO = PROJECT_ROOT / "assets" / "input" / "27.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "assets" / "output" / "27.mp4"
OUTPUT_CSV = OUTPUT_VIDEO.with_name(f"{OUTPUT_VIDEO.stem}_points.csv")

IMAGE_SIZE = 640
DETECTION_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.5

BOX_OVERLAP_THRESHOLD = 0.70
POINT_MIN_DISTANCE = 7.0

POINT_RADIUS = 6
POINT_COLOR = (0, 0, 255)
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
ID_COLOR = (255, 255, 255)
ID_OUTLINE_COLOR = (0, 0, 0)
ID_FONT_SCALE = 0.65
ID_THICKNESS = 2

DEVICE = None


def choose_device():
    if DEVICE is not None:
        return DEVICE
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def write_point_csv(csv_path, frame_records, point_ids):
    fieldnames = ["frame_index", "timestamp_sec"]
    for point_id in point_ids:
        fieldnames.extend(
            [f"id_{point_id}_x_px", f"id_{point_id}_y_px"]
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for frame_index, timestamp_sec, visible_points in frame_records:
            row = {
                "frame_index": frame_index,
                "timestamp_sec": f"{timestamp_sec:.9f}",
            }
            for point_id, (x, y) in visible_points.items():
                row[f"id_{point_id}_x_px"] = x
                row[f"id_{point_id}_y_px"] = y
            writer.writerow(row)


def box_overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0
    return intersection_area / smaller_area


def extract_ring_detections(result):
    if result.boxes is None or result.boxes.xyxy is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    box_confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    points = None
    point_confidences = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        points = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is not None:
            point_confidences = result.keypoints.conf.cpu().numpy()

    detections = []
    for index, box in enumerate(boxes):
        if class_ids[index] != 0:
            continue

        point = None
        point_confidence = 0.0
        if points is not None and index < len(points) and len(points[index]) > 0:
            center_x, center_y = points[index][0]
            if point_confidences is not None:
                point_confidence = float(point_confidences[index][0])
            else:
                point_confidence = 1.0

            if (
                center_x > 0
                and center_y > 0
                and point_confidence >= KEYPOINT_CONFIDENCE
            ):
                point = np.array([center_x, center_y], dtype=np.float32)

        detections.append(
            {
                "box": box.astype(np.float32),
                "box_confidence": float(box_confidences[index]),
                "point": point,
                "point_confidence": point_confidence,
            }
        )

    detections.sort(key=lambda detection: detection["box_confidence"], reverse=True)

    filtered_boxes = []
    for detection in detections:
        overlaps_existing_box = any(
            box_overlap_ratio(detection["box"], kept["box"])
            > BOX_OVERLAP_THRESHOLD
            for kept in filtered_boxes
        )
        if not overlaps_existing_box:
            filtered_boxes.append(detection)

    point_order = sorted(
        range(len(filtered_boxes)),
        key=lambda index: (
            filtered_boxes[index]["point_confidence"],
            filtered_boxes[index]["box_confidence"],
        ),
        reverse=True,
    )

    kept_points = []
    for index in point_order:
        point = filtered_boxes[index]["point"]
        if point is None:
            continue

        too_close = any(
            np.linalg.norm(point - kept_point) < POINT_MIN_DISTANCE
            for kept_point in kept_points
        )
        if too_close:
            filtered_boxes[index]["point"] = None
        else:
            kept_points.append(point)

    filtered_boxes.sort(
        key=lambda detection: (
            (detection["box"][1] + detection["box"][3]) / 2,
            (detection["box"][0] + detection["box"][2]) / 2,
        )
    )
    return filtered_boxes


def detection_feature(detection):
    x1, y1, x2, y2 = detection["box"]
    point = detection["point"]
    return np.array(
        [
            point[0],
            point[1],
            detection["box_confidence"],
            detection["point_confidence"],
            x2 - x1,
            y2 - y1,
        ],
        dtype=np.float32,
    )


def pack_video_detection_features(all_frame_detections, width, height):
    point_detections = [
        [detection for detection in detections if detection["point"] is not None]
        for detections in all_frame_detections
    ]
    max_detections = max(
        1,
        max((len(detections) for detections in point_detections), default=0),
    )

    frame_count = len(point_detections)
    features = np.zeros(
        (frame_count, max_detections, 6),
        dtype=np.float32,
    )
    masks = np.zeros((frame_count, max_detections), dtype=bool)

    for frame_index, detections in enumerate(point_detections):
        for detection_index, detection in enumerate(detections):
            features[frame_index, detection_index] = detection_feature(detection)
            masks[frame_index, detection_index] = True

    normalized = normalize_detection_features(features, width, height)
    return normalized, masks


class NeuralPointTracker:
    def __init__(self, model_path, device):
        self.model, checkpoint = load_coordinate_model(model_path, device)
        self.device = device
        self.training_max_tracks = int(checkpoint["max_tracks"])
        self.history_length = int(checkpoint["history_length"])
        checkpoint_radius = int(checkpoint.get("context_radius", -1))
        if checkpoint_radius != CONTEXT_RADIUS:
            raise RuntimeError(
                f"模型前后帧半径为{checkpoint_radius}，当前代码为{CONTEXT_RADIUS}"
            )

        self.history = np.zeros(
            (0, self.history_length, 3),
            dtype=np.float32,
        )
        self.track_ids = np.zeros(0, dtype=np.int64)
        self.next_id = 1

    @property
    def used_id_count(self):
        return self.next_id - 1

    def update(self, detection_features, detection_context):
        detection_features = np.asarray(detection_features, dtype=np.float32)
        detection_context = np.asarray(detection_context, dtype=np.float32)
        detection_count = len(detection_features)
        track_count = len(self.history)

        point_ids = []
        assigned_track_slots = []

        if detection_count > 0:
            track_mask = np.ones(track_count, dtype=bool)
            detection_mask = np.ones(detection_count, dtype=bool)

            history_tensor = torch.from_numpy(self.history).unsqueeze(0).to(self.device)
            track_mask_tensor = (
                torch.from_numpy(track_mask).unsqueeze(0).to(self.device)
            )
            detection_tensor = (
                torch.from_numpy(detection_features).unsqueeze(0).to(self.device)
            )
            context_tensor = (
                torch.from_numpy(detection_context).unsqueeze(0).to(self.device)
            )
            detection_mask_tensor = (
                torch.from_numpy(detection_mask).unsqueeze(0).to(self.device)
            )

            with torch.inference_mode():
                logits = self.model(
                    history_tensor,
                    track_mask_tensor,
                    detection_tensor,
                    context_tensor,
                    detection_mask_tensor,
                )[0].detach().cpu().numpy()

            point_id_array, next_id = assign_ids_from_logits(
                logits,
                track_mask,
                next_track_id=self.next_id,
                detection_mask=detection_mask,
            )
            point_ids = [int(track_id) for track_id in point_id_array]

            track_slot_by_id = {
                int(track_id): track_slot
                for track_slot, track_id in enumerate(self.track_ids)
            }
            new_track_ids = sorted(
                set(point_ids).difference(track_slot_by_id)
            )

            self.next_id = int(next_id)
        else:
            track_slot_by_id = {
                int(track_id): track_slot
                for track_slot, track_id in enumerate(self.track_ids)
            }
            new_track_ids = []

        if new_track_ids:
            first_new_track_slot = len(self.history)
            new_track_count = len(new_track_ids)
            new_history = np.zeros(
                (new_track_count, self.history_length, 3),
                dtype=np.float32,
            )
            new_track_ids = np.asarray(new_track_ids, dtype=np.int64)
            self.history = np.concatenate([self.history, new_history], axis=0)
            self.track_ids = np.concatenate([self.track_ids, new_track_ids])

            for offset, track_id in enumerate(new_track_ids):
                track_slot = first_new_track_slot + offset
                track_slot_by_id[int(track_id)] = track_slot

        assigned_track_slots = [
            track_slot_by_id[track_id]
            for track_id in point_ids
        ]

        previous_positions = self.history[:, -1, :2].copy()
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1, :2] = previous_positions
        self.history[:, -1, 2] = 0.0

        for point_index, track_slot in enumerate(assigned_track_slots):
            self.history[track_slot, -1, :2] = detection_features[point_index, :2]
            self.history[track_slot, -1, 2] = 1.0

        return point_ids

def draw_ring_results(frame, detections, point_ids):
    point_id_index = 0
    id_count = 0
    visible_points = {}

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            BOX_COLOR,
            thickness=BOX_THICKNESS,
            lineType=cv2.LINE_AA,
        )

        point = detection["point"]

        if point is None:
            # YOLO没有输出合格圆心：只显示绿框，不伪造红点
            continue

        # 红点始终使用YOLO输出的圆心坐标
        center_x = int(round(point[0]))
        center_y = int(round(point[1]))

        track_id = point_ids[point_id_index]
        point_id_index += 1
        visible_points[int(track_id)] = (center_x, center_y)

        # 每个 YOLO 合格圆心都已经由 Track Model 分配了 ID。
        cv2.circle(
            frame,
            (center_x, center_y),
            POINT_RADIUS,
            POINT_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        label = f"ID {track_id}"
        label_position = (center_x + 9, center_y - 9)

        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            ID_FONT_SCALE,
            ID_OUTLINE_COLOR,
            ID_THICKNESS + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            ID_FONT_SCALE,
            ID_COLOR,
            ID_THICKNESS,
            cv2.LINE_AA,
        )

        id_count += 1

    return len(detections), id_count, visible_points


def main():
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到YOLO模型：{YOLO_MODEL_PATH}")
    if not TRACK_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到坐标跟踪模型：{TRACK_MODEL_PATH}")
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"找不到输入视频：{INPUT_VIDEO}")
    if INPUT_VIDEO.resolve() == OUTPUT_VIDEO.resolve():
        raise ValueError("输出路径不能与输入视频相同")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{INPUT_VIDEO}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    print("第一遍：运行YOLO并收集完整视频的候选圆心")
    all_frame_detections = []
    scan_frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = yolo_model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            classes=[0],
            device=device,
            verbose=False,
        )[0]
        all_frame_detections.append(extract_ring_detections(result))
        scan_frame_index += 1
        if scan_frame_index % 30 == 0 or scan_frame_index == total_frames:
            print(f"YOLO进度：{scan_frame_index}/{total_frames}")
    capture.release()

    normalized_features, detection_masks = pack_video_detection_features(
        all_frame_detections,
        width,
        height,
    )

    tracker = NeuralPointTracker(
        TRACK_MODEL_PATH,
        device=device,
    )

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法重新打开输入视频：{INPUT_VIDEO}")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{OUTPUT_VIDEO}")

    print("第二遍：使用前后时序上下文进行ID关联")
    print(f"上下文：前{CONTEXT_RADIUS}帧 + 当前帧 + 后{CONTEXT_RADIUS}帧")
    print(f"训练数据最大轨迹数：{tracker.training_max_tracks}，推理容量：动态")

    frame_index = 0
    total_boxes = 0
    total_points = 0
    frame_records = []
    start_time = time.perf_counter()

    try:
        while frame_index < len(all_frame_detections):
            ok, frame = capture.read()
            if not ok:
                break

            current_mask = detection_masks[frame_index]
            current_features = normalized_features[frame_index, current_mask]
            current_context = build_temporal_context(
                normalized_features,
                detection_masks,
                frame_index,
                current_features,
            )
            point_ids = tracker.update(current_features, current_context)
            box_count, point_count, visible_points = draw_ring_results(
                frame,
                all_frame_detections[frame_index],
                point_ids,
            )

            timestamp_sec = frame_index / fps
            frame_records.append(
                (frame_index, timestamp_sec, visible_points)
            )

            total_boxes += box_count
            total_points += point_count
            writer.write(frame)
            frame_index += 1

            if frame_index % 30 == 0 or frame_index == len(all_frame_detections):
                print(f"关联进度：{frame_index}/{len(all_frame_detections)}")
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time
    processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
    write_point_csv(
        OUTPUT_CSV,
        frame_records,
        range(1, tracker.used_id_count + 1),
    )
    print(
        f"完成：共处理{frame_index}帧，绘制{total_boxes}个绿色框、"
        f"{total_points}个带ID红点"
    )
    print(f"共创建{tracker.used_id_count}个ID")
    print(f"第二遍平均处理速度：{processing_fps:.2f} FPS")
    print(f"结果视频：{OUTPUT_VIDEO}")
    print(f"坐标CSV：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()