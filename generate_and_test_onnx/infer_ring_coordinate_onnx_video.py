from pathlib import Path
import csv
import time

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
# 必须在创建 InferenceSession 之前执行
ort.preload_dlls(directory="")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RING_ONNX_PATH = PROJECT_ROOT / "model" / "onnx" / "ring_pose.onnx"
COORDINATE_ONNX_PATH = (
    PROJECT_ROOT / "model" / "onnx" / "coordinate_association.onnx"
)
INPUT_VIDEO = PROJECT_ROOT / "assets" / "input" / "25.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "assets" / "output" / "25.mp4"
OUTPUT_CSV = OUTPUT_VIDEO.with_name(f"{OUTPUT_VIDEO.stem}_points.csv")

DETECTION_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.5
NMS_IOU_THRESHOLD = 0.70
BOX_OVERLAP_THRESHOLD = 0.70
POINT_MIN_DISTANCE = 7.0
CONTEXT_RADIUS = 10


def providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def write_point_csv(csv_path, frame_records, point_ids):
    fieldnames = ["frame_index", "timestamp_sec"]
    for point_id in point_ids:
        fieldnames.extend([f"id_{point_id}_x_px", f"id_{point_id}_y_px"])

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


def letterbox(frame, image_size):
    height, width = frame.shape[:2]
    scale = min(image_size / height, image_size / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height))

    pad_width = image_size - new_width
    pad_height = image_size - new_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
    return np.ascontiguousarray(tensor), scale, left, top


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def nms(boxes, scores):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current], boxes[remaining]) <= NMS_IOU_THRESHOLD]
    return keep


def box_overlap_ratio(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    smaller_area = min(area_a, area_b)
    return 0.0 if smaller_area <= 0 else intersection / smaller_area


def decode_ring_output(output, scale, pad_x, pad_y, frame_width, frame_height):
    prediction = np.asarray(output)[0]
    if prediction.shape[1] not in (8, 9) and prediction.shape[0] in (8, 9):
        prediction = prediction.T

    channels = prediction.shape[1]
    if channels == 9:
        # YOLO26 end-to-end: xyxy, confidence, class_id, keypoint(x,y,conf)
        scores = prediction[:, 4]
        class_ids = np.rint(prediction[:, 5]).astype(np.int64)
        valid = (scores >= DETECTION_CONFIDENCE) & (class_ids == 0)
        prediction = prediction[valid]
        scores = scores[valid]
        boxes = prediction[:, :4].astype(np.float32)
        keypoints = prediction[:, 6:9]
        kept_indices = range(len(prediction))
    elif channels == 8:
        # 传统nms=False输出: xywh, class_confidence, keypoint(x,y,conf)
        valid = prediction[:, 4] >= DETECTION_CONFIDENCE
        prediction = prediction[valid]
        scores = prediction[:, 4]
        center_x, center_y, box_width, box_height = prediction[:, :4].T
        boxes = np.stack(
            [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ],
            axis=1,
        ).astype(np.float32)
        keypoints = prediction[:, 5:8]
        kept_indices = nms(boxes, scores) if len(boxes) else []
    else:
        raise RuntimeError(f"无法识别圆环ONNX输出形状：{np.asarray(output).shape}")

    detections = []
    for index in kept_indices:
        box = boxes[index].copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_height - 1)

        point_x = float((keypoints[index, 0] - pad_x) / scale)
        point_y = float((keypoints[index, 1] - pad_y) / scale)
        point_confidence = float(keypoints[index, 2])
        point = None
        if (
            point_confidence >= KEYPOINT_CONFIDENCE
            and 0 < point_x < frame_width
            and 0 < point_y < frame_height
        ):
            point = np.array([point_x, point_y], dtype=np.float32)

        detections.append(
            {
                "box": box,
                "box_confidence": float(scores[index]),
                "point": point,
                "point_confidence": point_confidence,
            }
        )

    detections.sort(key=lambda item: item["box_confidence"], reverse=True)
    filtered = []
    for detection in detections:
        if not any(
            box_overlap_ratio(detection["box"], kept["box"])
            > BOX_OVERLAP_THRESHOLD
            for kept in filtered
        ):
            filtered.append(detection)

    point_order = sorted(
        range(len(filtered)),
        key=lambda index: (
            filtered[index]["point_confidence"],
            filtered[index]["box_confidence"],
        ),
        reverse=True,
    )
    kept_points = []
    for index in point_order:
        point = filtered[index]["point"]
        if point is None:
            continue
        if any(np.linalg.norm(point - kept) < POINT_MIN_DISTANCE for kept in kept_points):
            filtered[index]["point"] = None
        else:
            kept_points.append(point)

    filtered.sort(
        key=lambda item: (
            (item["box"][1] + item["box"][3]) / 2,
            (item["box"][0] + item["box"][2]) / 2,
        )
    )
    return filtered


def detection_feature(detection):
    x1, y1, x2, y2 = detection["box"]
    x, y = detection["point"]
    return np.array(
        [
            x,
            y,
            detection["box_confidence"],
            detection["point_confidence"],
            x2 - x1,
            y2 - y1,
        ],
        dtype=np.float32,
    )


def pack_detection_features(all_detections, width, height):
    point_detections = [
        [item for item in frame if item["point"] is not None]
        for frame in all_detections
    ]
    max_detections = max(
        1,
        max((len(frame) for frame in point_detections), default=0),
    )
    features = np.zeros((len(all_detections), max_detections, 6), np.float32)
    masks = np.zeros((len(all_detections), max_detections), bool)
    for frame_index, valid in enumerate(point_detections):
        for detection_index, detection in enumerate(valid):
            features[frame_index, detection_index] = detection_feature(detection)
            masks[frame_index, detection_index] = True

    features[..., 0] /= width
    features[..., 1] /= height
    features[..., 4] /= width
    features[..., 5] /= height
    return features, masks


def build_context(all_features, all_masks, frame_index, current_features):
    context = np.zeros((len(current_features), 21, 8), np.float32)
    for detection_index, current in enumerate(current_features):
        anchor = current[:2]
        for window_index, offset in enumerate(range(-CONTEXT_RADIUS, CONTEXT_RADIUS + 1)):
            neighbor_frame = frame_index + offset
            if neighbor_frame < 0 or neighbor_frame >= len(all_features):
                continue
            neighbors = all_features[neighbor_frame][all_masks[neighbor_frame]]
            if len(neighbors) == 0:
                continue
            deltas = neighbors[:, :2] - anchor
            distances = np.linalg.norm(deltas, axis=1)
            nearest_index = int(np.argmin(distances))
            nearest = neighbors[nearest_index]
            context[detection_index, window_index] = [
                deltas[nearest_index, 0],
                deltas[nearest_index, 1],
                distances[nearest_index],
                nearest[2],
                nearest[3],
                nearest[4],
                nearest[5],
                1.0,
            ]
    return context


def assign_ids(logits, track_mask, detection_count, next_id):
    max_tracks = len(track_mask)
    if detection_count == 0:
        return [], next_id

    scores = np.full(
        (detection_count, max_tracks + detection_count),
        -1e12,
        dtype=np.float64,
    )
    scores[:, :max_tracks] = logits[:detection_count, :max_tracks]
    scores[:, :max_tracks][:, ~track_mask] = -1e12
    new_score = logits[:detection_count, max_tracks]
    new_columns = max_tracks + np.arange(detection_count)
    scores[np.arange(detection_count), new_columns] = new_score

    rows, columns = linear_sum_assignment(-scores)
    selected = np.full(detection_count, -1, np.int64)
    selected[rows] = columns
    used_ids = {index + 1 for index in np.flatnonzero(track_mask)}
    point_ids = []
    for column in selected:
        if column < max_tracks:
            point_ids.append(int(column + 1))
        else:
            while next_id in used_ids:
                next_id += 1
            point_ids.append(next_id)
            used_ids.add(next_id)
            next_id += 1
    return point_ids, next_id


class CoordinateTracker:
    def __init__(self, session):
        self.session = session
        inputs = {node.name: node for node in session.get_inputs()}
        history_shape = inputs["history"].shape
        detection_shape = inputs["detections"].shape
        if isinstance(history_shape[1], int) or isinstance(
            detection_shape[1], int
        ):
            raise RuntimeError(
                "当前coordinate_association.onnx仍是固定容量模型："
                f"history={history_shape}, detections={detection_shape}。"
                "请先用更新后的export_models_to_onnx.py重新导出。"
            )
        self.history_length = int(history_shape[2])
        context_shape = inputs["detection_context"].shape
        self.context_length = int(context_shape[2])
        self.history = np.zeros(
            (0, self.history_length, 3), np.float32
        )
        self.track_ids = np.zeros(0, dtype=np.int64)
        self.next_id = 1

    @property
    def used_id_count(self):
        return self.next_id - 1

    def update(self, detections, context):
        detections = np.asarray(detections, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)
        detection_count = len(detections)
        track_count = len(self.history)
        if context.shape != (detection_count, self.context_length, 8):
            raise ValueError(
                "detection_context形状错误："
                f"期望{(detection_count, self.context_length, 8)}，"
                f"实际{context.shape}"
            )

        # 没有检测时无需调用ONNX，但所有轨迹仍向前推进一个不可见时间步。
        if detection_count == 0:
            self._advance_history(detections, [])
            return []

        detection_mask = np.ones(detection_count, dtype=bool)
        if track_count == 0:
            # 旧ONNX tracer只记录了track_count>0分支。首帧用一个被mask掉的
            # 零轨迹占位，推理后删除该占位轨迹对应的logit列。
            onnx_history = np.zeros(
                (1, self.history_length, 3), dtype=np.float32
            )
            onnx_track_mask = np.zeros(1, dtype=bool)
        else:
            onnx_history = self.history
            onnx_track_mask = np.ones(track_count, dtype=bool)

        logits = self.session.run(
            ["logits"],
            {
                "history": onnx_history[None],
                "track_mask": onnx_track_mask[None],
                "detections": detections[None],
                "detection_context": context[None],
                "detection_mask": detection_mask[None],
            },
        )[0][0]
        if track_count == 0:
            # [dummy_track, NEW] -> [NEW]
            logits = logits[:, -1:]
            assignment_track_mask = np.zeros(0, dtype=bool)
        else:
            assignment_track_mask = np.ones(track_count, dtype=bool)

        point_ids, self.next_id = assign_ids(
            logits,
            assignment_track_mask,
            detection_count,
            self.next_id,
        )

        track_slot_by_id = {
            int(track_id): slot
            for slot, track_id in enumerate(self.track_ids)
        }
        new_track_ids = sorted(set(point_ids).difference(track_slot_by_id))
        if new_track_ids:
            first_new_slot = len(self.history)
            new_history = np.zeros(
                (len(new_track_ids), self.history_length, 3),
                dtype=np.float32,
            )
            new_track_ids_array = np.asarray(new_track_ids, dtype=np.int64)
            self.history = np.concatenate(
                [self.history, new_history], axis=0
            )
            self.track_ids = np.concatenate(
                [self.track_ids, new_track_ids_array]
            )
            for offset, track_id in enumerate(new_track_ids):
                track_slot_by_id[int(track_id)] = first_new_slot + offset

        assigned_slots = [track_slot_by_id[point_id] for point_id in point_ids]
        self._advance_history(detections, assigned_slots)
        return point_ids

    def _advance_history(self, detections, assigned_slots):
        """推进一帧历史，并把当前可见检测写回对应的动态轨迹槽。"""

        previous_positions = self.history[:, -1, :2].copy()
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1, :2] = previous_positions
        self.history[:, -1, 2] = 0
        for detection_index, slot in enumerate(assigned_slots):
            self.history[slot, -1, :2] = detections[detection_index, :2]
            self.history[slot, -1, 2] = 1


def draw_results(frame, detections, point_ids):
    point_index = 0
    visible_points = {}
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if detection["point"] is None:
            continue

        x, y = np.rint(detection["point"]).astype(int)
        point_id = point_ids[point_index]
        point_index += 1
        visible_points[point_id] = (int(x), int(y))
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"ID {point_id}"
        position = (x + 9, y - 9)
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return visible_points


def main():
    for path in (RING_ONNX_PATH, COORDINATE_ONNX_PATH, INPUT_VIDEO):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    execution_providers = providers()
    ring_session = ort.InferenceSession(
        str(RING_ONNX_PATH), providers=execution_providers
    )
    coordinate_session = ort.InferenceSession(
        str(COORDINATE_ONNX_PATH), providers=execution_providers
    )
    ring_input_name = ring_session.get_inputs()[0].name
    ring_image_size = int(ring_session.get_inputs()[0].shape[2])
    ring_output_name = ring_session.get_outputs()[0].name
    tracker = CoordinateTracker(coordinate_session)

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频：{INPUT_VIDEO}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 25.0

    print("第一遍：ring_pose.onnx检测圆环")
    all_detections = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tensor, scale, pad_x, pad_y = letterbox(frame, ring_image_size)
        output = ring_session.run(
            [ring_output_name], {ring_input_name: tensor}
        )[0]
        all_detections.append(
            decode_ring_output(
                output, scale, pad_x, pad_y, width, height
            )
        )
        if len(all_detections) % 30 == 0:
            print(f"检测进度：{len(all_detections)}/{total_frames}")
    capture.release()

    features, masks = pack_detection_features(
        all_detections, width, height
    )
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建视频：{OUTPUT_VIDEO}")

    print("第二遍：coordinate_association.onnx动态分配ID")
    video_max_detections = (
        int(masks.sum(axis=1).max()) if len(masks) else 0
    )
    print(
        "推理容量：动态；本视频单帧最多"
        f"{video_max_detections}个有效圆心"
    )
    frame_records = []
    start_time = time.perf_counter()
    for frame_index, detections in enumerate(all_detections):
        ok, frame = capture.read()
        if not ok:
            break
        current_features = features[frame_index, masks[frame_index]]
        context = build_context(features, masks, frame_index, current_features)
        point_ids = tracker.update(current_features, context)
        visible_points = draw_results(frame, detections, point_ids)
        frame_records.append((frame_index, frame_index / fps, visible_points))
        writer.write(frame)
        if (frame_index + 1) % 30 == 0:
            print(f"关联进度：{frame_index + 1}/{len(all_detections)}")

    capture.release()
    writer.release()
    write_point_csv(
        OUTPUT_CSV,
        frame_records,
        range(1, tracker.used_id_count + 1),
    )
    elapsed = time.perf_counter() - start_time
    print(f"完成，平均速度：{len(frame_records) / elapsed:.2f} FPS")
    print("结果视频：", OUTPUT_VIDEO)
    print("坐标CSV：", OUTPUT_CSV)


if __name__ == "__main__":
    main()