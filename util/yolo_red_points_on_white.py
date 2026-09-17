from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch
from ultralytics import YOLO

 
PROJECT_ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "ring_pose"
    / "yolo26n_pose_ring"
    / "weights"
    / "best_ring_class0.pt"
)
INPUT_VIDEO = PROJECT_ROOT / "assets" / "input" / "3.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "assets" / "output" / "3.mp4"

IMAGE_SIZE = 640
DETECTION_CONFIDENCE = 0.15
KEYPOINT_CONFIDENCE = 0.5
BOX_OVERLAP_THRESHOLD = 0.70
POINT_MIN_DISTANCE = 7.0
POINT_RADIUS = 6
POINT_COLOR = (0, 0, 255)  # OpenCV使用BGR，因此这是红色

CRF = 18
PRESET = "medium"
DEVICE = None


def choose_device():
    if DEVICE is not None:
        return DEVICE
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def box_overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)
    return 0.0 if smaller_area <= 0 else intersection_area / smaller_area


def extract_processed_points(result):
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
            point_x, point_y = points[index][0]
            point_confidence = (
                float(point_confidences[index][0])
                if point_confidences is not None
                else 1.0
            )
            if (
                point_x > 0
                and point_y > 0
                and point_confidence >= KEYPOINT_CONFIDENCE
            ):
                point = np.array([point_x, point_y], dtype=np.float32)

        detections.append(
            {
                "box": box.astype(np.float32),
                "box_confidence": float(box_confidences[index]),
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
        if any(
            np.linalg.norm(point - kept_point) < POINT_MIN_DISTANCE
            for kept_point in kept_points
        ):
            filtered[index]["point"] = None
        else:
            kept_points.append(point)

    return [
        (int(round(item["point"][0])), int(round(item["point"][1])))
        for item in filtered
        if item["point"] is not None
    ]


def start_video_encoder(output_path, width, height, fps):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("没有找到ffmpeg，请先安装：sudo apt install ffmpeg")

    output_width = width + width % 2
    output_height = height + height % 2

    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.10g}",
        "-i",
        "-",
        "-an",
        "-vf",
        f"pad={output_width}:{output_height}:0:0:white",
        "-c:v",
        "libx264",
        "-crf",
        str(CRF),
        "-preset",
        PRESET,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main():
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到YOLO模型：{YOLO_MODEL_PATH}\n"
            "请修改代码顶部的YOLO_MODEL_PATH。"
        )

    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"找不到输入视频：{INPUT_VIDEO}")
    if INPUT_VIDEO.resolve() == OUTPUT_VIDEO.resolve():
        raise ValueError("输入和输出视频不能是同一个文件")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{INPUT_VIDEO}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("无法读取视频分辨率")
    if fps <= 0:
        capture.release()
        raise RuntimeError("无法读取视频帧率")

    duration = expected_frames / fps if expected_frames > 0 else 0.0
    print(f"宽度：{width} 像素")
    print(f"高度：{height} 像素")
    print(f"分辨率：{width} x {height}")
    print(f"帧率：{fps:.6g} FPS")
    print(f"预计帧数：{expected_frames}")
    print(f"预计时长：{duration:.3f} 秒")

    device = choose_device()
    model = YOLO(str(YOLO_MODEL_PATH))
    encoder = start_video_encoder(OUTPUT_VIDEO, width, height, fps)

    frame_index = 0
    total_points = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=DETECTION_CONFIDENCE,
                classes=[0],
                device=device,
                verbose=False,
            )[0]

            white_frame = np.full((height, width, 3), 255, dtype=np.uint8)
            points = extract_processed_points(result)
            for point in points:
                cv2.circle(
                    white_frame,
                    point,
                    POINT_RADIUS,
                    POINT_COLOR,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            if encoder.stdin is None:
                raise RuntimeError("ffmpeg输入管道创建失败")
            encoder.stdin.write(white_frame.tobytes())

            frame_index += 1
            total_points += len(points)
            if frame_index % 30 == 0 or frame_index == expected_frames:
                print(f"处理进度：{frame_index}/{expected_frames}")
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()

    return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg编码失败，退出代码：{return_code}")

    actual_duration = frame_index / fps
    print(f"完成：处理 {frame_index} 帧，绘制 {total_points} 个红点")
    print(f"输出时长：{actual_duration:.3f} 秒")
    print(f"结果视频：{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()