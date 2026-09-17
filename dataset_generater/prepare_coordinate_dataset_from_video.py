from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.widgets import Button
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_MODEL_PATH = PROJECT_ROOT / "model/ring_pose/yolo26n_pose_ring/weights/best_ring_class0.pt"
INPUT_VIDEO = PROJECT_ROOT / "assets/input/20.mp4"
OUTPUT_DIR = PROJECT_ROOT / "coordinate_dataset/scenes"

IMAGE_SIZE = 640
# YOLO 推理时使用的输入尺寸
DETECTION_CONFIDENCE = 0.05
KEYPOINT_CONFIDENCE = 0.3
BOX_OVERLAP_THRESHOLD = 0.70
POINT_MIN_DISTANCE = 7.0
CLICK_MAX_DISTANCE = 50.0
DEVICE = None
WINDOW_NAME = "Video coordinate dataset annotator"


def find_chinese_font():
    common_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-VF.otf.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
    ]
    for font_path in common_paths:
        if Path(font_path).exists():
            return font_manager.FontProperties(fname=font_path)

    keywords = (
        "notosanscjk", "notoserifcjk", "sourcehansans", "sourcehanserif",
        "wenquanyi", "wqy", "droidsansfallback", "simhei", "msyh",
    )
    for font_path in font_manager.findSystemFonts():
        name = Path(font_path).name.lower().replace("-", "").replace("_", "")
        if any(keyword in name for keyword in keywords):
            return font_manager.FontProperties(fname=font_path)
    return None


CHINESE_FONT = find_chinese_font()
plt.rcParams["axes.unicode_minus"] = False

# 这些按键由标注器处理，避免 Matplotlib 工具栏同时响应。
RESERVED_KEYS = {"s", "f", "q", "c", "left", "right", "backspace", " "}
for keymap_name in [name for name in plt.rcParams if name.startswith("keymap.")]:
    plt.rcParams[keymap_name] = [
        key for key in plt.rcParams[keymap_name]
        if str(key).lower() not in RESERVED_KEYS
    ]


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
            point_confidence = (
                float(point_confidences[index][0])
                if point_confidences is not None else 1.0
            )
            if center_x > 0 and center_y > 0 and point_confidence >= KEYPOINT_CONFIDENCE:
                point = np.array([center_x, center_y], dtype=np.float32)
        detections.append({
            "box": box.astype(np.float32),
            "box_confidence": float(box_confidences[index]),
            "point": point,
            "point_confidence": point_confidence,
        })

    detections.sort(key=lambda item: item["box_confidence"], reverse=True)
    filtered = []
    for detection in detections:
        if not any(
            box_overlap_ratio(detection["box"], kept["box"]) > BOX_OVERLAP_THRESHOLD
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

    filtered = [item for item in filtered if item["point"] is not None]
    filtered.sort(key=lambda item: (item["point"][1], item["point"][0]))
    return filtered


def detection_feature(detection):
    x1, y1, x2, y2 = detection["box"]
    point_x, point_y = detection["point"]
    return np.array([
        point_x, point_y,
        detection["box_confidence"], detection["point_confidence"],
        x2 - x1, y2 - y1,
    ], dtype=np.float32)


def detections_from_record(record):
    return [
        {
            "point": feature[:2].copy(),
            "box_confidence": float(feature[2]),
            "point_confidence": float(feature[3]),
            "box": box.copy(),
        }
        for feature, box in zip(record["features"], record["boxes"])
    ]


class VideoAnnotationApp:
    def __init__(self):
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"找不到 YOLO 模型：{YOLO_MODEL_PATH}")
        if not INPUT_VIDEO.exists():
            raise FileNotFoundError(f"找不到输入视频：{INPUT_VIDEO}")

        self.video_size_bytes = INPUT_VIDEO.stat().st_size
        self.cache_path = OUTPUT_DIR / f"{INPUT_VIDEO.name}.npz"
        self.model = YOLO(str(YOLO_MODEL_PATH))
        self.device = choose_device()
        self.capture = cv2.VideoCapture(str(INPUT_VIDEO))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开输入视频：{INPUT_VIDEO}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps <= 0:
            self.fps = 25.0
        if self.total_frames <= 0:
            raise RuntimeError("视频总帧数无效")

        self.records = [None] * self.total_frames
        self.frame_index = 0
        self.frame = None
        self.detections = []
        self.assignments = {}
        self.skipped_ids = set()
        self.actions = []
        self.current_id = 1
        self.status_message = ""
        self.closed = False

        self.figure = None
        self.image_axes = None
        self.image_artist = None
        self.info_artist = None
        self.help_artist = None
        self.overlay_artists = []
        self.buttons = []

        if self.cache_path.exists():
            self.load_dataset()
        else:
            self.save_dataset()
        self.load_frame(self.frame_index)

    def set_status(self, message):
        self.status_message = message
        print(message)

    def validate_cache(self, data):
        required = {
            "positions", "visible", "detection_features", "detection_mask",
            "detection_ids", "width", "height", "source_video_name",
            "source_video_size_bytes", "frame_count",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(
                f"缓存缺少字段 {missing}，请移走旧文件：{self.cache_path}"
            )

        cached_identity = (
            str(data["source_video_name"].item()),
            int(data["source_video_size_bytes"]),
            int(data["frame_count"]),
            int(data["width"]),
            int(data["height"]),
        )
        current_identity = (
            INPUT_VIDEO.name, self.video_size_bytes, self.total_frames,
            self.width, self.height,
        )
        if cached_identity != current_identity:
            raise RuntimeError(
                "NPZ 与当前视频不匹配，请勿混用：\n"
                f"缓存={cached_identity}\n当前={current_identity}"
            )

    def load_dataset(self):
        with np.load(self.cache_path, allow_pickle=False) as data:
            self.validate_cache(data)
            features = data["detection_features"].astype(np.float32)
            masks = data["detection_mask"].astype(bool)
            ids = data["detection_ids"].astype(np.int64)
            if features.shape[0] != self.total_frames:
                raise RuntimeError("NPZ 第一维与视频总帧数不一致")

            if "detection_boxes" in data.files:
                boxes = data["detection_boxes"].astype(np.float32)
            else:
                boxes = np.zeros((*features.shape[:2], 4), dtype=np.float32)
                boxes[..., 0] = features[..., 0] - features[..., 4] / 2
                boxes[..., 1] = features[..., 1] - features[..., 5] / 2
                boxes[..., 2] = features[..., 0] + features[..., 4] / 2
                boxes[..., 3] = features[..., 1] + features[..., 5] / 2

            for frame_index in range(self.total_frames):
                slots = np.flatnonzero(masks[frame_index])
                if len(slots) > 0:
                    self.records[frame_index] = {
                        "features": features[frame_index, slots].copy(),
                        "boxes": boxes[frame_index, slots].copy(),
                        "ids": ids[frame_index, slots].copy(),
                    }
            if "last_frame_index" in data.files:
                self.frame_index = min(
                    max(int(data["last_frame_index"]), 0), self.total_frames - 1
                )
        self.set_status(f"已读取缓存：{self.cache_path}")

    def save_dataset(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        max_detections = max(1, max(
            (len(record["features"]) for record in self.records if record),
            default=0,
        ))
        max_track_id = max((
            int(record["ids"].max())
            for record in self.records
            if record is not None and len(record["ids"]) > 0
        ), default=0)

        detection_features = np.zeros(
            (self.total_frames, max_detections, 6), dtype=np.float32
        )
        detection_boxes = np.zeros(
            (self.total_frames, max_detections, 4), dtype=np.float32
        )
        detection_mask = np.zeros((self.total_frames, max_detections), dtype=bool)
        detection_ids = np.zeros((self.total_frames, max_detections), dtype=np.int64)
        positions = np.full(
            (self.total_frames, max_track_id, 2), np.nan, dtype=np.float32
        )
        visible = np.zeros((self.total_frames, max_track_id), dtype=bool)

        for frame_index, record in enumerate(self.records):
            if record is None:
                continue
            count = len(record["features"])
            detection_features[frame_index, :count] = record["features"]
            detection_boxes[frame_index, :count] = record["boxes"]
            detection_mask[frame_index, :count] = True
            detection_ids[frame_index, :count] = record["ids"]
            for detection_index, track_id in enumerate(record["ids"]):
                if track_id > 0:
                    positions[frame_index, track_id - 1] = record["features"][detection_index, :2]
                    visible[frame_index, track_id - 1] = True

        temporary_path = self.cache_path.with_name(self.cache_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary_path,
            positions=positions,
            visible=visible,
            detection_features=detection_features,
            detection_boxes=detection_boxes,
            detection_mask=detection_mask,
            detection_ids=detection_ids,
            fps=np.float32(self.fps),
            width=np.int32(self.width),
            height=np.int32(self.height),
            expected_track_count=np.int32(max_track_id),
            source_video=np.array(str(INPUT_VIDEO)),
            source_video_name=np.array(INPUT_VIDEO.name),
            source_video_size_bytes=np.int64(self.video_size_bytes),
            frame_count=np.int32(self.total_frames),
            last_frame_index=np.int32(self.frame_index),
        )
        temporary_path.replace(self.cache_path)
        print(
            f"已缓存：{self.cache_path} | 帧数={self.total_frames} | "
            f"当前最大 ID={max_track_id}"
        )

    def predict_frame(self, frame):
        result = self.model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        return extract_ring_detections(result)

    def load_frame(self, frame_index):
        frame_index = min(max(int(frame_index), 0), self.total_frames - 1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.set_status(f"无法读取第 {frame_index + 1} 帧")
            return

        self.frame_index = frame_index
        self.frame = frame
        record = self.records[frame_index]
        if record is None:
            self.detections = self.predict_frame(frame)
            self.assignments = {}
            self.current_id = 1
            source = "YOLO"
        else:
            self.detections = detections_from_record(record)
            self.assignments = {
                index: int(track_id)
                for index, track_id in enumerate(record["ids"])
                if track_id > 0
            }
            self.current_id = max(self.assignments.values(), default=0) + 1
            source = "NPZ 缓存"

        assigned_ids = set(self.assignments.values())
        self.skipped_ids = set(range(1, self.current_id)).difference(assigned_ids)
        self.actions = []
        self.set_status(
            f"第 {frame_index + 1}/{self.total_frames} 帧："
            f"从 {source} 读取 {len(self.detections)} 个候选圆心"
        )

    def assign_nearest_detection(self, x, y):
        click_point = np.array([x, y], dtype=np.float32)
        candidates = []
        for detection_index, detection in enumerate(self.detections):
            if detection_index not in self.assignments:
                distance = float(np.linalg.norm(detection["point"] - click_point))
                candidates.append((distance, detection_index))
        if not candidates:
            self.set_status("当前帧没有可点击的未使用圆心")
            return
        distance, detection_index = min(candidates)
        if distance > CLICK_MAX_DISTANCE:
            self.set_status(f"距离最近圆心 {distance:.1f} 像素，未执行标记")
            return

        track_id = self.current_id
        self.assignments[detection_index] = track_id
        self.actions.append(("assign", track_id, detection_index))
        self.current_id += 1
        self.set_status(f"候选 {detection_index} -> ID {track_id}")

    def skip_current_id(self):
        track_id = self.current_id
        self.skipped_ids.add(track_id)
        self.actions.append(("skip", track_id, None))
        self.current_id += 1
        self.set_status(f"ID {track_id} 漏检")

    def undo(self):
        if not self.actions:
            self.set_status("没有可撤销的新操作；旧标注请按 C 后重新标记")
            return
        action, track_id, detection_index = self.actions.pop()
        self.current_id = track_id
        if action == "assign":
            self.assignments.pop(detection_index, None)
        else:
            self.skipped_ids.discard(track_id)
        self.set_status(f"已撤销 ID {track_id}")

    def clear_current_frame(self):
        self.assignments = {}
        self.skipped_ids = set()
        self.actions = []
        self.current_id = 1
        self.set_status("当前帧已清屏；按空格后才会覆盖 NPZ")

    def commit_frame(self):
        unassigned_indices = [
            detection_index
            for detection_index in range(len(self.detections))
            if detection_index not in self.assignments
        ]
        if unassigned_indices:
            self.set_status(
                f"还有 {len(unassigned_indices)} 个候选圆心没有 ID，"
                "请全部分配后再提交"
            )
            return False

        saved_frame_index = self.frame_index
        features = np.array(
            [detection_feature(item) for item in self.detections], dtype=np.float32
        )
        boxes = np.array([item["box"] for item in self.detections], dtype=np.float32)
        if len(features) == 0:
            features = np.zeros((0, 6), dtype=np.float32)
            boxes = np.zeros((0, 4), dtype=np.float32)
        ids = np.zeros(len(self.detections), dtype=np.int64)
        for detection_index, track_id in self.assignments.items():
            ids[detection_index] = track_id
        self.records[saved_frame_index] = {
            "features": features, "boxes": boxes, "ids": ids,
        }

        next_frame_index = min(saved_frame_index + 1, self.total_frames - 1)
        self.frame_index = next_frame_index
        self.save_dataset()
        self.load_frame(next_frame_index)
        self.set_status(
            f"第 {saved_frame_index + 1} 帧已写入 NPZ；"
            f"当前为第 {next_frame_index + 1} 帧"
        )
        return True

    def browse(self, step):
        target = self.frame_index + step
        if 0 <= target < self.total_frames:
            self.load_frame(target)
        else:
            self.set_status("已经到达视频边界")

    def build_interface(self):
        self.figure = plt.figure(figsize=(11.5, 6.5))
        try:
            self.figure.canvas.manager.set_window_title(WINDOW_NAME)
        except AttributeError:
            pass

        self.image_axes = self.figure.add_axes((0.02, 0.16, 0.96, 0.71))
        self.image_axes.set_axis_off()
        self.info_artist = self.figure.text(
            0.02, 0.955, "", ha="left", va="top", fontsize=12,
            weight="bold", fontproperties=CHINESE_FONT,
        )
        self.help_artist = self.figure.text(
            0.02, 0.905, "", ha="left", va="top", fontsize=9,
            fontproperties=CHINESE_FONT,
        )

        button_specs = [
            ("上一帧 ←", lambda _: self.browse_and_refresh(-1)),
            ("下一帧 →", lambda _: self.browse_and_refresh(1)),
            ("漏检 S", self.on_skip_button),
            ("撤销", self.on_undo_button),
            ("清空 C", self.on_clear_button),
            ("提交 Space", self.on_commit_button),
            ("全屏 F", self.on_fullscreen_button),
            ("保存退出 Q", self.on_quit_button),
        ]
        button_width = 0.108
        gap = 0.008
        total_width = len(button_specs) * button_width + (len(button_specs) - 1) * gap
        start_x = (1.0 - total_width) / 2.0
        for index, (label, callback) in enumerate(button_specs):
            axes = self.figure.add_axes((
                start_x + index * (button_width + gap), 0.045,
                button_width, 0.055,
            ))
            button = Button(axes, label)
            if CHINESE_FONT is not None:
                button.label.set_fontproperties(CHINESE_FONT)
                button.label.set_fontsize(9)
            button.on_clicked(callback)
            self.buttons.append(button)

        self.figure.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.refresh_display()

    def on_mouse_press(self, event):
        if event.inaxes is not self.image_axes:
            return
        if event.button == 3:
            self.undo()
        elif event.button == 1 and event.xdata is not None and event.ydata is not None:
            if 0 <= event.xdata < self.width and 0 <= event.ydata < self.height:
                self.assign_nearest_detection(event.xdata, event.ydata)
        self.refresh_display()

    def on_key_press(self, event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            self.finish()
        elif key == "s":
            self.skip_current_id()
            self.refresh_display()
        elif key in ("backspace", "ctrl+z"):
            self.undo()
            self.refresh_display()
        elif key == "c":
            self.clear_current_frame()
            self.refresh_display()
        elif key in (" ", "space"):
            self.commit_frame()
            self.refresh_display()
        elif key == "left":
            self.browse_and_refresh(-1)
        elif key == "right":
            self.browse_and_refresh(1)
        elif key == "f":
            self.toggle_fullscreen()

    def browse_and_refresh(self, step):
        self.browse(step)
        self.refresh_display()

    def on_skip_button(self, _):
        self.skip_current_id()
        self.refresh_display()

    def on_undo_button(self, _):
        self.undo()
        self.refresh_display()

    def on_clear_button(self, _):
        self.clear_current_frame()
        self.refresh_display()

    def on_commit_button(self, _):
        self.commit_frame()
        self.refresh_display()

    def on_fullscreen_button(self, _):
        self.toggle_fullscreen()

    def on_quit_button(self, _):
        self.finish()

    def on_close(self, _):
        if not self.closed:
            self.closed = True
            self.save_dataset()
            self.capture.release()

    def toggle_fullscreen(self):
        manager = self.figure.canvas.manager
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()

    def refresh_display(self):
        if self.figure is None or self.frame is None:
            return

        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.image_artist is None:
            self.image_artist = self.image_axes.imshow(
                rgb_frame, interpolation="nearest", origin="upper",
                extent=(-0.5, self.width - 0.5, self.height - 0.5, -0.5),
            )
            self.image_axes.set_xlim(-0.5, self.width - 0.5)
            self.image_axes.set_ylim(self.height - 0.5, -0.5)
            self.image_axes.set_aspect("equal", adjustable="box")
        else:
            self.image_artist.set_data(rgb_frame)

        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists = []
        for detection_index, detection in enumerate(self.detections):
            point_x, point_y = detection["point"]
            x1, y1, x2, y2 = detection["box"]
            if detection_index in self.assignments:
                color = "#00e676"
                label = f"ID {self.assignments[detection_index]}"
            else:
                color = "#ffb300"
                label = f"C{detection_index}"

            rectangle = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2,
            )
            self.image_axes.add_patch(rectangle)
            point_artist = self.image_axes.scatter(
                [point_x], [point_y], s=55, c=color,
                edgecolors="black", linewidths=0.7, zorder=3,
            )
            label_artist = self.image_axes.text(
                point_x + 8, point_y - 8, label, color=color,
                fontsize=10, weight="bold",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"},
                zorder=4,
            )
            self.overlay_artists.extend([rectangle, point_artist, label_artist])

        marked_ids = " ".join(str(track_id) for track_id in sorted(self.assignments.values()))
        skipped_ids = " ".join(str(track_id) for track_id in sorted(self.skipped_ids))
        self.info_artist.set_text(
            f"第 {self.frame_index + 1}/{self.total_frames} 帧  |  "
            f"当前 ID：{self.current_id}  |  已标记：{marked_ids or '-'}  |  "
            f"漏检：{skipped_ids or '-'}"
        )
        self.help_artist.set_text(
            f"{self.status_message}\n"
            "左键：分配 ID | S：漏检并增加 ID | C：清屏 | Space：覆盖当前帧并缓存 | "
            "←/→：浏览（未提交修改会丢弃） | F：全屏 | Q/Esc：保存退出 | "
            "所有候选必须分配 ID"
        )
        self.figure.canvas.draw_idle()

    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.save_dataset()
        self.capture.release()
        if self.figure is not None:
            figure = self.figure
            self.figure = None
            plt.close(figure)

    def run(self):
        print(f"视频：{INPUT_VIDEO}")
        print(f"缓存：{self.cache_path}")
        print("ID 数量不设上限；按空格覆盖当前帧并进入下一帧")
        self.build_interface()
        plt.show()


def main():
    VideoAnnotationApp().run()


if __name__ == "__main__":
    main()