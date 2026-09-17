from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "assets" / "input"
DATASET_DIR = PROJECT_ROOT / "ring_dataset"

VIDEO_SPLITS = {
    "1.mp4": "train",
    "2.mp4": "train",
    "3.mp4": "train",
    "6.mp4": "train",
    "7.mp4": "train",
    "8.mp4": "train",
    "10.mp4": "train",
    "11.mp4": "train",
    "12.mp4": "train", 
    "14.mp4": "train",
    "4.mp4": "val",
    "5.mp4": "test",
}

SAMPLE_FPS = 2.0
MAX_FRAMES_PER_VIDEO = 150
JPEG_QUALITY = 95
MIN_BOX_SIZE = 6

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


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
        "notosanscjk",
        "notoserifcjk",
        "sourcehansans",
        "sourcehanserif",
        "wenquanyi",
        "wqy",
        "droidsansfallback",
        "simhei",
        "msyh",
    )
    for font_path in font_manager.findSystemFonts():
        normalized_name = Path(font_path).name.lower().replace("-", "").replace("_", "")
        if any(keyword in normalized_name for keyword in keywords):
            return font_manager.FontProperties(fname=font_path)

    return None


CHINESE_FONT = find_chinese_font()


def ensure_dataset_layout():
    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    config = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [1, 3],
        "flip_idx": [0],
        "names": {0: "ring", 1: "distractor"},
        "kpt_names": {0: ["center"]},
    }
    with (DATASET_DIR / "ring_pose.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def sample_frame_indices(total_frames, fps):
    duration = total_frames / fps
    wanted = max(1, int(round(duration * SAMPLE_FPS)))
    wanted = min(wanted, MAX_FRAMES_PER_VIDEO, total_frames)
    return np.unique(np.linspace(0, total_frames - 1, wanted, dtype=np.int64))


def extract_frames():
    print("开始检查并抽取视频帧……")
    for video_name, split in VIDEO_SPLITS.items():
        video_path = INPUT_DIR / video_name
        if not video_path.exists():
            print(f"跳过不存在的视频：{video_path}")
            continue

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"无法打开视频：{video_path}")
            continue

        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            capture.release()
            print(f"无法读取视频参数：{video_path}")
            continue

        output_dir = DATASET_DIR / "images" / split
        indices = sample_frame_indices(total_frames, fps)
        created = 0

        for frame_index in indices:
            output_path = output_dir / f"{video_path.stem}_f{int(frame_index):08d}.jpg"
            if output_path.exists():
                continue

            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                print(f"读取失败：{video_name} 第 {frame_index} 帧")
                continue

            if cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            ):
                created += 1

        capture.release()
        print(
            f"{video_name} -> {split}: 计划 {len(indices)} 帧，"
            f"本次新增 {created} 帧"
        )


def collect_images():
    items = []
    for split in ("train", "val", "test"):
        image_dir = DATASET_DIR / "images" / split
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append((split, path))
    return items


class RingPoseAnnotator:
    def __init__(self, items):
        self.items = items
        self.index = self.first_unlabeled_index()
        self.image_bgr = None
        self.image_rgb = None
        self.height = 0
        self.width = 0
        self.annotations = []
        self.class_id = 0
        self.pending_box = None
        self.drag_start = None
        self.drag_patch = None

        self.figure, self.axis = plt.subplots(figsize=(10, 8))
        self.figure.canvas.manager.set_window_title("圆环与杂物标注")
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)

        self.load_current()

    def first_unlabeled_index(self):
        for index, (split, image_path) in enumerate(self.items):
            if not self.label_path(split, image_path).exists():
                return index
        return 0

    @staticmethod
    def label_path(split, image_path):
        return DATASET_DIR / "labels" / split / f"{image_path.stem}.txt"

    def load_labels(self, label_path):
        annotations = []
        if not label_path.exists():
            return annotations

        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 8:
                continue
            class_id_value, cx, cy, bw, bh, px, py, visibility = map(float, values)
            if class_id_value not in (0.0, 1.0):
                continue
            class_id = int(class_id_value)
            x1 = (cx - bw / 2) * self.width
            y1 = (cy - bh / 2) * self.height
            x2 = (cx + bw / 2) * self.width
            y2 = (cy + bh / 2) * self.height
            annotations.append(
                {
                    "class_id": class_id,
                    "box": (x1, y1, x2, y2),
                    "center": (
                        (px * self.width, py * self.height)
                        if class_id == 0 and int(visibility) > 0
                        else None
                    ),
                    "visibility": int(visibility),
                }
            )
        return annotations

    def load_current(self):
        split, image_path = self.items[self.index]
        self.image_bgr = cv2.imread(str(image_path))
        if self.image_bgr is None:
            raise RuntimeError(f"无法读取图片：{image_path}")

        self.height, self.width = self.image_bgr.shape[:2]
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.annotations = self.load_labels(self.label_path(split, image_path))
        self.pending_box = None
        self.drag_start = None
        self.drag_patch = None
        self.redraw()

    def completed_count(self):
        count = 0
        for split, image_path in self.items:
            if self.label_path(split, image_path).exists():
                count += 1
        return count

    def update_title(self, message=""):
        split, image_path = self.items[self.index]
        if CHINESE_FONT is not None:
            mode = "圆环 class 0" if self.class_id == 0 else "杂物 class 1"
            instruction = (
                f"当前模式：{mode} | 0圆环 | 1杂物 | 左键拖框 | 右键撤销 | C清空 | "
                "Enter保存并下一张 | A上一张 | Q退出"
            )
            if self.pending_box is not None:
                instruction = "请左键点击这个圆环的圆心 | 右键取消当前框"
            title = (
                f"[{self.index + 1}/{len(self.items)}] {split} / {image_path.name} | "
                f"已确认 {self.completed_count()} 张\n{instruction}"
            )
            if message:
                title += f"\n{message}"
            self.axis.set_title(title, fontproperties=CHINESE_FONT, fontsize=10)
        else:
            mode = "ring class 0" if self.class_id == 0 else "distractor class 1"
            instruction = (
                f"Mode: {mode} | 0: ring | 1: distractor | Drag box | "
                "Right click: undo | C: clear | "
                "Enter: save/next | A: previous | Q: quit"
            )
            if self.pending_box is not None:
                instruction = "Click the ring center | Right click: cancel box"
            title = (
                f"[{self.index + 1}/{len(self.items)}] {split} / {image_path.name} | "
                f"confirmed {self.completed_count()}\n{instruction}"
            )
            self.axis.set_title(title, fontsize=10)

    def redraw(self, message=""):
        self.axis.clear()
        self.axis.imshow(self.image_rgb)
        self.axis.set_xlim(0, self.width)
        self.axis.set_ylim(self.height, 0)
        self.axis.set_axis_off()

        for annotation in self.annotations:
            x1, y1, x2, y2 = annotation["box"]
            class_id = annotation["class_id"]
            self.axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime" if class_id == 0 else "magenta",
                    linewidth=2,
                )
            )
            if class_id == 0 and annotation["center"] is not None:
                cx, cy = annotation["center"]
                self.axis.plot(cx, cy, "o", color="red", markersize=6)

        if self.pending_box is not None:
            x1, y1, x2, y2 = self.pending_box
            self.axis.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=2,
                )
            )

        self.update_title(message)
        self.figure.canvas.draw_idle()

    def valid_position(self, event):
        return (
            event.inaxes == self.axis
            and event.xdata is not None
            and event.ydata is not None
        )

    def clamp_point(self, x, y):
        return (
            float(np.clip(x, 0, self.width - 1)),
            float(np.clip(y, 0, self.height - 1)),
        )

    def on_press(self, event):
        if event.button == 3:
            if self.pending_box is not None:
                self.pending_box = None
            elif self.annotations:
                self.annotations.pop()
            self.drag_start = None
            self.drag_patch = None
            self.redraw("已撤销")
            return

        if event.button != 1 or not self.valid_position(event):
            return

        x, y = self.clamp_point(event.xdata, event.ydata)
        if self.pending_box is not None:
            x1, y1, x2, y2 = self.pending_box
            if not (x1 <= x <= x2 and y1 <= y <= y2):
                self.redraw("圆心必须点在黄色框内部")
                return
            self.annotations.append(
                {
                    "class_id": 0,
                    "box": self.pending_box,
                    "center": (x, y),
                    "visibility": 2,
                }
            )
            self.pending_box = None
            self.redraw("已添加一个圆环，可继续拖框")
            return

        self.drag_start = (x, y)
        self.drag_patch = Rectangle(
            (x, y),
            0,
            0,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
        self.axis.add_patch(self.drag_patch)

    def on_motion(self, event):
        if self.drag_start is None or self.drag_patch is None:
            return
        if not self.valid_position(event):
            return

        x, y = self.clamp_point(event.xdata, event.ydata)
        x0, y0 = self.drag_start
        x1, x2 = sorted((x0, x))
        y1, y2 = sorted((y0, y))
        self.drag_patch.set_xy((x1, y1))
        self.drag_patch.set_width(x2 - x1)
        self.drag_patch.set_height(y2 - y1)
        self.figure.canvas.draw_idle()

    def on_release(self, event):
        if event.button != 1 or self.drag_start is None:
            return

        start_x, start_y = self.drag_start
        self.drag_start = None
        self.drag_patch = None

        if not self.valid_position(event):
            self.redraw("框选已取消")
            return

        end_x, end_y = self.clamp_point(event.xdata, event.ydata)
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))
        if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
            self.redraw("框太小，请重新框选")
            return

        self.pending_box = (x1, y1, x2, y2)
        if self.class_id == 0:
            self.redraw("圆环框已确定，现在点击圆心")
        else:
            self.annotations.append(
                {
                    "class_id": 1,
                    "box": self.pending_box,
                    "center": None,
                    "visibility": 0,
                }
            )
            self.pending_box = None
            self.redraw("已添加一个杂物，可继续拖框")

    def save_current(self):
        if self.pending_box is not None:
            self.redraw("黄色框还没有圆心，不能保存")
            return False

        split, image_path = self.items[self.index]
        label_path = self.label_path(split, image_path)
        lines = []

        for annotation in self.annotations:
            x1, y1, x2, y2 = annotation["box"]
            class_id = annotation["class_id"]
            cx = ((x1 + x2) / 2) / self.width
            cy = ((y1 + y2) / 2) / self.height
            bw = (x2 - x1) / self.width
            bh = (y2 - y1) / self.height
            if class_id == 0:
                if annotation["center"] is None:
                    self.redraw("发现没有圆心的圆环标注，请右键撤销后重新标定")
                    return False
                px, py = annotation["center"]
                kx = px / self.width
                ky = py / self.height
                visibility = annotation["visibility"]
            else:
                kx = 0.0
                ky = 0.0
                visibility = 0
            lines.append(
                f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
                f"{kx:.6f} {ky:.6f} {visibility}"
            )

        text = "\n".join(lines)
        if text:
            text += "\n"
        label_path.write_text(text, encoding="utf-8")
        ring_count = sum(item["class_id"] == 0 for item in self.annotations)
        distractor_count = sum(item["class_id"] == 1 for item in self.annotations)
        print(
            f"已保存：{label_path}，圆环 {ring_count} 个，杂物 {distractor_count} 个"
        )
        return True

    def go_next(self):
        if not self.save_current():
            return
        if self.index >= len(self.items) - 1:
            self.redraw("全部图片已经标注完成")
            return
        self.index += 1
        self.load_current()

    def go_previous(self):
        if self.index <= 0:
            self.redraw("已经是第一张")
            return
        self.index -= 1
        self.load_current()

    def on_key(self, event):
        if event.key == "0":
            if self.pending_box is not None:
                self.redraw("请先为当前圆环框点击圆心，或右键取消")
                return
            self.class_id = 0
            self.redraw("已切换到圆环 class 0：拖框后还要点击圆心")
        elif event.key == "1":
            if self.pending_box is not None:
                self.redraw("请先为当前圆环框点击圆心，或右键取消")
                return
            self.class_id = 1
            self.redraw("已切换到杂物 class 1：拖框后直接完成")
        elif event.key in ("enter", "return"):
            self.go_next()
        elif event.key in ("a", "left"):
            self.go_previous()
        elif event.key in ("c", "delete"):
            self.annotations.clear()
            self.pending_box = None
            self.drag_start = None
            self.drag_patch = None
            self.redraw("当前图片的标注已清空，按 Enter 才会保存")
        elif event.key in ("q", "escape"):
            plt.close(self.figure)

    def run(self):
        plt.tight_layout()
        plt.show()


def main():
    if CHINESE_FONT is None:
        print("没有找到中文字体，窗口标题将暂时使用英文。")
        print("Ubuntu安装命令：sudo apt install fonts-noto-cjk")
    else:
        print(f"中文字体：{CHINESE_FONT.get_file()}")
    ensure_dataset_layout()
    extract_frames()
    items = collect_images()
    if not items:
        raise RuntimeError("没有抽取到任何图片，请检查 assets/input 中的视频")

    print(f"共有 {len(items)} 张待检查图片")
    print("按0标圆环：绿色框包住圆环，再点击圆心生成红点。")
    print("按1标杂物：紫色框包住易混淆区域，不需要点击圆心。")
    RingPoseAnnotator(items).run()


if __name__ == "__main__":
    main()