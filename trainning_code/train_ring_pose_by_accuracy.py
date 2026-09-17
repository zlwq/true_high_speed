import csv
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "ring_dataset"
DATASET_YAML = DATASET_DIR / "ring_pose.yaml"
PRETRAINED_MODEL = PROJECT_ROOT / "model" / "pretrained" / "yolo26s-pose.pt"
OUTPUT_DIR = PROJECT_ROOT / "model" / "ring_pose"

MAX_EPOCHS = 400
IMAGE_SIZE = 640
GPU_BATCH_SIZE = -1
CPU_BATCH_SIZE = 8
DEVICE = None
WORKERS = 4
BUILTIN_PATIENCE = 0
CLASS0_MIN_EPOCHS = 50
CLASS0_PATIENCE = 50
CLASS0_MIN_DELTA = 0.0005
RUN_NAME = "yolo26n_pose_ring"
RUN_TEST_AFTER_TRAINING = True

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def f1_score(precision, recall):
    total = precision + recall
    return 0.0 if total == 0 else 2.0 * precision * recall / total


def get_class0_metrics(metrics):
    box_precision, box_recall, _, _ = metrics.box.class_result(0)
    pose_precision, pose_recall, _, _ = metrics.pose.class_result(0)

    box_precision = float(box_precision)
    box_recall = float(box_recall)
    pose_precision = float(pose_precision)
    pose_recall = float(pose_recall)

    box_f1 = f1_score(box_precision, box_recall)
    pose_f1 = f1_score(pose_precision, pose_recall)
    score = (box_f1 + pose_f1) / 2.0

    return {
        "box_precision": box_precision,
        "box_recall": box_recall,
        "box_f1": box_f1,
        "pose_precision": pose_precision,
        "pose_recall": pose_recall,
        "pose_f1": pose_f1,
        "score": score,
    }


class Class0EarlyStopping:
    def __init__(self):
        self.best_score = -1.0
        self.best_epoch = 0
        self.stale_epochs = 0
        self.last_epoch = 0
        self.best_model_path = None
        self.csv_path = None

    def save_metrics(self, epoch, values):
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(
                    [
                        "epoch",
                        "class0_box_precision",
                        "class0_box_recall",
                        "class0_box_f1",
                        "class0_pose_precision",
                        "class0_pose_recall",
                        "class0_pose_f1",
                        "class0_score",
                        "best_score",
                        "epochs_without_improvement",
                    ]
                )
            writer.writerow(
                [
                    epoch,
                    values["box_precision"],
                    values["box_recall"],
                    values["box_f1"],
                    values["pose_precision"],
                    values["pose_recall"],
                    values["pose_f1"],
                    values["score"],
                    self.best_score,
                    self.stale_epochs,
                ]
            )

    def __call__(self, trainer):
        epoch = trainer.epoch + 1
        if epoch <= self.last_epoch:
            return
        self.last_epoch = epoch

        if self.csv_path is None:
            self.csv_path = Path(trainer.save_dir) / "class0_metrics.csv"
            self.best_model_path = Path(trainer.wdir) / "best_ring_class0.pt"

        values = get_class0_metrics(trainer.validator.metrics)
        improved = values["score"] > self.best_score + CLASS0_MIN_DELTA

        if improved:
            self.best_score = values["score"]
            self.best_epoch = epoch
            self.stale_epochs = 0
            if not Path(trainer.last).exists():
                raise RuntimeError(f"找不到当前轮模型：{trainer.last}")
            shutil.copy2(trainer.last, self.best_model_path)
            status = "已保存新的class 0最佳模型"
        else:
            self.stale_epochs += 1
            status = f"连续{self.stale_epochs}轮没有明显提升"

        self.save_metrics(epoch, values)
        print(
            f"\n[class 0] epoch {epoch}: "
            f"Box P={values['box_precision']:.4f}, R={values['box_recall']:.4f}, "
            f"F1={values['box_f1']:.4f} | "
            f"Pose P={values['pose_precision']:.4f}, R={values['pose_recall']:.4f}, "
            f"F1={values['pose_f1']:.4f} | "
            f"综合分数={values['score']:.4f} | {status}"
        )

        if epoch >= CLASS0_MIN_EPOCHS and self.stale_epochs >= CLASS0_PATIENCE:
            print(
                f"\nclass 0指标已经连续{CLASS0_PATIENCE}轮没有至少"
                f"{CLASS0_MIN_DELTA:.4f}的提升，停止训练。"
                f"最佳轮次：{self.best_epoch}，最佳分数：{self.best_score:.4f}"
            )
            trainer.stop = True


def write_dataset_yaml():
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
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_YAML.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def validate_label(label_path):
    ring_count = 0
    distractor_count = 0
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 8:
            raise ValueError(
                f"{label_path} 第 {line_number} 行应有8个数，实际为{len(parts)}个"
            )

        try:
            values = [float(value) for value in parts]
        except ValueError as error:
            raise ValueError(
                f"{label_path} 第 {line_number} 行包含非数字内容"
            ) from error

        class_id_value = values[0]
        class_id = int(class_id_value)
        if class_id_value != class_id or class_id not in (0, 1):
            raise ValueError(
                f"{label_path} 第 {line_number} 行类别必须为0或1，实际为{class_id_value}"
            )

        cx, cy, width, height, keypoint_x, keypoint_y = values[1:7]
        visibility = values[7]
        if not all(0.0 <= value <= 1.0 for value in values[1:7]):
            raise ValueError(
                f"{label_path} 第 {line_number} 行的坐标必须在0到1之间"
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行的框宽和框高必须大于0"
            )
        if visibility not in (0, 1, 2):
            raise ValueError(
                f"{label_path} 第 {line_number} 行的可见性必须是0、1或2"
            )
        if class_id == 0 and visibility == 0:
            raise ValueError(
                f"{label_path} 第 {line_number} 行是圆环(class 0)，必须标记圆心"
            )
        if class_id == 1 and (
            keypoint_x != 0 or keypoint_y != 0 or visibility != 0
        ):
            raise ValueError(
                f"{label_path} 第 {line_number} 行是杂物(class 1)，"
                "关键点必须写成0 0 0"
            )
        if visibility > 0 and not (
            cx - width / 2 <= keypoint_x <= cx + width / 2
            and cy - height / 2 <= keypoint_y <= cy + height / 2
        ):
            raise ValueError(
                f"{label_path} 第 {line_number} 行的圆心不在圆环框内部"
            )
        if class_id == 0:
            ring_count += 1
        else:
            distractor_count += 1

    return ring_count, distractor_count


def inspect_split(split, required):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if required and not image_paths:
        raise RuntimeError(f"{split} 集没有图片：{image_dir}")

    missing_labels = []
    ring_count = 0
    distractor_count = 0
    ring_images = 0
    distractor_images = 0
    background_images = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        image_ring_count, image_distractor_count = validate_label(label_path)
        ring_count += image_ring_count
        distractor_count += image_distractor_count
        if image_ring_count > 0:
            ring_images += 1
        if image_distractor_count > 0:
            distractor_images += 1
        if image_ring_count == 0 and image_distractor_count == 0:
            background_images += 1

    if missing_labels:
        preview = ", ".join(missing_labels[:8])
        raise RuntimeError(
            f"{split} 集有 {len(missing_labels)} 张图片尚未确认标注，例如：{preview}。"
            "请先运行 prepare_ring_pose_dataset.py，并对每张图片按 Enter 确认；"
            "没有圆环的图片也要按 Enter 生成空标签。"
        )

    if required and ring_count == 0:
        raise RuntimeError(f"{split} 集没有任何圆环标注")

    print(
        f"{split:5s}: 图片 {len(image_paths):4d}，"
        f"圆环图片 {ring_images:4d} / 实例 {ring_count:5d}，"
        f"杂物图片 {distractor_images:4d} / 实例 {distractor_count:5d}，"
        f"纯背景图片 {background_images:4d}"
    )
    return len(image_paths)


def choose_device_and_batch():
    if DEVICE is not None:
        device = DEVICE
    elif torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    batch_size = GPU_BATCH_SIZE if device != "cpu" else CPU_BATCH_SIZE
    return device, batch_size


def main():
    write_dataset_yaml()
    print("检查数据集……")
    inspect_split("train", required=True)
    inspect_split("val", required=True)
    test_image_count = inspect_split("test", required=False)

    PRETRAINED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device, batch_size = choose_device_and_batch()
    print(f"训练设备：{device}，batch：{batch_size}")
    print(f"预训练模型：{PRETRAINED_MODEL}")

    model = YOLO(str(PRETRAINED_MODEL))
    class0_early_stopping = Class0EarlyStopping()
    model.add_callback("on_fit_epoch_end", class0_early_stopping)
    train_arguments = {
        "data": str(DATASET_YAML),
        "epochs": MAX_EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": batch_size,
        "device": device,
        "workers": WORKERS,
        "patience": BUILTIN_PATIENCE,
        "project": str(OUTPUT_DIR),
        "name": RUN_NAME,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "auto",
        "amp": True,
        "seed": 42,
        "deterministic": True,
        "plots": True,
        "save": True,
        "save_period": -1,
        "hsv_h": 0.01,
        "hsv_s": 0.30,
        "hsv_v": 0.35,
        "degrees": 20.0,
        "translate": 0.10,
        "scale": 0.35,
        "shear": 3.0,
        "perspective": 0.0005,
        "flipud": 0.20,
        "fliplr": 0.50,
        "mosaic": 0.50,
        "mixup": 0.0,
        "close_mosaic": 15,
    }

    model.train(**train_arguments)

    save_dir = Path(model.trainer.save_dir)
    best_model = class0_early_stopping.best_model_path
    if best_model is None or not best_model.exists():
        raise RuntimeError("训练结束后没有生成class 0最佳模型")
    print(
        f"训练完成，class 0最佳轮次：{class0_early_stopping.best_epoch}，"
        f"最佳分数：{class0_early_stopping.best_score:.4f}"
    )
    print(f"class 0最佳模型：{best_model}")
    print(f"class 0逐轮指标：{save_dir / 'class0_metrics.csv'}")

    if RUN_TEST_AFTER_TRAINING and test_image_count > 0 and best_model.exists():
        print("开始在test集上评估……")
        test_model = YOLO(str(best_model))
        test_results = test_model.val(
            data=str(DATASET_YAML),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=16,
            device=device,
            workers=WORKERS,
            plots=True,
        )
        values = get_class0_metrics(test_results)
        print(
            "\ntest集class 0最终结果："
            f"Box P={values['box_precision']:.4f}, R={values['box_recall']:.4f}, "
            f"F1={values['box_f1']:.4f} | "
            f"Pose P={values['pose_precision']:.4f}, R={values['pose_recall']:.4f}, "
            f"F1={values['pose_f1']:.4f}"
        )


if __name__ == "__main__":
    main()