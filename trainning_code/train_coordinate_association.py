from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from coordinate_association_model import (
    ARCHITECTURE_VERSION,
    CONTEXT_RADIUS,
    CoordinateAssociationModel,
    build_temporal_context,
    normalize_detection_features,
)


DATA_DIR = PROJECT_ROOT / "coordinate_dataset" / "scenes"
OUTPUT_DIR = PROJECT_ROOT / "model" / "coordinate_tracker"
BEST_MODEL_PATH = OUTPUT_DIR / "best_coordinate_association.pt"

HISTORY_LENGTH = 15
HIDDEN_SIZE = 64
BATCH_SIZE = 32
MAX_EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_RATIO = 0.20
EARLY_STOP_PATIENCE = 100
MIN_IMPROVEMENT = 1e-4
COORDINATE_NOISE_STD = 0.001
FALSE_NEW_PENALTY_WEIGHT = 1.5
FALSE_NEW_MARGIN = 1.0
RANDOM_SEED = 42
DEVICE = None


class CoordinateSceneDataset(Dataset):
    def __init__(
        self,
        scene_paths,
        max_tracks,
        max_detections,
        augment=False,
    ):
        self.max_tracks = max_tracks
        self.max_detections = max_detections
        self.augment = augment
        self.scenes = []
        self.sample_indices = []
        self.legacy_unassigned_count = 0

        for scene_path in scene_paths:
            with np.load(scene_path) as data:
                required_fields = {
                    "positions",
                    "visible",
                    "detection_features",
                    "detection_mask",
                    "detection_ids",
                    "width",
                    "height",
                }
                missing_fields = required_fields - set(data.files)
                if missing_fields:
                    raise RuntimeError(
                        f"{scene_path.name}缺少新数据字段："
                        f"{sorted(missing_fields)}"
                    )

                width = float(data["width"])
                height = float(data["height"])
                detection_features = data["detection_features"].astype(
                    np.float32
                )
                detection_ids = data["detection_ids"].astype(np.int64)
                raw_detection_mask = data["detection_mask"].astype(bool)
                legacy_unassigned_mask = (
                    raw_detection_mask & (detection_ids <= 0)
                )
                self.legacy_unassigned_count += int(
                    legacy_unassigned_mask.sum()
                )

                # 保持旧 NPZ 的格式和内容不变。旧数据中 ID=0 的候选没有
                # 可恢复的身份标签，只从关联训练中排除；新标注器会阻止
                # 继续提交这种候选。
                effective_detection_mask = (
                    raw_detection_mask & (detection_ids > 0)
                )
                scene = {
                    "positions": data["positions"].astype(np.float32),
                    "visible": data["visible"].astype(bool),
                    "detection_features": normalize_detection_features(
                        detection_features,
                        width,
                        height,
                    ),
                    "detection_mask": effective_detection_mask,
                    "detection_ids": detection_ids,
                    "width": width,
                    "height": height,
                }

            scene_index = len(self.scenes)
            self.scenes.append(scene)
            for frame_index in range(len(scene["positions"])):
                if scene["detection_mask"][frame_index].any():
                    self.sample_indices.append((scene_index, frame_index))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        scene_index, frame_index = self.sample_indices[index]
        scene = self.scenes[scene_index]
        positions = scene["positions"]
        visible = scene["visible"]
        track_count = positions.shape[1]
        scale = np.array(
            [scene["width"], scene["height"]],
            dtype=np.float32,
        )

        history = np.zeros(
            (self.max_tracks, HISTORY_LENGTH, 3),
            dtype=np.float32,
        )
        track_mask = np.zeros(self.max_tracks, dtype=bool)

        history_start = max(0, frame_index - HISTORY_LENGTH)
        history_frame_indices = list(range(history_start, frame_index))
        destination_start = HISTORY_LENGTH - len(history_frame_indices)

        for track_index in range(track_count):
            last_position = np.zeros(2, dtype=np.float32)
            has_been_seen = False

            earlier_visible_frames = np.flatnonzero(
                visible[:history_start, track_index]
            )
            if len(earlier_visible_frames) > 0:
                last_visible_frame = int(earlier_visible_frames[-1])
                last_position = positions[last_visible_frame, track_index] / scale
                has_been_seen = True

            for offset, source_frame in enumerate(history_frame_indices):
                destination_frame = destination_start + offset
                is_visible = bool(visible[source_frame, track_index])

                if is_visible:
                    last_position = positions[source_frame, track_index] / scale
                    has_been_seen = True

                if has_been_seen:
                    history[track_index, destination_frame, :2] = last_position
                history[track_index, destination_frame, 2] = float(is_visible)

            track_mask[track_index] = has_been_seen

        scene_detection_features = scene["detection_features"]
        scene_detection_mask = scene["detection_mask"]
        current_raw_slots = np.flatnonzero(scene_detection_mask[frame_index])
        current_features = scene_detection_features[
            frame_index,
            current_raw_slots,
        ]
        current_ids = scene["detection_ids"][frame_index, current_raw_slots]
        current_context = build_temporal_context(
            scene_detection_features,
            scene_detection_mask,
            frame_index,
            current_features,
        )

        order = np.arange(len(current_features))
        np.random.shuffle(order)

        detections = np.zeros(
            (self.max_detections, current_features.shape[-1]),
            dtype=np.float32,
        )
        detection_context = np.zeros(
            (
                self.max_detections,
                CONTEXT_RADIUS * 2 + 1,
                current_context.shape[-1],
            ),
            dtype=np.float32,
        )
        detection_mask = np.zeros(self.max_detections, dtype=bool)
        targets = np.full(
            self.max_detections,
            self.max_tracks,
            dtype=np.int64,
        )

        for detection_slot, source_index in enumerate(order):
            detection = current_features[source_index].copy()
            if self.augment:
                detection[:2] += np.random.normal(
                    0.0,
                    COORDINATE_NOISE_STD,
                    size=2,
                ).astype(np.float32)
                detection[:2] = np.clip(detection[:2], 0.0, 1.0)

            detections[detection_slot] = detection
            detection_context[detection_slot] = current_context[source_index]
            detection_mask[detection_slot] = True

            track_id = int(current_ids[source_index])
            if track_id <= 0:
                raise RuntimeError(
                    "有效训练候选必须拥有正 ID；请检查 detection_mask"
                )

            track_index = track_id - 1
            if track_index >= self.max_tracks:
                raise RuntimeError(
                    f"检测ID {track_id}超过场景最大轨迹数{self.max_tracks}"
                )
            if track_mask[track_index]:
                targets[detection_slot] = track_index
            else:
                targets[detection_slot] = self.max_tracks

        return {
            "history": torch.from_numpy(history),
            "track_mask": torch.from_numpy(track_mask),
            "detections": torch.from_numpy(detections),
            "detection_context": torch.from_numpy(detection_context),
            "detection_mask": torch.from_numpy(detection_mask),
            "targets": torch.from_numpy(targets),
        }


def choose_device():
    if DEVICE is not None:
        return DEVICE
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def split_scene_paths():
    scene_paths = sorted(DATA_DIR.glob("*.npz"))
    if len(scene_paths) < 2:
        raise RuntimeError(
            f"至少需要2个场景才能按场景划分训练集和验证集，目前只有"
            f"{len(scene_paths)}个"
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(scene_paths)
    validation_count = max(1, int(round(len(scene_paths) * VALIDATION_RATIO)))
    validation_paths = scene_paths[:validation_count]
    training_paths = scene_paths[validation_count:]
    return training_paths, validation_paths


def get_dataset_sizes(scene_paths):
    max_tracks = 0
    max_detections = 0

    for scene_path in scene_paths:
        with np.load(scene_path) as data:
            if "detection_features" not in data.files:
                raise RuntimeError(
                    f"{scene_path.name}是旧版NPZ，请使用新视频标注器重新生成"
                )
            max_tracks = max(max_tracks, int(data["positions"].shape[1]))
            max_detections = max(
                max_detections,
                int(data["detection_features"].shape[1]),
            )

    if max_tracks <= 0:
        raise RuntimeError("数据集中没有有效轨迹")
    if max_detections <= 0:
        raise RuntimeError("数据集中没有YOLO候选圆心")
    return max_tracks, max_detections


def calculate_loss(logits, targets, detection_mask, max_tracks):
    existing_mask = detection_mask & (targets < max_tracks)
    new_mask = detection_mask & (targets == max_tracks)

    losses = []
    for category_mask in (existing_mask, new_mask):
        if category_mask.any():
            losses.append(
                nn.functional.cross_entropy(
                    logits[category_mask],
                    targets[category_mask],
                )
            )

    if not losses:
        return logits.sum() * 0.0

    loss = torch.stack(losses).mean()

    # 旧轨迹被错判成 NEW 会永久制造新 ID，因此给予额外且带间隔的惩罚。
    if existing_mask.any():
        correct_existing_logits = logits.gather(
            dim=-1,
            index=targets.unsqueeze(-1),
        ).squeeze(-1)
        new_logits = logits[..., max_tracks]
        false_new_penalty = nn.functional.relu(
            new_logits[existing_mask]
            - correct_existing_logits[existing_mask]
            + FALSE_NEW_MARGIN
        ).mean()
        loss = loss + FALSE_NEW_PENALTY_WEIGHT * false_new_penalty

    return loss


def run_epoch(model, loader, device, max_tracks, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    batch_count = 0
    correct_counts = [0, 0]
    total_counts = [0, 0]
    false_new_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            history = batch["history"].to(device)
            track_mask = batch["track_mask"].to(device)
            detections = batch["detections"].to(device)
            detection_context = batch["detection_context"].to(device)
            detection_mask = batch["detection_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(
                history,
                track_mask,
                detections,
                detection_context,
                detection_mask,
            )
            loss = calculate_loss(
                logits,
                targets,
                detection_mask,
                max_tracks,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

            predictions = logits.argmax(dim=-1)
            category_masks = (
                detection_mask & (targets < max_tracks),
                detection_mask & (targets == max_tracks),
            )
            for category_index, category_mask in enumerate(category_masks):
                correct_counts[category_index] += int(
                    ((predictions == targets) & category_mask).sum().item()
                )
                total_counts[category_index] += int(category_mask.sum().item())
            false_new_count += int(
                (
                    (predictions == max_tracks)
                    & category_masks[0]
                ).sum().item()
            )

    average_loss = total_loss / max(batch_count, 1)
    accuracies = [
        correct / max(total, 1)
        for correct, total in zip(correct_counts, total_counts)
    ]
    false_new_rate = false_new_count / max(total_counts[0], 1)
    return (
        average_loss,
        accuracies[0],
        accuracies[1],
        false_new_rate,
    )


def save_checkpoint(
    model,
    epoch,
    validation_loss,
    validation_existing_accuracy,
    validation_new_accuracy,
    validation_false_new_rate,
    max_tracks,
    max_detections,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_existing_accuracy": validation_existing_accuracy,
            "validation_new_accuracy": validation_new_accuracy,
            "validation_false_new_rate": validation_false_new_rate,
            "max_tracks": max_tracks,
            "max_detections": max_detections,
            "history_length": HISTORY_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "context_radius": CONTEXT_RADIUS,
            "architecture_version": ARCHITECTURE_VERSION,
            "false_new_penalty_weight": FALSE_NEW_PENALTY_WEIGHT,
            "false_new_margin": FALSE_NEW_MARGIN,
        },
        BEST_MODEL_PATH,
    )


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    training_paths, validation_paths = split_scene_paths()
    all_paths = training_paths + validation_paths
    max_tracks, max_detections = get_dataset_sizes(all_paths)

    training_dataset = CoordinateSceneDataset(
        training_paths,
        max_tracks,
        max_detections,
        augment=True,
    )
    validation_dataset = CoordinateSceneDataset(
        validation_paths,
        max_tracks,
        max_detections,
        augment=False,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = choose_device()
    model = CoordinateAssociationModel(
        max_tracks=max_tracks,
        hidden_size=HIDDEN_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Device: {device}")
    print(f"Context: previous {CONTEXT_RADIUS} + current + next {CONTEXT_RADIUS}")
    print(f"Maximum tracks in one scene: {max_tracks}")
    print(f"Maximum detections in one frame: {max_detections}")
    print(
        f"False NEW penalty: weight={FALSE_NEW_PENALTY_WEIGHT}, "
        f"margin={FALSE_NEW_MARGIN}"
    )
    print(
        f"Training scenes: {len(training_paths)}, "
        f"validation scenes: {len(validation_paths)}"
    )
    print(
        f"Training samples: {len(training_dataset)}, "
        f"validation samples: {len(validation_dataset)}"
    )
    legacy_unassigned_count = (
        training_dataset.legacy_unassigned_count
        + validation_dataset.legacy_unassigned_count
    )
    if legacy_unassigned_count > 0:
        print(
            f"Legacy ID=0 candidates excluded without changing NPZ: "
            f"{legacy_unassigned_count}"
        )

    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        (
            train_loss,
            train_existing,
            train_new,
            train_false_new,
        ) = run_epoch(
            model,
            training_loader,
            device,
            max_tracks,
            optimizer,
        )
        (
            val_loss,
            val_existing,
            val_new,
            val_false_new,
        ) = run_epoch(
            model,
            validation_loader,
            device,
            max_tracks,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.5f}, existing={train_existing:.4f}, "
            f"new={train_new:.4f}, "
            f"false_new={train_false_new:.4f} | "
            f"val loss={val_loss:.5f}, existing={val_existing:.4f}, "
            f"new={val_new:.4f}, "
            f"false_new={val_false_new:.4f}"
        )

        if val_loss < best_validation_loss - MIN_IMPROVEMENT:
            best_validation_loss = val_loss
            stale_epochs = 0
            save_checkpoint(
                model,
                epoch,
                val_loss,
                val_existing,
                val_new,
                val_false_new,
                max_tracks,
                max_detections,
            )
            print(f"Saved best model: {BEST_MODEL_PATH}")
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping after {stale_epochs} epochs "
                    f"without improvement"
                )
                break

    print(f"Best model: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()