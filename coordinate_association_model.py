import numpy as np
import torch
from torch import nn


CONTEXT_RADIUS = 10
ARCHITECTURE_VERSION = 3
DETECTION_FEATURE_SIZE = 6
CONTEXT_FEATURE_SIZE = 8
TRACK_RELATION_FEATURE_SIZE = 9
MOTION_FEATURE_SIZE = 7
PAIR_GEOMETRY_SIZE = 11
NEW_EVIDENCE_SIZE = 6


def normalize_detection_features(features, width, height):
    normalized = np.asarray(features, dtype=np.float32).copy()
    normalized[..., 0] /= float(width)
    normalized[..., 1] /= float(height)
    normalized[..., 4] /= float(width)
    normalized[..., 5] /= float(height)
    return normalized


def build_temporal_context(
    all_detection_features,
    all_detection_masks,
    frame_index,
    current_detection_features,
    radius=CONTEXT_RADIUS,
):
    detection_count = len(current_detection_features)
    window_length = radius * 2 + 1
    context = np.zeros(
        (detection_count, window_length, CONTEXT_FEATURE_SIZE),
        dtype=np.float32,
    )

    frame_count = len(all_detection_features)
    for detection_index, current_feature in enumerate(current_detection_features):
        anchor = current_feature[:2]

        for window_index, offset in enumerate(range(-radius, radius + 1)):
            neighbor_frame = frame_index + offset
            if neighbor_frame < 0 or neighbor_frame >= frame_count:
                continue

            neighbor_mask = all_detection_masks[neighbor_frame]
            neighbor_features = all_detection_features[neighbor_frame][neighbor_mask]
            if len(neighbor_features) == 0:
                continue

            deltas = neighbor_features[:, :2] - anchor
            distances = np.linalg.norm(deltas, axis=1)
            nearest_index = int(np.argmin(distances))
            nearest_feature = neighbor_features[nearest_index]

            context[detection_index, window_index] = np.array(
                [
                    deltas[nearest_index, 0],
                    deltas[nearest_index, 1],
                    distances[nearest_index],
                    nearest_feature[2],
                    nearest_feature[3],
                    nearest_feature[4],
                    nearest_feature[5],
                    1.0,
                ],
                dtype=np.float32,
            )

    return context


class CoordinateAssociationModel(nn.Module):
    def __init__(self, max_tracks=None, hidden_size=64):
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be even")

        self.max_tracks = max_tracks
        self.hidden_size = hidden_size
        dropout_probability = 0.10

        self.history_encoder = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.context_encoder = nn.GRU(
            input_size=CONTEXT_FEATURE_SIZE,
            hidden_size=hidden_size // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.detection_encoder = nn.Sequential(
            nn.Linear(DETECTION_FEATURE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 从多条轨迹的相对距离、相对速度和共同可见性中学习软刚体关系。
        self.rigidity_gate = nn.Sequential(
            nn.Linear(TRACK_RELATION_FEATURE_SIZE, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.track_message = nn.Sequential(
            nn.Linear(hidden_size + TRACK_RELATION_FEATURE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.track_update = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_size, hidden_size),
        )
        self.track_norm = nn.LayerNorm(hidden_size)

        # 预测下一位置和不确定度，掉线后不再只比较最后一次坐标。
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_size + MOTION_FEATURE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

        # 同一帧候选点之间交换相对位置和形状信息。
        self.detection_relation_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.detection_message = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.detection_update = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_size, hidden_size),
        )
        self.detection_norm = nn.LayerNorm(hidden_size)

        self.pair_scorer = nn.Sequential(
            nn.Linear(
                hidden_size * 2 + PAIR_GEOMETRY_SIZE,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_size, 1),
        )

        # NEW 必须同时参考全部旧轨迹，而不是只看当前候选本身。
        self.new_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2 + NEW_EVIDENCE_SIZE, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_size, 1),
        )
        self.new_logit_bias = nn.Parameter(torch.tensor(-1.0))

    def _encode_tracks(self, history, track_mask):
        batch_size, track_count, history_length, _ = history.shape
        if track_count == 0:
            empty_embeddings = history.new_zeros(
                (batch_size, 0, self.hidden_size)
            )
            empty_positions = history.new_zeros((batch_size, 0, 2))
            empty_scalars = history.new_zeros((batch_size, 0, 1))
            empty_relations = history.new_zeros((batch_size, 0, 0))
            return (
                empty_embeddings,
                empty_positions,
                empty_positions,
                empty_scalars,
                history.new_zeros((batch_size, 0, MOTION_FEATURE_SIZE)),
                empty_relations,
                empty_relations,
            )

        history_flat = history.reshape(
            batch_size * track_count,
            history_length,
            3,
        )
        _, history_hidden = self.history_encoder(history_flat)
        track_embeddings = history_hidden[-1].reshape(
            batch_size,
            track_count,
            self.hidden_size,
        )

        positions = history[..., :2]
        visibility = history[..., 2].clamp(0.0, 1.0)
        last_positions = positions[:, :, -1]

        transition_mask = visibility[:, :, 1:] * visibility[:, :, :-1]
        position_steps = positions[:, :, 1:] - positions[:, :, :-1]
        velocity = (
            (position_steps * transition_mask.unsqueeze(-1)).sum(dim=2)
            / transition_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        )
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        trailing_missing = torch.cumprod(
            1.0 - visibility.flip(dims=[-1]),
            dim=-1,
        ).sum(dim=-1, keepdim=True) / float(history_length)
        visible_ratio = visibility.mean(dim=-1, keepdim=True)
        motion_features = torch.cat(
            [
                last_positions,
                velocity,
                speed,
                trailing_missing,
                visible_ratio,
            ],
            dim=-1,
        )

        pair_delta_history = positions.unsqueeze(2) - positions.unsqueeze(1)
        pair_distance_history = torch.linalg.norm(
            pair_delta_history,
            dim=-1,
        )
        co_visible = visibility.unsqueeze(2) * visibility.unsqueeze(1)
        co_visible_count = co_visible.sum(dim=-1)
        mean_distance = (
            (pair_distance_history * co_visible).sum(dim=-1)
            / co_visible_count.clamp_min(1.0)
        )
        distance_variance = (
            (
                (pair_distance_history - mean_distance.unsqueeze(-1)).square()
                * co_visible
            ).sum(dim=-1)
            / co_visible_count.clamp_min(1.0)
        )
        distance_std = torch.sqrt(distance_variance + 1e-8)

        relative_position = (
            last_positions.unsqueeze(2) - last_positions.unsqueeze(1)
        )
        current_distance = torch.linalg.norm(
            relative_position,
            dim=-1,
            keepdim=True,
        )
        relative_velocity = velocity.unsqueeze(2) - velocity.unsqueeze(1)
        co_visible_ratio = (
            co_visible_count / float(history_length)
        ).unsqueeze(-1)
        visibility_difference = torch.abs(
            visibility.unsqueeze(2) - visibility.unsqueeze(1)
        ).mean(dim=-1, keepdim=True)
        relation_features = torch.cat(
            [
                relative_position,
                current_distance,
                mean_distance.unsqueeze(-1),
                distance_std.unsqueeze(-1),
                relative_velocity,
                co_visible_ratio,
                visibility_difference,
            ],
            dim=-1,
        )

        valid_edges = track_mask.unsqueeze(2) & track_mask.unsqueeze(1)
        valid_edges &= co_visible_count > 0
        identity = torch.eye(
            track_count,
            dtype=torch.bool,
            device=history.device,
        ).unsqueeze(0)
        valid_edges &= ~identity

        rigidity_weights = torch.sigmoid(
            self.rigidity_gate(relation_features).squeeze(-1)
        ) * valid_edges.to(history.dtype)
        neighbor_embeddings = track_embeddings.unsqueeze(1).expand(
            -1,
            track_count,
            -1,
            -1,
        )
        messages = self.track_message(
            torch.cat([neighbor_embeddings, relation_features], dim=-1)
        )
        message_sum = (
            messages * rigidity_weights.unsqueeze(-1)
        ).sum(dim=2)
        weight_sum = rigidity_weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        relational_message = message_sum / weight_sum
        track_embeddings = self.track_norm(
            track_embeddings
            + self.track_update(
                torch.cat([track_embeddings, relational_message], dim=-1)
            )
        )
        track_embeddings = track_embeddings * track_mask.unsqueeze(-1)

        motion_output = self.motion_head(
            torch.cat([track_embeddings, motion_features], dim=-1)
        )
        missing_steps = 1.0 + trailing_missing * float(history_length)
        predicted_positions = (
            last_positions
            + velocity * missing_steps
            + 0.10 * torch.tanh(motion_output[..., :2])
        )
        uncertainty = nn.functional.softplus(
            motion_output[..., 2:3]
        ) + 1e-3
        return (
            track_embeddings,
            last_positions,
            predicted_positions,
            uncertainty,
            motion_features,
            rigidity_weights,
            mean_distance,
        )

    def _encode_detections(
        self,
        detections,
        detection_context,
        detection_mask,
    ):
        batch_size, detection_count, _ = detections.shape
        context_flat = detection_context.reshape(
            batch_size * detection_count,
            detection_context.shape[2],
            CONTEXT_FEATURE_SIZE,
        )
        _, context_hidden = self.context_encoder(context_flat)
        context_embeddings = torch.cat(
            [context_hidden[-2], context_hidden[-1]],
            dim=-1,
        ).reshape(batch_size, detection_count, self.hidden_size)
        detection_embeddings = (
            self.detection_encoder(detections) + context_embeddings
        )

        detection_i = detection_embeddings.unsqueeze(2).expand(
            -1,
            -1,
            detection_count,
            -1,
        )
        detection_j = detection_embeddings.unsqueeze(1).expand(
            -1,
            detection_count,
            -1,
            -1,
        )
        relative_delta = (
            detections[:, :, :2].unsqueeze(2)
            - detections[:, :, :2].unsqueeze(1)
        )
        relative_distance = torch.linalg.norm(
            relative_delta,
            dim=-1,
            keepdim=True,
        )
        geometry = torch.cat(
            [relative_delta, relative_distance],
            dim=-1,
        )

        valid_edges = detection_mask.unsqueeze(2) & detection_mask.unsqueeze(1)
        identity = torch.eye(
            detection_count,
            dtype=torch.bool,
            device=detections.device,
        ).unsqueeze(0)
        valid_edges &= ~identity
        relation_weights = torch.sigmoid(
            self.detection_relation_gate(
                torch.cat([detection_i, detection_j, geometry], dim=-1)
            ).squeeze(-1)
        ) * valid_edges.to(detections.dtype)
        messages = self.detection_message(
            torch.cat([detection_j, geometry], dim=-1)
        )
        message_sum = (
            messages * relation_weights.unsqueeze(-1)
        ).sum(dim=2)
        weight_sum = relation_weights.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1e-6)
        relational_message = message_sum / weight_sum
        detection_embeddings = self.detection_norm(
            detection_embeddings
            + self.detection_update(
                torch.cat(
                    [detection_embeddings, relational_message],
                    dim=-1,
                )
            )
        )
        return detection_embeddings

    def forward(
        self,
        history,
        track_mask,
        detections,
        detection_context,
        detection_mask,
    ):
        batch_size, track_count, _, _ = history.shape
        detection_count = detections.shape[1]
        if detection_count == 0:
            return detections.new_zeros(
                (batch_size, 0, track_count + 1)
            )

        (
            track_embeddings,
            last_track_positions,
            predicted_track_positions,
            track_uncertainty,
            motion_features,
            rigidity_weights,
            mean_track_distances,
        ) = self._encode_tracks(history, track_mask)
        detection_embeddings = self._encode_detections(
            detections,
            detection_context,
            detection_mask,
        )

        context_center = detection_context.shape[2] // 2
        past_presence = detection_context[
            :, :, :context_center, 7
        ].mean(dim=2, keepdim=True)
        future_presence = detection_context[
            :, :, context_center + 1 :, 7
        ].mean(dim=2, keepdim=True)

        if track_count > 0:
            track_expanded = track_embeddings.unsqueeze(1).expand(
                -1,
                detection_count,
                -1,
                -1,
            )
            detection_expanded = detection_embeddings.unsqueeze(2).expand(
                -1,
                -1,
                track_count,
                -1,
            )

            delta_last = (
                detections[:, :, :2].unsqueeze(2)
                - last_track_positions.unsqueeze(1)
            )
            distance_last = torch.linalg.norm(
                delta_last,
                dim=-1,
                keepdim=True,
            )
            delta_predicted = (
                detections[:, :, :2].unsqueeze(2)
                - predicted_track_positions.unsqueeze(1)
            )
            distance_predicted = torch.linalg.norm(
                delta_predicted,
                dim=-1,
                keepdim=True,
            )
            normalized_predicted_distance = (
                distance_predicted
                / track_uncertainty.unsqueeze(1).clamp_min(1e-3)
            )

            candidate_to_neighbor_distance = torch.linalg.norm(
                detections[:, :, None, None, :2]
                - predicted_track_positions[:, None, None, :, :],
                dim=-1,
            )
            rigidity_error = torch.abs(
                candidate_to_neighbor_distance
                - mean_track_distances[:, None, :, :]
            )
            rigidity_weight_sum = rigidity_weights.sum(
                dim=-1,
            ).clamp_min(1e-6)
            rigidity_residual = (
                (rigidity_error * rigidity_weights[:, None]).sum(dim=-1)
                / rigidity_weight_sum[:, None]
            ).unsqueeze(-1)

            track_gap = motion_features[..., 5:6].unsqueeze(1).expand(
                -1,
                detection_count,
                -1,
                -1,
            )
            track_visible_ratio = motion_features[..., 6:7].unsqueeze(1).expand(
                -1,
                detection_count,
                -1,
                -1,
            )
            uncertainty_expanded = track_uncertainty.unsqueeze(1).expand(
                -1,
                detection_count,
                -1,
                -1,
            )

            pair_features = torch.cat(
                [
                    track_expanded,
                    detection_expanded,
                    delta_last,
                    distance_last,
                    delta_predicted,
                    distance_predicted,
                    normalized_predicted_distance,
                    rigidity_residual,
                    track_gap,
                    track_visible_ratio,
                    uncertainty_expanded,
                ],
                dim=-1,
            )
            pair_logits = self.pair_scorer(pair_features).squeeze(-1)
            pair_logits = pair_logits.masked_fill(
                ~track_mask.unsqueeze(1),
                -1e9,
            )
        else:
            pair_logits = detection_embeddings.new_zeros(
                (batch_size, detection_count, 0)
            )

            distance_predicted = detections.new_zeros(
                (batch_size, detection_count, 0, 1)
            )
            rigidity_residual = detections.new_zeros(
                (batch_size, detection_count, 0, 1)
            )

        valid_track_count = track_mask.sum(dim=1, keepdim=True).clamp_min(1)
        global_track_embedding = (
            (track_embeddings * track_mask.unsqueeze(-1)).sum(dim=1)
            / valid_track_count.to(detections.dtype)
        )
        global_track_embedding = global_track_embedding.unsqueeze(1).expand(
            -1,
            detection_count,
            -1,
        )
        has_tracks = track_mask.any(dim=1, keepdim=True)
        has_tracks_per_detection = has_tracks.unsqueeze(1).expand(
            -1,
            detection_count,
            -1,
        )

        if track_count > 0:
            best_existing_logit = pair_logits.max(dim=-1, keepdim=True).values
            masked_predicted_distance = distance_predicted.squeeze(-1).masked_fill(
                ~track_mask.unsqueeze(1),
                float("inf"),
            )
            minimum_predicted_distance = masked_predicted_distance.min(
                dim=-1,
                keepdim=True,
            ).values
            masked_rigidity_residual = rigidity_residual.squeeze(-1).masked_fill(
                ~track_mask.unsqueeze(1),
                float("inf"),
            )
            minimum_rigidity_residual = masked_rigidity_residual.min(
                dim=-1,
                keepdim=True,
            ).values
            best_existing_logit = torch.where(
                has_tracks_per_detection,
                best_existing_logit,
                torch.zeros_like(best_existing_logit),
            )
            minimum_predicted_distance = torch.where(
                has_tracks_per_detection,
                minimum_predicted_distance,
                torch.zeros_like(minimum_predicted_distance),
            )
            minimum_rigidity_residual = torch.where(
                has_tracks_per_detection,
                minimum_rigidity_residual,
                torch.zeros_like(minimum_rigidity_residual),
            )
        else:
            best_existing_logit = detections.new_zeros(
                (batch_size, detection_count, 1)
            )
            minimum_predicted_distance = torch.zeros_like(
                best_existing_logit
            )
            minimum_rigidity_residual = torch.zeros_like(
                best_existing_logit
            )

        new_features = torch.cat(
            [
                detection_embeddings,
                global_track_embedding,
                best_existing_logit,
                minimum_predicted_distance,
                minimum_rigidity_residual,
                has_tracks_per_detection.to(detections.dtype),
                past_presence,
                future_presence,
            ],
            dim=-1,
        )
        new_logits = self.new_scorer(new_features) + self.new_logit_bias
        logits = torch.cat(
            [pair_logits, new_logits],
            dim=-1,
        )
        return logits.masked_fill(~detection_mask.unsqueeze(-1), -1e9)


def assign_ids_from_logits(
    logits,
    track_mask,
    next_track_id,
    detection_mask=None,
):
    """把单帧关联分数解码为一对一 ID 分配。

    logits 的最后一列是 NEW，之前的每一列对应 ``track_index + 1``。
    每个有效检测都会得到一个正 ID；同一个已有 ID 在一帧中最多使用一次。
    多个 NEW 检测会按照检测顺序获得互不相同的递增 ID。
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError(
            "离线一对一 ID 分配需要 scipy，请先安装 scipy"
        ) from error

    if torch.is_tensor(logits):
        logits_array = logits.detach().cpu().numpy()
    else:
        logits_array = np.asarray(logits)
    if torch.is_tensor(track_mask):
        track_mask_array = track_mask.detach().cpu().numpy()
    else:
        track_mask_array = np.asarray(track_mask)

    if logits_array.ndim != 2:
        raise ValueError("logits 必须是 [detection_count, track_count + 1]")
    detection_count, class_count = logits_array.shape
    if class_count < 1:
        raise ValueError("logits 至少需要包含 NEW 一列")

    track_count = class_count - 1
    track_mask_array = np.asarray(track_mask_array, dtype=bool)
    if track_mask_array.shape != (track_count,):
        raise ValueError(
            f"track_mask 形状应为 {(track_count,)}，"
            f"实际为 {track_mask_array.shape}"
        )

    if detection_mask is None:
        detection_mask_array = np.ones(detection_count, dtype=bool)
    else:
        if torch.is_tensor(detection_mask):
            detection_mask = detection_mask.detach().cpu().numpy()
        detection_mask_array = np.asarray(detection_mask, dtype=bool)
        if detection_mask_array.shape != (detection_count,):
            raise ValueError(
                f"detection_mask 形状应为 {(detection_count,)}，"
                f"实际为 {detection_mask_array.shape}"
            )

    next_track_id = int(next_track_id)
    if next_track_id <= 0:
        raise ValueError("next_track_id 必须是正整数")
    if not np.isfinite(logits_array[detection_mask_array]).all():
        raise ValueError("有效检测的 logits 中包含 NaN 或无穷值")

    assigned_ids = np.zeros(detection_count, dtype=np.int64)
    valid_detection_indices = np.flatnonzero(detection_mask_array)
    valid_detection_count = len(valid_detection_indices)
    if valid_detection_count == 0:
        return assigned_ids, next_track_id

    valid_logits = logits_array[valid_detection_indices].astype(
        np.float64,
        copy=False,
    )
    invalid_score = -1e12
    assignment_scores = np.full(
        (valid_detection_count, track_count + valid_detection_count),
        invalid_score,
        dtype=np.float64,
    )
    if track_count > 0:
        assignment_scores[:, :track_count] = valid_logits[:, :track_count]
        assignment_scores[:, :track_count][:, ~track_mask_array] = invalid_score

    new_scores = valid_logits[:, track_count]
    new_columns = track_count + np.arange(valid_detection_count)
    assignment_scores[np.arange(valid_detection_count), new_columns] = new_scores

    row_indices, column_indices = linear_sum_assignment(
        -assignment_scores,
    )
    selected_columns = np.full(valid_detection_count, -1, dtype=np.int64)
    selected_columns[row_indices] = column_indices

    used_track_ids = set(np.flatnonzero(track_mask_array).tolist())
    used_track_ids = {track_index + 1 for track_index in used_track_ids}
    for local_index, detection_index in enumerate(valid_detection_indices):
        selected_column = int(selected_columns[local_index])
        if selected_column < track_count:
            assigned_ids[detection_index] = selected_column + 1
            continue

        while next_track_id in used_track_ids:
            next_track_id += 1
        assigned_ids[detection_index] = next_track_id
        used_track_ids.add(next_track_id)
        next_track_id += 1

    return assigned_ids, next_track_id


def load_coordinate_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_version = int(checkpoint.get("architecture_version", 1))
    if checkpoint_version != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"模型权重架构版本为{checkpoint_version}，当前代码需要"
            f"版本{ARCHITECTURE_VERSION}，请重新训练模型"
        )
    model = CoordinateAssociationModel(
        max_tracks=checkpoint["max_tracks"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint