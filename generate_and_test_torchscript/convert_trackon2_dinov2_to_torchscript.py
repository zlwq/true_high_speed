import math
import os
import shutil
import subprocess
import sys
import types
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path.home() / "true_high_speed"
TRACKON_REPO = PROJECT_ROOT / "infering_code" / "third_party" / "track_on-track-on2"
TRACKON_ARCHIVE_URL = "https://github.com/gorkaydemir/track_on/archive/refs/heads/track-on2.zip"
DINOV2_DEFAULT_DIR = PROJECT_ROOT / "infering_code" / "third_party" / "dinov2-small"
DINOV2_CONFIG_URL = "https://huggingface.co/facebook/dinov2-small/resolve/main/config.json"
DINOV2_WEIGHT_URL = "https://huggingface.co/facebook/dinov2-small/resolve/main/model.safetensors"
CHECKPOINT_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "trackon2_dinov2_checkpoint.pt"
OUTPUT_PATH = PROJECT_ROOT / "model" / "trackon2_dinov2_pure_torchscript.pt"
CONFIG_PATH = TRACKON_REPO / "config" / "test_dinov2.yaml"

# 可选：如果已经手工下载 facebook/dinov2-small，请把环境变量 DINOV2_DIR
# 指向同时包含 config.json 和 model.safetensors 的目录。
DINOV2_DIR = os.environ.get("DINOV2_DIR", "").strip()

# 导出时使用的示例点数。导出后的点数维度仍会额外用不同点数进行验证。
EXAMPLE_POINT_COUNT = 2


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    batch_size = value.shape[0]
    num_heads = value.shape[2]
    head_dim = value.shape[3]
    num_queries = sampling_locations.shape[1]
    num_levels = sampling_locations.shape[3]
    num_points = sampling_locations.shape[4]

    sampling_grids = sampling_locations * 2.0 - 1.0
    sampled_levels = []
    level_start = 0

    for level in range(num_levels):
        height = spatial_shapes[level, 0]
        width = spatial_shapes[level, 1]
        level_length = height * width

        value_level = value[:, level_start : level_start + level_length]
        value_level = value_level.flatten(2).transpose(1, 2)
        value_level = value_level.reshape(
            batch_size * num_heads, head_dim, height, width
        )

        grid_level = sampling_grids[:, :, :, level]
        grid_level = grid_level.transpose(1, 2).flatten(0, 1)

        sampled_level = F.grid_sample(
            value_level,
            grid_level,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_levels.append(sampled_level)
        level_start = level_start + level_length

    weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )

    output = torch.stack(sampled_levels, dim=-2).flatten(-2)
    output = (output * weights).sum(-1)
    output = output.reshape(batch_size, num_heads * head_dim, num_queries)
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg=None,
        init_cfg=None,
        value_proj_ratio: float = 1.0,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims 必须能被 num_heads 整除")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        value_proj_size = int(embed_dims * value_proj_ratio)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)

        theta = torch.arange(self.num_heads, dtype=torch.float32)
        theta = theta * (2.0 * math.pi / self.num_heads)
        grid = torch.stack((theta.cos(), theta.sin()), dim=-1)
        grid = grid / grid.abs().amax(dim=-1, keepdim=True)
        grid = grid.view(self.num_heads, 1, 1, 2)
        grid = grid.repeat(1, self.num_levels, self.num_points, 1)
        for point_index in range(self.num_points):
            grid[:, :, point_index] *= point_index + 1
        self.sampling_offsets.bias.data.copy_(grid.reshape(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        batch_size = query.shape[0]
        num_queries = query.shape[1]
        num_values = value.shape[1]

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.reshape(batch_size, num_values, self.num_heads, -1)

        offsets = self.sampling_offsets(query).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        weights = self.attention_weights(query).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels * self.num_points,
        )
        weights = weights.softmax(dim=-1).reshape(
            batch_size,
            num_queries,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        if reference_points.shape[-1] == 2:
            normalizer = torch.stack(
                (spatial_shapes[..., 1], spatial_shapes[..., 0]), dim=-1
            )
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        else:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / float(self.num_points)
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, locations, weights
        )
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def install_mmcv_compatibility_module() -> None:
    mmcv_module = types.ModuleType("mmcv")
    ops_module = types.ModuleType("mmcv.ops")
    ops_module.MultiScaleDeformableAttention = MultiScaleDeformableAttention
    mmcv_module.ops = ops_module
    sys.modules["mmcv"] = mmcv_module
    sys.modules["mmcv.ops"] = ops_module


def download_url_to_file_compat(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(destination.name + ".part")
    if part_path.exists():
        part_path.unlink()

    proxy = ""
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name, "").strip()
        if value:
            proxy = value
            break

    try:
        if proxy.lower().startswith("socks://"):
            curl_proxy = "socks5h://" + proxy[len("socks://"):]
            print(f"检测到 SOCKS 代理，使用 curl 下载：{curl_proxy}")
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--progress-bar",
                    "--proxy",
                    curl_proxy,
                    "-o",
                    str(part_path),
                    url,
                ],
                check=True,
            )
        else:
            torch.hub.download_url_to_file(
                url,
                str(part_path),
                progress=True,
            )

        part_path.replace(destination)
    finally:
        if part_path.exists():
            part_path.unlink()


def ensure_dinov2_directory() -> Path:
    if DINOV2_DIR:
        local_dir = Path(DINOV2_DIR).expanduser().resolve()
    else:
        local_dir = DINOV2_DEFAULT_DIR

    config_path = local_dir / "config.json"
    weight_path = local_dir / "model.safetensors"

    if not config_path.is_file():
        print("未检测到 DINOv2-Small config.json，开始自动下载……")
        download_url_to_file_compat(DINOV2_CONFIG_URL, config_path)

    if not weight_path.is_file():
        print("未检测到 DINOv2-Small model.safetensors，开始自动下载……")
        download_url_to_file_compat(DINOV2_WEIGHT_URL, weight_path)

    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError(f"DINOv2-Small 下载不完整：{local_dir}")

    return local_dir


def redirect_dinov2_to_local_directory() -> None:
    local_dir = ensure_dinov2_directory()

    from transformers import AutoModel

    original_from_pretrained = AutoModel.from_pretrained

    def local_from_pretrained(cls, model_name, *args, **kwargs):
        if model_name == "facebook/dinov2-small":
            kwargs.pop("local_files_only", None)
            return original_from_pretrained(
                str(local_dir),
                *args,
                local_files_only=True,
                **kwargs,
            )
        return original_from_pretrained(model_name, *args, **kwargs)

    AutoModel.from_pretrained = classmethod(local_from_pretrained)
    print(f"使用本地 DINOv2-Small：{local_dir}")


class TrackOn2StreamingTorchScript(nn.Module):
    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.core = predictor.model
        self.delta_v = float(predictor.delta_v)
        self.model_height = int(self.core.input_size[0])
        self.model_width = int(self.core.input_size[1])
        self.memory_size = int(self.core.M)
        self.feature_dim = int(self.core.D)

    def _extract_query_features(
        self, fused_features: torch.Tensor, query_points: torch.Tensor
    ) -> torch.Tensor:
        feature_map = fused_features.reshape(
            1, self.core.Hf, self.core.Wf, self.feature_dim
        )
        feature_map = feature_map.permute(0, 3, 1, 2)

        normalized_points = query_points.clone()
        normalized_points[:, 0] = normalized_points[:, 0] / float(self.model_width)
        normalized_points[:, 1] = normalized_points[:, 1] / float(self.model_height)
        normalized_points = normalized_points * 2.0 - 1.0
        grid = normalized_points.unsqueeze(0).unsqueeze(2)

        query_features = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return query_features.squeeze(-1).squeeze(0).permute(1, 0).contiguous()

    def initialize(
        self, frame: torch.Tensor, query_points: torch.Tensor
    ):
        f4, f8, f16, f32, fused = self.core.extract_frame_features(frame)
        query_features = self._extract_query_features(fused, query_points)

        point_count = query_features.shape[0]
        point_memory = torch.zeros(
            point_count,
            self.memory_size,
            self.feature_dim,
            dtype=query_features.dtype,
            device=query_features.device,
        )
        temporal_mask = torch.ones(
            point_count,
            self.memory_size,
            dtype=torch.bool,
            device=query_features.device,
        )

        positions, visibility_logits, new_features = self.core.track_frame(
            query_features,
            temporal_mask,
            point_memory,
            (f4, f8, f16, f32, fused),
            self.model_height,
            self.model_width,
        )

        point_memory = torch.roll(point_memory, shifts=-1, dims=1)
        point_memory[:, -1] = new_features
        temporal_mask = torch.roll(temporal_mask, shifts=-1, dims=1)
        temporal_mask[:, -1] = False
        visibility = visibility_logits.sigmoid() >= self.delta_v

        return (
            positions,
            visibility,
            query_features,
            temporal_mask,
            point_memory,
        )

    def step(
        self,
        frame: torch.Tensor,
        query_features: torch.Tensor,
        temporal_mask: torch.Tensor,
        point_memory: torch.Tensor,
    ):
        f4, f8, f16, f32, fused = self.core.extract_frame_features(frame)
        positions, visibility_logits, new_features = self.core.track_frame(
            query_features,
            temporal_mask,
            point_memory,
            (f4, f8, f16, f32, fused),
            self.model_height,
            self.model_width,
        )

        point_memory = torch.roll(point_memory, shifts=-1, dims=1)
        point_memory[:, -1] = new_features
        temporal_mask = torch.roll(temporal_mask, shifts=-1, dims=1)
        temporal_mask[:, -1] = False
        visibility = visibility_logits.sigmoid() >= self.delta_v

        return positions, visibility, temporal_mask, point_memory


def ensure_trackon_repo() -> None:
    module_file = TRACKON_REPO / "model" / "trackon_predictor.py"

    if module_file.is_file() and CONFIG_PATH.is_file():
        return

    third_party_dir = TRACKON_REPO.parent
    archive_path = third_party_dir / "track-on2.zip.part"
    third_party_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        archive_path.unlink()
    if TRACKON_REPO.exists():
        shutil.rmtree(TRACKON_REPO)

    print("未检测到完整的 Track-On2 第三方代码，开始自动下载官方源码……")
    print(f"下载地址：{TRACKON_ARCHIVE_URL}")

    try:
        download_url_to_file_compat(
            TRACKON_ARCHIVE_URL,
            archive_path,
        )
        print("下载完成，正在解压……")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(third_party_dir)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    if not module_file.is_file() or not CONFIG_PATH.is_file():
        raise RuntimeError("没有找到完整的 Track-On2 官方源码")

    print(f"Track-On2 第三方代码已准备完成：{TRACKON_REPO}")


def check_required_files() -> None:
    required = [TRACKON_REPO, CONFIG_PATH, CHECKPOINT_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少以下文件或目录：\n" + "\n".join(missing))


def load_predictor(device: torch.device):
    install_mmcv_compatibility_module()
    redirect_dinov2_to_local_directory()

    sys.path.insert(0, str(TRACKON_REPO))
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        model_args = Namespace(**yaml.safe_load(file))

    from model.trackon_predictor import Predictor

    predictor = Predictor(
        model_args,
        checkpoint_path=str(CHECKPOINT_PATH),
        support_grid_size=0,
    )
    predictor = predictor.to(device).eval()
    return predictor


def export_torchscript() -> None:
    ensure_trackon_repo()
    check_required_files()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝回退到 CPU。")

    device = torch.device("cuda:0")
    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"Track-On2 权重：{CHECKPOINT_PATH}")

    predictor = load_predictor(device)
    wrapper = TrackOn2StreamingTorchScript(predictor).to(device).eval()

    height = wrapper.model_height
    width = wrapper.model_width
    memory_size = wrapper.memory_size
    feature_dim = wrapper.feature_dim

    frame = torch.randint(
        0, 256, (1, 3, height, width), device=device, dtype=torch.int32
    ).float()
    points = torch.tensor(
        [[width * 0.40, height * 0.45], [width * 0.65, height * 0.60]],
        device=device,
        dtype=torch.float32,
    )[:EXAMPLE_POINT_COUNT]

    query_features = torch.zeros(
        EXAMPLE_POINT_COUNT, feature_dim, device=device, dtype=torch.float32
    )
    temporal_mask = torch.ones(
        EXAMPLE_POINT_COUNT, memory_size, device=device, dtype=torch.bool
    )
    point_memory = torch.zeros(
        EXAMPLE_POINT_COUNT,
        memory_size,
        feature_dim,
        device=device,
        dtype=torch.float32,
    )

    print("开始追踪 initialize 和 step 两个 TorchScript 方法……")
    with torch.inference_mode():
        scripted = torch.jit.trace_module(
            wrapper,
            {
                "initialize": (frame, points),
                "step": (frame, query_features, temporal_mask, point_memory),
            },
            check_trace=False,
            strict=False,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(OUTPUT_PATH))
    print(f"已保存：{OUTPUT_PATH}")

    del scripted, wrapper, predictor
    torch.cuda.empty_cache()

    print("重新加载并用不同点数验证动态点维度……")
    loaded = torch.jit.load(str(OUTPUT_PATH), map_location=device).eval()
    test_points = torch.tensor(
        [
            [width * 0.25, height * 0.30],
            [width * 0.50, height * 0.50],
            [width * 0.75, height * 0.70],
        ],
        device=device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        positions, visible, q, mask, memory = loaded.initialize(frame, test_points)
        positions2, visible2, mask2, memory2 = loaded.step(
            frame, q, mask, memory
        )

    if positions.shape != (3, 2) or positions2.shape != (3, 2):
        raise RuntimeError(
            f"动态点数验证失败：{positions.shape}、{positions2.shape}"
        )
    if visible.shape != (3,) or visible2.shape != (3,):
        raise RuntimeError(
            f"可见性输出形状错误：{visible.shape}、{visible2.shape}"
        )
    if mask2.shape != (3, memory_size):
        raise RuntimeError(f"时间掩码形状错误：{mask2.shape}")
    if memory2.shape != (3, memory_size, feature_dim):
        raise RuntimeError(f"缓存形状错误：{memory2.shape}")

    print("转换成功。")
    print("输入帧：float32 [1, 3, 384, 512]，数值范围 0~255")
    print("点坐标：float32 [N, 2]，顺序为 (x, y)，基于 512×384 模型坐标")
    print("Qt/LibTorch 方法：initialize、step")


if __name__ == "__main__":
    export_torchscript()