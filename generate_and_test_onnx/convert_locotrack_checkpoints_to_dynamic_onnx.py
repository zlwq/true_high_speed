import gc
import shutil
import sys
import zipfile
from pathlib import Path

import torch

try:
    from torch.onnx._internal.torchscript_exporter import symbolic_helper
except ImportError:
    from torch.onnx import symbolic_helper

try:
    import onnx
except ImportError as error:
    raise RuntimeError(
        "请先安装：pip install onnx onnxscript"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "model"
MODEL_RESOLUTION = 256
QUERY_CHUNK_SIZE = 64
EXAMPLE_FRAMES = 2
EXAMPLE_POINTS = 4
MAX_DYNAMIC_FRAMES = 256
MAX_DYNAMIC_POINTS = 64
MODEL_SIZES = ("small", "base")
OPSET_VERSION = 20

LOCO_REPO_URL = (
    "https://github.com/cvlab-kaist/locotrack/"
    "archive/refs/heads/main.zip"
)


@symbolic_helper.parse_args("v", "v", "i", "i", "b")
def grid_sampler_onnx20(
    graph,
    input_tensor,
    grid,
    mode_enum,
    padding_mode_enum,
    align_corners,
):
    modes = {
        0: "linear",
        1: "nearest",
        2: "cubic",
    }
    padding_modes = {
        0: "zeros",
        1: "border",
        2: "reflection",
    }
    if mode_enum not in modes:
        raise RuntimeError(f"不支持的 GridSample 模式：{mode_enum}")
    if padding_mode_enum not in padding_modes:
        raise RuntimeError(
            f"不支持的 GridSample padding 模式：{padding_mode_enum}"
        )
    return graph.op(
        "GridSample",
        input_tensor,
        grid,
        mode_s=modes[mode_enum],
        padding_mode_s=padding_modes[padding_mode_enum],
        align_corners_i=int(align_corners),
    )


def ensure_locotrack_source() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir / "third_party" / "locotrack-main"
    module_file = (
        repo_dir
        / "locotrack_pytorch"
        / "models"
        / "locotrack_model.py"
    )

    if not module_file.is_file():
        original_repo_dir = (
            PROJECT_ROOT
            / "infering_code"
            / "third_party"
            / "locotrack-main"
        )
        original_module = (
            original_repo_dir
            / "locotrack_pytorch"
            / "models"
            / "locotrack_model.py"
        )
        if original_module.is_file():
            return original_repo_dir / "locotrack_pytorch"

        third_party_dir = script_dir / "third_party"
        archive_path = third_party_dir / "locotrack-main.zip.part"
        third_party_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.exists():
            archive_path.unlink()
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        print("正在下载 LocoTrack 官方源码……")
        try:
            torch.hub.download_url_to_file(
                LOCO_REPO_URL,
                str(archive_path),
                progress=True,
            )
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(third_party_dir)
        finally:
            if archive_path.exists():
                archive_path.unlink()

    if not module_file.is_file():
        raise RuntimeError("没有找到 LocoTrack 官方 PyTorch 源码")

    return repo_dir / "locotrack_pytorch"


class LocoTrackOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video_uint8, query_points):
        video = video_uint8.to(dtype=torch.float32) / 255.0 * 2.0 - 1.0
        output = self.model(
            video,
            query_points,
            query_chunk_size=QUERY_CHUNK_SIZE,
        )
        tracks = output["tracks"]
        occlusion = output["occlusion"]
        expected_dist = output["expected_dist"]
        occlusion_probability = torch.sigmoid(occlusion)
        occlusion_probability = 1.0 - (
            (1.0 - occlusion_probability)
            * (1.0 - torch.sigmoid(expected_dist))
        )
        return tracks, occlusion_probability > 0.5


def convert_grid_coordinates_export_safe(
    coords,
    input_grid_size,
    output_grid_size,
    coordinate_format="xy",
):
    if coordinate_format == "xy":
        x = coords[..., 0] * output_grid_size[0] / input_grid_size[0]
        y = coords[..., 1] * output_grid_size[1] / input_grid_size[1]
        return torch.stack((x, y), dim=-1)

    if coordinate_format == "tyx":
        time = coords[..., 0]
        y = coords[..., 1] * output_grid_size[1] / input_grid_size[1]
        x = coords[..., 2] * output_grid_size[2] / input_grid_size[2]
        return torch.stack((time, y, x), dim=-1)

    raise ValueError("coordinate_format 必须是 xy 或 tyx")


def checkpoint_path(model_size: str) -> Path:
    return (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / f"locotrack_{model_size}.ckpt"
    )


def onnx_path(model_size: str) -> Path:
    return OUTPUT_DIR / f"locotrack_{model_size}_dynamic.onnx"


def require_dynamic_axis(value_info, axis: int, expected_name: str):
    dimensions = value_info.type.tensor_type.shape.dim
    if axis >= len(dimensions):
        raise RuntimeError(
            f"{value_info.name} 缺少第 {axis} 维，动态 ONNX 导出失败"
        )
    if dimensions[axis].dim_param != expected_name:
        actual = dimensions[axis].dim_param
        if not actual and dimensions[axis].HasField("dim_value"):
            actual = str(dimensions[axis].dim_value)
        raise RuntimeError(
            f"{value_info.name} 第 {axis} 维仍是 {actual!r}，"
            f"没有成为动态轴 {expected_name!r}，拒绝保留该 ONNX"
        )


def check_dynamic_onnx(onnx_model):
    values = {
        value.name: value
        for value in (
            list(onnx_model.graph.input)
            + list(onnx_model.graph.output)
        )
    }
    requirements = {
        "video": ((1, "frame_count"),),
        "query_points": ((1, "point_count"),),
        "tracks": ((1, "point_count"), (2, "frame_count")),
        "occluded": ((1, "point_count"), (2, "frame_count")),
    }
    for name, axes in requirements.items():
        if name not in values:
            raise RuntimeError(f"ONNX 中没有找到输入或输出：{name}")
        for axis, expected_name in axes:
            require_dynamic_axis(values[name], axis, expected_name)


def load_checkpoint_model(model_size: str, load_model):
    checkpoint = checkpoint_path(model_size)
    if checkpoint.is_file():
        print(f"使用已有权重：{checkpoint}")
        return load_model(
            ckpt_path=str(checkpoint),
            model_size=model_size,
        )

    print(f"没有找到 {checkpoint.name}，现在自动下载。")
    return load_model(model_size=model_size)


def make_query_points(point_count: int):
    coordinates = torch.linspace(
        32.0,
        MODEL_RESOLUTION - 32.0,
        point_count,
        dtype=torch.float32,
        device="cuda",
    )
    query = torch.zeros(
        (1, point_count, 3),
        dtype=torch.float32,
        device="cuda",
    )
    query[0, :, 1] = coordinates
    query[0, :, 2] = torch.flip(coordinates, dims=(0,))
    return query


def make_video(frame_count: int, seed: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randint(
        0,
        256,
        (
            1,
            frame_count,
            MODEL_RESOLUTION,
            MODEL_RESOLUTION,
            3,
        ),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )


def export_one(model_size: str, load_model):
    model = load_checkpoint_model(model_size, load_model)
    model = model.cuda().eval()
    original_feature_chunk_size = model.feature_extractor_chunk_size
    model.feature_extractor_chunk_size = 0
    wrapper = LocoTrackOnnxWrapper(model).cuda().eval()

    print(
        "导出时关闭模型内部的 Python 帧分块循环；原值为："
        f"{original_feature_chunk_size}。推理时仍由外层代码自动分段。"
    )

    example_video = make_video(EXAMPLE_FRAMES, 20260831)
    example_query = make_query_points(EXAMPLE_POINTS)
    gc.collect()
    torch.cuda.empty_cache()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = onnx_path(model_size)
    print(
        f"开始导出 LocoTrack-{model_size.upper()} 动态 ONNX："
        f"T=2~{MAX_DYNAMIC_FRAMES}，N=1~{MAX_DYNAMIC_POINTS}"
    )
    print(
        f"本次仅用 {EXAMPLE_FRAMES} 帧、{EXAMPLE_POINTS} 个点描图；"
        "该样例尺寸不限制导出后的动态输入长度。"
    )

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example_video, example_query),
            str(destination),
            input_names=["video", "query_points"],
            output_names=["tracks", "occluded"],
            opset_version=OPSET_VERSION,
            dynamo=False,
            dynamic_axes={
                "video": {1: "frame_count"},
                "query_points": {1: "point_count"},
                "tracks": {
                    1: "point_count",
                    2: "frame_count",
                },
                "occluded": {
                    1: "point_count",
                    2: "frame_count",
                },
            },
            external_data=False,
        )

    onnx_model = onnx.load(str(destination))
    onnx.checker.check_model(onnx_model)
    check_dynamic_onnx(onnx_model)
    del onnx_model
    print(f"ONNX 结构与动态轴检查通过：{destination}")

    del example_video
    del example_query
    gc.collect()
    torch.cuda.empty_cache()

    del wrapper
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，导出已停止")

    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    torch.onnx.register_custom_op_symbolic(
        "aten::grid_sampler",
        grid_sampler_onnx20,
        OPSET_VERSION,
    )
    print("已启用 ONNX opset 20 五维 GridSample 导出规则")
    locotrack_root = ensure_locotrack_source()
    sys.path.insert(0, str(locotrack_root))
    from models import utils as locotrack_utils
    from models.locotrack_model import load_model

    locotrack_utils.convert_grid_coordinates = (
        convert_grid_coordinates_export_safe
    )

    for model_size in MODEL_SIZES:
        export_one(model_size, load_model)

    print("Small 和 Base 动态 ONNX 导出完成。")


if __name__ == "__main__":
    main()