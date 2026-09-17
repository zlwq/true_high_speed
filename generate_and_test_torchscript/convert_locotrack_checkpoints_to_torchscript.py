import gc
import shutil
import sys
import zipfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "model"
MODEL_RESOLUTION = 256
QUERY_CHUNK_SIZE = 64
TRACE_POINTS = 4
FRAME_PROFILES = (64, 32, 16, 8, 4, 2)
MODEL_SIZES = ("small", "base")

LOCO_REPO_URL = (
    "https://github.com/cvlab-kaist/locotrack/"
    "archive/refs/heads/main.zip"
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


class LocoTrackTorchScriptWrapper(torch.nn.Module):
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


def checkpoint_path(model_size: str) -> Path:
    return (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / f"locotrack_{model_size}.ckpt"
    )


def make_example_query():
    coordinates = torch.linspace(
        48.0,
        MODEL_RESOLUTION - 48.0,
        TRACE_POINTS,
        dtype=torch.float32,
        device="cuda",
    )
    query = torch.zeros(
        (1, TRACE_POINTS, 3),
        dtype=torch.float32,
        device="cuda",
    )
    query[0, :, 1] = coordinates
    query[0, :, 2] = torch.flip(coordinates, dims=(0,))
    return query


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


def convert_model_size(model_size: str, load_model):
    model = load_checkpoint_model(model_size, load_model)
    model = model.cuda().eval()
    wrapper = LocoTrackTorchScriptWrapper(model).cuda().eval()
    example_query = make_example_query()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    for frame_count in FRAME_PROFILES:
        example_video = torch.zeros(
            (
                1,
                frame_count,
                MODEL_RESOLUTION,
                MODEL_RESOLUTION,
                3,
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        output_path = OUTPUT_DIR / (
            f"locotrack_{model_size}_t{frame_count}_torchscript.pt"
        )

        print(
            f"开始转换 LocoTrack-{model_size.upper()} "
            f"固定 {frame_count} 帧模型……"
        )
        try:
            with torch.inference_mode():
                scripted = torch.jit.trace(
                    wrapper,
                    (example_video, example_query),
                    strict=False,
                    check_trace=False,
                )
            scripted.save(str(output_path))
            generated.append(output_path)
            print(f"已保存：{output_path}")
            del scripted
        except (torch.OutOfMemoryError, RuntimeError) as error:
            message = str(error).lower()
            if not (
                isinstance(error, torch.OutOfMemoryError)
                or ("cuda" in message and "out of memory" in message)
            ):
                raise
            print(
                f"转换 {frame_count} 帧档位时显存不足，"
                "跳过该档位并继续转换更小档位。"
            )

        del example_video
        gc.collect()
        torch.cuda.empty_cache()

    del example_query
    del wrapper
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not generated:
        raise RuntimeError(
            f"LocoTrack-{model_size.upper()} 所有档位均转换失败"
        )

    print(f"LocoTrack-{model_size.upper()} 已生成：")
    for path in generated:
        print(f"  {path.name}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，转换已停止")

    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"固定帧数档位：{FRAME_PROFILES}")
    print(f"转换示例点数：{TRACE_POINTS}，运行时支持 1~64 个点")

    locotrack_root = ensure_locotrack_source()
    sys.path.insert(0, str(locotrack_root))
    from models.locotrack_model import load_model

    for model_size in MODEL_SIZES:
        convert_model_size(model_size, load_model)

    print("Small 和 Base 的可用固定帧档位已全部转换完成。")


if __name__ == "__main__":
    main()
