from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from tapnet.tapnext import tapnext_lru_modules, tapnext_torch
from tapnet.tapnextpp.votsp2026.model import TAPNextPP


SCRIPT_DIR = Path(__file__).resolve().parent
# 文件放在项目根目录或项目下的torchscript_model目录都能正确识别。
PROJECT_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "assets").is_dir() else SCRIPT_DIR.parent
)
DEFAULT_CHECKPOINT = Path(
    os.path.expanduser("~/.cache/tapnextpp/tapnextpp_512.ckpt")
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "torchscript"

INPUT_RESOLUTION = 512
MODEL_COORDINATE_SIZE = 256.0
CACHE_LAYER_COUNT = 12


class InferenceSqrt:
    """替代训练专用SqrtBoundDerivative的纯推理实现。"""

    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        # SqrtBoundDerivative的forward本来就是torch.sqrt；区别只在backward。
        return torch.sqrt(x)


def install_scriptable_inference_sqrt() -> None:
    """移除TorchScript无法保存的Python autograd.Function调用。"""

    sqrt_class = getattr(
        tapnext_lru_modules,
        "SqrtBoundDerivative",
        None,
    )
    if sqrt_class is None:
        raise RuntimeError("当前tapnet版本中找不到SqrtBoundDerivative")
    tapnext_lru_modules.SqrtBoundDerivative = InferenceSqrt
    print("已将训练专用SqrtBoundDerivative替换为等价torch.sqrt推理实现")


def _next_step_tensor(step, video: torch.Tensor) -> torch.Tensor:
    if isinstance(step, torch.Tensor):
        return step.to(dtype=torch.int64)
    # 使用输入Tensor创建，避免在Trace图中写死cuda:0设备。
    return video.new_full((), int(step), dtype=torch.int64)


class TAPNextStableTorchScriptWrapper(nn.Module):
    """面向LibTorch的逐帧TAPNext++纯Tensor接口。

    initialize()负责第一帧；step()负责之后的递归帧。两个方法共享同一份权重。
    """

    def __init__(self, model: nn.Module, use_cuda_autocast: bool) -> None:
        super().__init__()
        self.model = model
        self.use_cuda_autocast = use_cuda_autocast

    def _flatten_outputs(self, outputs, video: torch.Tensor):
        tracks, track_logits, visible_logits, new_state = outputs
        flat = [
            tracks,
            track_logits,
            visible_logits,
            _next_step_tensor(new_state.step, video),
            new_state.query_points,
        ]
        for cache in new_state.hidden_state:
            flat.extend([cache.rg_lru_state, cache.conv1d_state])
        return tuple(flat)

    def initialize(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
    ):
        if self.use_cuda_autocast:
            # cache_enabled=False很重要：Trace需要记录显式类型转换，不能把
            # autocast临时缓存的Half权重误当成常量塞进图中。
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                cache_enabled=False,
            ):
                outputs = self.model(
                    video=video,
                    query_points=query_points,
                )
        else:
            outputs = self.model(
                video=video,
                query_points=query_points,
            )
        return self._flatten_outputs(outputs, video)

    def step(
        self,
        video: torch.Tensor,
        step: torch.Tensor,
        query_points: torch.Tensor,
        cache_00_rg_lru: torch.Tensor,
        cache_00_conv1d: torch.Tensor,
        cache_01_rg_lru: torch.Tensor,
        cache_01_conv1d: torch.Tensor,
        cache_02_rg_lru: torch.Tensor,
        cache_02_conv1d: torch.Tensor,
        cache_03_rg_lru: torch.Tensor,
        cache_03_conv1d: torch.Tensor,
        cache_04_rg_lru: torch.Tensor,
        cache_04_conv1d: torch.Tensor,
        cache_05_rg_lru: torch.Tensor,
        cache_05_conv1d: torch.Tensor,
        cache_06_rg_lru: torch.Tensor,
        cache_06_conv1d: torch.Tensor,
        cache_07_rg_lru: torch.Tensor,
        cache_07_conv1d: torch.Tensor,
        cache_08_rg_lru: torch.Tensor,
        cache_08_conv1d: torch.Tensor,
        cache_09_rg_lru: torch.Tensor,
        cache_09_conv1d: torch.Tensor,
        cache_10_rg_lru: torch.Tensor,
        cache_10_conv1d: torch.Tensor,
        cache_11_rg_lru: torch.Tensor,
        cache_11_conv1d: torch.Tensor,
    ):
        hidden_state = [
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_00_rg_lru,
                conv1d_state=cache_00_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_01_rg_lru,
                conv1d_state=cache_01_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_02_rg_lru,
                conv1d_state=cache_02_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_03_rg_lru,
                conv1d_state=cache_03_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_04_rg_lru,
                conv1d_state=cache_04_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_05_rg_lru,
                conv1d_state=cache_05_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_06_rg_lru,
                conv1d_state=cache_06_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_07_rg_lru,
                conv1d_state=cache_07_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_08_rg_lru,
                conv1d_state=cache_08_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_09_rg_lru,
                conv1d_state=cache_09_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_10_rg_lru,
                conv1d_state=cache_10_conv1d,
            ),
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=cache_11_rg_lru,
                conv1d_state=cache_11_conv1d,
            ),
        ]
        state = tapnext_torch.TAPNextTrackingState(
            step=step,
            query_points=query_points,
            hidden_state=hidden_state,
        )
        if self.use_cuda_autocast:
            with torch.amp.autocast(
                "cuda",
                dtype=torch.float16,
                cache_enabled=False,
            ):
                outputs = self.model(video=video, state=state)
        else:
            outputs = self.model(video=video, state=state)
        return self._flatten_outputs(outputs, video)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把TAPNext++ Stable 512导出为Windows LibTorch可加载的TorchScript"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"默认：{DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TorchScript输出路径；默认写入model/torchscript",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=4,
        help="Trace与验证使用的跟踪点数，默认4",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=("fp16", "fp32"),
        default="fp16",
        help=(
            "fp16表示FP32权重+CUDA autocast混合精度，与官方track_frame一致；"
            "CPU只能使用fp32"
        ),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA，但torch.cuda.is_available()为False")
    return torch.device(name)


def load_model(
    checkpoint: Path,
    device: torch.device,
    precision: str,
) -> nn.Module:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到checkpoint：{checkpoint}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16导出需要CUDA；CPU导出请使用--precision fp32")

    kwargs = {
        "device": device,
        "input_resolution": INPUT_RESOLUTION,
    }
    parameters = inspect.signature(TAPNextPP.from_checkpoint).parameters
    if "half_precision" in parameters:
        # 不能对整模调用half()：官方prediction_heads会主动执行x.float()，
        # 若Linear权重被转成Half就会出现Float/Half dtype不一致。
        kwargs["half_precision"] = False
    if "compile_model" in parameters:
        kwargs["compile_model"] = False

    high_level_model = TAPNextPP.from_checkpoint(str(checkpoint), **kwargs)
    inner = high_level_model._model.eval()
    if precision == "fp16" and not next(inner.parameters()).is_cuda:
        raise RuntimeError("模型没有加载到CUDA")
    if next(inner.parameters()).dtype != torch.float32:
        inner = inner.float()
    return inner


def state_names() -> list[str]:
    names = ["step", "query_points"]
    for layer in range(CACHE_LAYER_COUNT):
        names.extend(
            [
                f"cache_{layer:02d}_rg_lru",
                f"cache_{layer:02d}_conv1d",
            ]
        )
    return names


def output_names() -> list[str]:
    return [
        "tracks",
        "track_logits",
        "visible_logits",
        *[f"next_{name}" for name in state_names()],
    ]


def state_inputs(outputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(outputs[3:])


def primary_cpu(outputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.detach().float().cpu().clone() for tensor in outputs[:3]]


def assert_primary_close(
    actual: Sequence[torch.Tensor],
    expected: Sequence[torch.Tensor],
    precision: str,
    frame_index: int,
) -> None:
    rtol, atol = (5e-3, 5e-3) if precision == "fp16" else (1e-4, 1e-5)
    for name, actual_tensor, expected_tensor in zip(
        output_names()[:3], actual[:3], expected
    ):
        torch.testing.assert_close(
            actual_tensor.detach().float().cpu(),
            expected_tensor,
            rtol=rtol,
            atol=atol,
            msg=lambda message, n=name, f=frame_index: (
                f"第{f}帧输出{n}不一致：{message}"
            ),
        )


def assert_reference_sequences_close(
    actual_references,
    expected_references,
    precision: str,
) -> None:
    if len(actual_references) != len(expected_references):
        raise AssertionError("参考序列帧数不一致")
    for frame_index, (actual, expected) in enumerate(
        zip(actual_references, expected_references)
    ):
        assert_primary_close(actual, expected, precision, frame_index)


def make_examples(
    point_count: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if point_count <= 0:
        raise ValueError("--points必须大于0")
    shape = (1, 1, INPUT_RESOLUTION, INPUT_RESOLUTION, 3)
    frames = [
        torch.zeros(shape, dtype=dtype, device=device),
        torch.full(shape, 0.125, dtype=dtype, device=device),
        torch.full(shape, -0.25, dtype=dtype, device=device),
    ]
    queries = torch.zeros(
        (1, point_count, 3),
        dtype=dtype,
        device=device,
    )
    queries[..., 1] = MODEL_COORDINATE_SIZE * 0.45
    queries[..., 2] = MODEL_COORDINATE_SIZE * 0.55
    return frames, queries


def build_references(wrapper, frames, queries):
    references = []
    with torch.inference_mode():
        outputs = wrapper.initialize(frames[0], queries)
        references.append(primary_cpu(outputs))
        for frame in frames[1:]:
            outputs = wrapper.step(frame, *state_inputs(outputs))
            references.append(primary_cpu(outputs))
    state_example = state_inputs(outputs)
    state_manifest = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for name, tensor in zip(state_names(), state_example)
    ]
    return references, state_example, state_manifest


def trace_and_save(
    wrapper: TAPNextStableTorchScriptWrapper,
    frames,
    queries,
    state_example,
    output_path: Path,
) -> None:
    print("开始Trace initialize与step；两种方法共享同一份模型权重……")
    with torch.inference_mode():
        traced = torch.jit.trace_module(
            wrapper,
            {
                "initialize": (frames[0], queries),
                "step": (frames[1], *state_example),
            },
            check_trace=False,
            strict=False,
        )
    traced.eval()
    for method_name in ("initialize", "step"):
        graph_text = str(getattr(traced, method_name).graph)
        if "prim::PythonOp" in graph_text:
            raise RuntimeError(
                f"{method_name}图中仍存在prim::PythonOp，无法交给Windows LibTorch"
            )
    print("TorchScript图检查通过：没有残留PythonOp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path))


def validate_saved(
    output_path: Path,
    device: torch.device,
    frames,
    queries,
    references,
    precision: str,
) -> None:
    print("重新加载TorchScript并验证三帧递归结果……")
    module = torch.jit.load(str(output_path), map_location=device).eval()
    with torch.inference_mode():
        outputs = module.initialize(frames[0], queries)
        assert_primary_close(outputs, references[0], precision, 0)
        for frame_index, frame in enumerate(frames[1:], start=1):
            outputs = module.step(
                frame,
                *state_inputs(outputs),
            )
            assert_primary_close(
                outputs,
                references[frame_index],
                precision,
                frame_index,
            )
    print("原始PyTorch / 已保存TorchScript三帧递归对齐通过")


def write_manifest(
    output_path: Path,
    point_count: int,
    precision: str,
    state_manifest,
) -> Path:
    manifest = {
        "model": "TAPNext++ Stable 512",
        "format": "TorchScript",
        "windows_runtime": "LibTorch CUDA, same PyTorch major/minor recommended",
        "compute_precision": (
            "cuda_fp16_autocast" if precision == "fp16" else "fp32"
        ),
        "model_weight_dtype": "float32",
        "point_count_used_for_trace": point_count,
        "point_count_note": (
            "Treat this artifact as fixed-Q unless separately validated with other Q values."
        ),
        "input_resolution": INPUT_RESOLUTION,
        "methods": {
            "initialize": {
                "inputs": ["video", "query_points"],
                "outputs": output_names(),
                "usage": "first frame only",
            },
            "step": {
                "inputs": ["video", *state_names()],
                "outputs": output_names(),
                "usage": "all subsequent frames",
            },
        },
        "video": {
            "shape": [1, 1, 512, 512, 3],
            "layout": "NTHWC RGB",
            "dtype": "float32",
            "range": "[-1, 1]",
        },
        "query_points": {
            "shape": [1, point_count, 3],
            "layout": "[time, y, x]",
            "coordinate_space": "256x256 model space",
            "dtype": "float32",
        },
        "state": state_manifest,
        "recurrence": (
            "Call initialize once, then feed outputs[3:] into step inputs[1:] "
            "for every following frame. Keep all tensors on CUDA in C++."
        ),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    precision = args.precision
    # 与官方track_frame一致：预处理输入与权重保持FP32，模型内部由
    # CUDA autocast选择适合FP16执行的算子。
    dtype = torch.float32
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / (
            f"tapnextpp_stable_512_q{args.points}_{precision}_ts.pt"
        )
    output_path = output_path.expanduser().resolve()

    print("checkpoint：", args.checkpoint.expanduser().resolve())
    print("输出：", output_path)
    print("设备：", device)
    print("精度：", precision)
    print("固定验证点数：", args.points)

    inner_model = load_model(
        args.checkpoint.expanduser().resolve(),
        device,
        precision,
    )
    wrapper = TAPNextStableTorchScriptWrapper(
        inner_model,
        use_cuda_autocast=precision == "fp16",
    ).eval()
    frames, queries = make_examples(args.points, device, dtype)

    print("生成未修改原始模型的三帧递归参考结果……")
    references, original_state_example, _ = build_references(
        wrapper,
        frames,
        queries,
    )
    del original_state_example
    if device.type == "cuda":
        torch.cuda.empty_cache()

    install_scriptable_inference_sqrt()
    print("验证纯torch.sqrt推理实现与原模型等价……")
    patched_references, state_example, state_manifest = build_references(
        wrapper,
        frames,
        queries,
    )
    assert_reference_sequences_close(
        patched_references,
        references,
        precision,
    )
    del patched_references
    print("原始SqrtBoundDerivative / 纯torch.sqrt三帧递归对齐通过")

    trace_and_save(
        wrapper,
        frames,
        queries,
        state_example,
        output_path,
    )

    # 释放Python模型后重新加载，防止同时保留两份大权重造成显存不足。
    del state_example
    del wrapper
    del inner_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validate_saved(
        output_path,
        device,
        frames,
        queries,
        references,
        precision,
    )
    manifest_path = write_manifest(
        output_path,
        args.points,
        precision,
        state_manifest,
    )
    print("TorchScript导出完成：", output_path)
    print("C++接口说明：", manifest_path)


if __name__ == "__main__":
    main()