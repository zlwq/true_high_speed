from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import os
import platform
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from tapnet.tapnext import tapnext_lru_modules, tapnext_torch
from tapnet.tapnextpp.votsp2026.model import TAPNextPP


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "assets").is_dir() else SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = Path(os.path.expanduser("~/.cache/tapnextpp/tapnextpp_512.ckpt"))
DEFAULT_TORCHSCRIPT = (
    PROJECT_ROOT
    / "model"
    / "torchscript"
    / "tapnextpp_stable_512_q4_fp16_ts.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tapnextpp_perf_profile"

INPUT_RESOLUTION = 512
MODEL_COORDINATE_SIZE = 256.0
CACHE_LAYER_COUNT = 12


class InferenceSqrt:
    @staticmethod
    def apply(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)


def install_inference_sqrt() -> None:
    """与导出脚本一致，低层Eager对照也移除训练专用autograd.Function。"""
    if not hasattr(tapnext_lru_modules, "SqrtBoundDerivative"):
        raise RuntimeError("当前tapnet版本中找不到SqrtBoundDerivative")
    tapnext_lru_modules.SqrtBoundDerivative = InferenceSqrt


class EagerTensorWrapper(nn.Module):
    """与导出包装器等价，但允许切换 autocast 权重缓存。"""

    def __init__(self, model: nn.Module, cache_enabled: bool) -> None:
        super().__init__()
        self.model = model
        self.cache_enabled = cache_enabled

    @staticmethod
    def _flatten(outputs: Any, video: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tracks, track_logits, visible_logits, new_state = outputs
        if isinstance(new_state.step, torch.Tensor):
            step = new_state.step.to(dtype=torch.int64)
        else:
            step = video.new_full((), int(new_state.step), dtype=torch.int64)
        flat = [tracks, track_logits, visible_logits, step, new_state.query_points]
        for cache in new_state.hidden_state:
            flat.extend((cache.rg_lru_state, cache.conv1d_state))
        return tuple(flat)

    def initialize(
        self, video: torch.Tensor, query_points: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, cache_enabled=self.cache_enabled
        ):
            outputs = self.model(video=video, query_points=query_points)
        return self._flatten(outputs, video)

    def step(
        self,
        video: torch.Tensor,
        step: torch.Tensor,
        query_points: torch.Tensor,
        *flat_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(flat_cache) != CACHE_LAYER_COUNT * 2:
            raise ValueError(f"期望24个缓存Tensor，实际{len(flat_cache)}个")
        hidden_state = [
            tapnext_lru_modules.RecurrentBlockCache(
                rg_lru_state=flat_cache[2 * layer],
                conv1d_state=flat_cache[2 * layer + 1],
            )
            for layer in range(CACHE_LAYER_COUNT)
        ]
        state = tapnext_torch.TAPNextTrackingState(
            step=step,
            query_points=query_points,
            hidden_state=hidden_state,
        )
        with torch.amp.autocast(
            "cuda", dtype=torch.float16, cache_enabled=self.cache_enabled
        ):
            outputs = self.model(video=video, state=state)
        return self._flatten(outputs, video)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "同口径比较TAPNext++ Stable 512官方CKPT、低层Eager与TorchScript，"
            "并生成CUDA Event统计及torch.profiler trace"
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torchscript", type=Path, default=DEFAULT_TORCHSCRIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="跳过官方默认from_checkpoint配置的低层同口径基准",
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help="跳过低层Eager autocast cache开/关基准",
    )
    parser.add_argument(
        "--skip-torchscript",
        action="store_true",
        help="跳过TorchScript基准",
    )
    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="只跑CUDA Event，不生成profiler trace",
    )
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="不对加载后的TorchScript执行torch.jit.freeze",
    )
    args = parser.parse_args()
    for name in ("points", "warmup", "iterations", "profile_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')}必须大于0")
    if args.profile_warmup < 0:
        parser.error("--profile-warmup不能小于0")
    return args


def synchronize_and_collect() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def make_inputs(point_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # 固定一张非退化RGB图，三种后端反复使用相同内容，排除视频解码和绘制。
    frame_rgb = rng.integers(
        0, 256, size=(INPUT_RESOLUTION, INPUT_RESOLUTION, 3), dtype=np.uint8
    )
    margin = INPUT_RESOLUTION * 0.1
    query_xy = rng.uniform(
        margin,
        INPUT_RESOLUTION - margin,
        size=(point_count, 2),
    ).astype(np.float32)
    return frame_rgb, query_xy


def make_low_level_inputs(
    frame_rgb: np.ndarray,
    query_xy: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_np = np.ascontiguousarray(frame_rgb.astype(np.float32) / 127.5 - 1.0)
    video = torch.from_numpy(video_np).unsqueeze(0).unsqueeze(0).to(device)
    queries_np = np.zeros((1, len(query_xy), 3), dtype=np.float32)
    queries_np[0, :, 1] = query_xy[:, 1] / INPUT_RESOLUTION * MODEL_COORDINATE_SIZE
    queries_np[0, :, 2] = query_xy[:, 0] / INPUT_RESOLUTION * MODEL_COORDINATE_SIZE
    queries = torch.from_numpy(queries_np).to(device)
    return video, queries


def dtype_inventory(module: nn.Module) -> dict[str, Any]:
    inventory: dict[str, Counter[str]] = {
        "parameter_tensors": Counter(),
        "parameter_numel": Counter(),
        "buffer_tensors": Counter(),
        "buffer_numel": Counter(),
    }
    for tensor in module.parameters():
        key = str(tensor.dtype).removeprefix("torch.")
        inventory["parameter_tensors"][key] += 1
        inventory["parameter_numel"][key] += tensor.numel()
    for tensor in module.buffers():
        key = str(tensor.dtype).removeprefix("torch.")
        inventory["buffer_tensors"][key] += 1
        inventory["buffer_numel"][key] += tensor.numel()
    return {name: dict(counts) for name, counts in inventory.items()}


def module_dtype_inventory(module: nn.Module) -> dict[str, Any]:
    """按模块类型统计直接持有参数的dtype，便于找选择性FP16。"""
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for submodule in module.modules():
        module_type = type(submodule).__name__
        for tensor in submodule.parameters(recurse=False):
            key = str(tensor.dtype).removeprefix("torch.")
            result[module_type][key] += tensor.numel()
    return {module_type: dict(counts) for module_type, counts in sorted(result.items())}


def stats_from_ms(samples_ms: Sequence[float], wall_ms: float) -> dict[str, float]:
    ordered = sorted(samples_ms)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    mean_ms = statistics.fmean(samples_ms)
    return {
        "event_mean_ms": mean_ms,
        "event_median_ms": statistics.median(samples_ms),
        "event_p95_ms": percentile(0.95),
        "event_min_ms": min(samples_ms),
        "event_max_ms": max(samples_ms),
        "event_fps": 1000.0 / mean_ms,
        "wall_mean_ms": wall_ms,
        "wall_fps": 1000.0 / wall_ms,
    }


State = Any
InitFn = Callable[[], State]
StepFn = Callable[[State], State]


def cuda_event_benchmark(
    init_fn: InitFn,
    step_fn: StepFn,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float], State]:
    state = init_fn()
    for _ in range(warmup):
        state = step_fn(state)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    wall_started = time.perf_counter()
    for index in range(iterations):
        starts[index].record()
        state = step_fn(state)
        ends[index].record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000.0 / iterations
    samples_ms = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return stats_from_ms(samples_ms, wall_ms), state


def event_device_time_us(event: Any) -> float:
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def event_total_device_time_us(event: Any) -> float:
    for attribute in ("device_time_total", "cuda_time_total"):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_steps(
    name: str,
    init_fn: InitFn,
    step_fn: StepFn,
    warmup: int,
    steps: int,
    output_dir: Path,
) -> dict[str, Any]:
    state = init_fn()
    for _ in range(warmup):
        state = step_fn(state)
    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        for _ in range(steps):
            state = step_fn(state)
            profiler.step()
    torch.cuda.synchronize()

    trace_path = output_dir / f"{name}_trace.json"
    table_path = output_dir / f"{name}_operators.txt"
    csv_path = output_dir / f"{name}_operators.csv"
    profiler.export_chrome_trace(str(trace_path))
    table_path.write_text(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=100
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "operator": event.key,
                "count": int(event.count),
                "self_cpu_us": float(event.self_cpu_time_total),
                "cpu_total_us": float(event.cpu_time_total),
                "self_device_us": event_device_time_us(event),
                "device_total_us": event_total_device_time_us(event),
                "input_shapes": str(event.input_shapes),
            }
        )
    rows.sort(key=lambda row: row["self_device_us"], reverse=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["operator"])
        writer.writeheader()
        writer.writerows(rows)

    cast_pattern = re.compile(r"autocast|\bto\b|_to_copy|copy_", re.IGNORECASE)
    attention_pattern = re.compile(
        r"attention|scaled_dot|flash|efficient|softmax|einsum|matmul|\bbmm\b|\bmm\b",
        re.IGNORECASE,
    )
    cast_rows = [row for row in rows if cast_pattern.search(row["operator"])]
    attention_rows = [row for row in rows if attention_pattern.search(row["operator"])]
    return {
        "trace": str(trace_path),
        "operator_table": str(table_path),
        "operator_csv": str(csv_path),
        "total_operator_calls": sum(row["count"] for row in rows),
        "cast_like_calls": sum(row["count"] for row in cast_rows),
        "cast_like_self_device_ms": sum(row["self_device_us"] for row in cast_rows)
        / 1000.0,
        "attention_like_calls": sum(row["count"] for row in attention_rows),
        "attention_like_self_device_ms": sum(
            row["self_device_us"] for row in attention_rows
        )
        / 1000.0,
        "top_cuda_operators": rows[:15],
    }


GRAPH_PATTERNS = (
    "aten::_autocast_to_reduced_precision",
    "aten::_autocast_to_full_precision",
    "aten::_to_copy",
    "aten::to",
    "aten::copy_",
    "aten::scaled_dot_product_attention",
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::softmax",
    "aten::einsum",
    "aten::matmul",
    "aten::bmm",
    "aten::mm",
    "aten::linear",
    "aten::conv",
)


def graph_inventory(method: Any) -> dict[str, Any]:
    graph = str(method.graph)
    return {
        "characters": len(graph),
        "patterns": {pattern: graph.count(pattern) for pattern in GRAPH_PATTERNS},
    }


def make_export_compatible_model(
    checkpoint: Path, device: torch.device
) -> tuple[Any, nn.Module, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "device": device,
        "input_resolution": INPUT_RESOLUTION,
    }
    signature = inspect.signature(TAPNextPP.from_checkpoint)
    if "half_precision" in signature.parameters:
        kwargs["half_precision"] = False
    if "compile_model" in signature.parameters:
        kwargs["compile_model"] = False
    high_level = TAPNextPP.from_checkpoint(str(checkpoint), **kwargs)
    inner = high_level._model.eval()
    if next(inner.parameters()).dtype != torch.float32:
        inner = inner.float()
    return high_level, inner, kwargs


def print_result(name: str, result: dict[str, Any]) -> None:
    timing = result["timing"]
    print(
        f"{name:24s} CUDA Event {timing['event_fps']:7.2f} FPS "
        f"({timing['event_mean_ms']:.3f} ms, p95 {timing['event_p95_ms']:.3f} ms) | "
        f"wall {timing['wall_fps']:7.2f} FPS | "
        f"peak {result['peak_allocated_gib']:.3f} GiB"
    )


def run_official(
    checkpoint: Path,
    device: torch.device,
    video: torch.Tensor,
    queries: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n[1/3] 加载官方CKPT（from_checkpoint参数与原使用脚本一致）……")
    model = TAPNextPP.from_checkpoint(
        str(checkpoint), device=device, input_resolution=INPUT_RESOLUTION
    )
    inner = model._model.eval()
    wrapper = EagerTensorWrapper(inner, cache_enabled=True).eval()
    result: dict[str, Any] = {
        "dtype": dtype_inventory(inner),
        "module_dtype_numel": module_dtype_inventory(inner),
        "note": (
            "使用官方默认from_checkpoint配置，但绕过track_frame的CPU预处理/传输/"
            "输出转numpy；直接以与TorchScript相同的CUDA Tensor调用_model。"
        ),
    }

    def initialize() -> tuple[torch.Tensor, ...]:
        return wrapper.initialize(video, queries)

    def step(outputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return wrapper.step(video, *outputs[3:])

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        result["timing"], final_state = cuda_event_benchmark(
            initialize, step, args.warmup, args.iterations
        )
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        del final_state
        if not args.no_profiler:
            result["profiler"] = profile_steps(
                "official_ckpt",
                initialize,
                step,
                args.profile_warmup,
                args.profile_steps,
                args.output_dir,
            )
    print_result("official_ckpt", result)
    del wrapper, model, inner
    synchronize_and_collect()
    return result


def run_eager(
    checkpoint: Path,
    device: torch.device,
    video: torch.Tensor,
    queries: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n[2/3] 加载与导出脚本相同的FP32低层模型……")
    high_level, inner, load_kwargs = make_export_compatible_model(checkpoint, device)
    result: dict[str, Any] = {
        "load_kwargs": {key: str(value) for key, value in load_kwargs.items()},
        "dtype": dtype_inventory(inner),
        "module_dtype_numel": module_dtype_inventory(inner),
        "variants": {},
    }
    for cache_enabled in (True, False):
        name = "eager_cache_on" if cache_enabled else "eager_cache_off"
        print(f"  测试{name}……")
        wrapper = EagerTensorWrapper(inner, cache_enabled=cache_enabled).eval()

        def initialize() -> tuple[torch.Tensor, ...]:
            return wrapper.initialize(video, queries)

        def step(outputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
            return wrapper.step(video, *outputs[3:])

        variant: dict[str, Any] = {}
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            variant["timing"], final_state = cuda_event_benchmark(
                initialize, step, args.warmup, args.iterations
            )
            variant["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            del final_state
            if not args.no_profiler:
                variant["profiler"] = profile_steps(
                    name,
                    initialize,
                    step,
                    args.profile_warmup,
                    args.profile_steps,
                    args.output_dir,
                )
        result["variants"][name] = variant
        print_result(name, variant)
        del wrapper
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    del high_level, inner
    synchronize_and_collect()
    return result


def run_torchscript(
    model_path: Path,
    device: torch.device,
    video: torch.Tensor,
    queries: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print("\n[3/3] 加载TorchScript……")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    result: dict[str, Any] = {
        "dtype_before_freeze": dtype_inventory(model),
        "frozen": not args.no_freeze,
    }
    if not args.no_freeze:
        model = torch.jit.freeze(model, preserved_attrs=["initialize", "step"])
    result["dtype_after_freeze"] = dtype_inventory(model)
    result["initialize_graph"] = graph_inventory(model.initialize)
    result["step_graph"] = graph_inventory(model.step)

    def initialize() -> tuple[torch.Tensor, ...]:
        return model.initialize(video, queries)

    def step(outputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return model.step(video, *outputs[3:])

    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        result["timing"], final_state = cuda_event_benchmark(
            initialize, step, args.warmup, args.iterations
        )
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
        del final_state
        if not args.no_profiler:
            result["profiler"] = profile_steps(
                "torchscript",
                initialize,
                step,
                args.profile_warmup,
                args.profile_steps,
                args.output_dir,
            )
    print_result("torchscript", result)
    del model
    synchronize_and_collect()
    return result


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    checkpoint = args.checkpoint.expanduser().resolve()
    torchscript_path = args.torchscript.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到checkpoint：{checkpoint}")
    if not args.skip_torchscript and not torchscript_path.is_file():
        raise FileNotFoundError(f"找不到TorchScript：{torchscript_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    frame_rgb, query_xy = make_inputs(args.points, args.seed)
    video, queries = make_low_level_inputs(frame_rgb, query_xy, device)
    report: dict[str, Any] = {
        "environment": environment_info(),
        "settings": {
            "checkpoint": str(checkpoint),
            "torchscript": str(torchscript_path),
            "from_checkpoint_signature": str(inspect.signature(TAPNextPP.from_checkpoint)),
            "points": args.points,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_warmup": args.profile_warmup,
            "profile_steps": args.profile_steps,
            "seed": args.seed,
            "torchscript_frozen": not args.no_freeze,
        },
        "scope": (
            "CUDA Event只包围单次递归step；固定同一张512x512输入和相同点数；"
            "不包含视频解码、resize、CPU到GPU传输、绘制或CSV。wall时间同时报告。"
        ),
        "results": {},
    }
    print("GPU：", report["environment"]["gpu"])
    print("输出目录：", args.output_dir)
    print("计时口径：仅递归step GPU工作；预处理、传输、绘制、CSV均排除")

    if not args.skip_official:
        report["results"]["official_ckpt"] = run_official(
            checkpoint, device, video, queries, args
        )
    if not args.skip_eager:
        install_inference_sqrt()
        report["results"]["eager_export_compatible"] = run_eager(
            checkpoint, device, video, queries, args
        )
    if not args.skip_torchscript:
        report["results"]["torchscript"] = run_torchscript(
            torchscript_path, device, video, queries, args
        )

    report_path = args.output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n汇总：", report_path)
    if not args.no_profiler:
        print("请把summary.json与各*_operators.csv发回来；Chrome trace仅在需要时查看。")


if __name__ == "__main__":
    main()
