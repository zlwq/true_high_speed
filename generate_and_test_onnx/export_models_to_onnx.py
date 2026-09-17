from __future__ import annotations

import argparse
import gc
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "model").is_dir():
            return candidate
    return script_dir.parent


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_DIR = PROJECT_ROOT / "model" / "onnx"

RING_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "ring_pose"
    / "yolo26n_pose_ring"
    / "weights"
    / "best_ring_class0.pt"
)
COORDINATE_MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "coordinate_tracker"
    / "best_coordinate_association.pt"
)

TAPNEXTPP_CHECKPOINT = (
    Path.home() / ".cache" / "tapnextpp" / "tapnextpp_512.ckpt"
)
TAPNEXT_FAST_CHECKPOINT = (
    Path.home() / ".cache" / "tapnext" / "bootstapnext_ckpt.npz"
)

OPSET_VERSION = 18
YOLO_IMAGE_SIZE = 640


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"找不到{description}：{path}")


def check_onnx_model(path: Path) -> None:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("请先安装 onnx：pip install onnx") from error

    print(f"检查 ONNX：{path}")
    onnx.checker.check_model(str(path), full_check=False)


def coordinate_cumprod_symbolic(graph, tensor, dim, dtype=None):
    """把本模型的非负 CumProd 转为标准 ONNX 算子组合。

    此处输入固定来自 1 - clamp(visibility, 0, 1)，范围为 [0, 1]，因此
    CumProd(x) == Exp(CumSum(Log(x)))。x=0 时 Log(0)=-inf，Exp(-inf)=0，
    与原始前缀乘积语义一致。
    """
    from torch.onnx import symbolic_helper

    del dtype
    axis = symbolic_helper._get_const(dim, "i", "dim")
    axis_tensor = graph.op(
        "Constant",
        value_t=torch.tensor(axis, dtype=torch.int64),
    )
    log_values = graph.op("Log", tensor)
    cumulative_log = graph.op("CumSum", log_values, axis_tensor)
    return graph.op("Exp", cumulative_log)


def coordinate_boolean_iand_symbolic(graph, left, right):
    """将布尔 Tensor 的原地 &= 转为无副作用的标准 ONNX And。"""
    return graph.op("And", left, right)


def coordinate_boolean_eye_symbolic(graph, size, *unused_arguments):
    """用 Range + Equal 生成布尔单位矩阵，避免布尔 EyeLike。"""
    del unused_arguments
    zero = graph.op(
        "Constant",
        value_t=torch.tensor(0, dtype=torch.int64),
    )
    one = graph.op(
        "Constant",
        value_t=torch.tensor(1, dtype=torch.int64),
    )
    axis_zero = graph.op(
        "Constant",
        value_t=torch.tensor([0], dtype=torch.int64),
    )
    axis_one = graph.op(
        "Constant",
        value_t=torch.tensor([1], dtype=torch.int64),
    )
    indices = graph.op("Range", zero, size, one)
    row_indices = graph.op("Unsqueeze", indices, axis_one)
    column_indices = graph.op("Unsqueeze", indices, axis_zero)
    return graph.op("Equal", row_indices, column_indices)


def export_ring_pose() -> Path:
    """导出圆环 YOLO Pose；letterbox、解码和 NMS 留在 C++。"""
    require_file(RING_MODEL_PATH, "圆环YOLO模型")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "请先安装 ultralytics：pip install ultralytics"
        ) from error

    print(f"加载圆环模型：{RING_MODEL_PATH}")
    model = YOLO(str(RING_MODEL_PATH))
    exported_path = model.export(
        format="onnx",
        imgsz=YOLO_IMAGE_SIZE,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=OPSET_VERSION,
        nms=False,
        device="cpu",
    )

    exported_path = Path(exported_path)
    output_path = ONNX_DIR / "ring_pose.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)

    check_onnx_model(output_path)
    print(f"圆环模型导出完成：{output_path}")
    return output_path


def export_coordinate_association(validate_runtime: bool = True) -> Path:
    """导出轨迹数与检测数均动态的坐标关联网络。

    batch、历史长度、上下文长度和特征维度保持固定；track_count、
    detection_count 以及输出类别数(track_count + 1)保持动态。匈牙利分配
    与 ID 状态管理不属于神经网络，不进入 ONNX。
    """
    require_file(COORDINATE_MODEL_PATH, "坐标关联模型")

    try:
        from coordinate_association_model import load_coordinate_model
    except ImportError as error:
        raise RuntimeError(
            "找不到 coordinate_association_model.py，请把本脚本放在项目内运行"
        ) from error

    device = torch.device("cpu")
    model, checkpoint = load_coordinate_model(
        COORDINATE_MODEL_PATH,
        device=device,
    )
    model.eval()

    max_tracks = int(checkpoint["max_tracks"])
    max_detections = int(checkpoint["max_detections"])
    history_length = int(checkpoint["history_length"])
    context_radius = int(checkpoint["context_radius"])
    context_length = context_radius * 2 + 1

    if max_tracks <= 0 or max_detections <= 0:
        raise RuntimeError("checkpoint 中的轨迹或检测容量无效")

    generator = torch.Generator(device="cpu").manual_seed(42)
    history = torch.rand(
        1,
        max_tracks,
        history_length,
        3,
        generator=generator,
        dtype=torch.float32,
    )
    track_mask = torch.ones(1, max_tracks, dtype=torch.bool)
    detections = torch.rand(
        1,
        max_detections,
        6,
        generator=generator,
        dtype=torch.float32,
    )
    detection_context = torch.rand(
        1,
        max_detections,
        context_length,
        8,
        generator=generator,
        dtype=torch.float32,
    )
    detection_mask = torch.ones(1, max_detections, dtype=torch.bool)
    example_inputs = (
        history,
        track_mask,
        detections,
        detection_context,
        detection_mask,
    )

    output_path = ONNX_DIR / "coordinate_association.onnx"
    temporary_output_path = ONNX_DIR / "coordinate_association.dynamic.tmp.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path.unlink(missing_ok=True)

    print(
        "导出动态坐标关联模型："
        f"示例tracks={max_tracks}, 示例detections={max_detections}, "
        f"history={history_length}, context={context_length}"
    )
    # ONNX 没有 CumProd 标准算子。新导出器会卡在 prims.prod，旧导出器
    # 会卡在 aten::cumprod，因此只在本模型导出期间注册等价转换。
    torch.onnx.register_custom_op_symbolic(
        "aten::cumprod",
        coordinate_cumprod_symbolic,
        OPSET_VERSION,
    )
    torch.onnx.register_custom_op_symbolic(
        "aten::__iand_",
        coordinate_boolean_iand_symbolic,
        OPSET_VERSION,
    )
    torch.onnx.register_custom_op_symbolic(
        "aten::eye",
        coordinate_boolean_eye_symbolic,
        OPSET_VERSION,
    )
    try:
        torch.onnx.export(
            model,
            example_inputs,
            str(temporary_output_path),
            input_names=[
                "history",
                "track_mask",
                "detections",
                "detection_context",
                "detection_mask",
            ],
            output_names=["logits"],
            dynamic_axes={
                "history": {1: "track_count"},
                "track_mask": {1: "track_count"},
                "detections": {1: "detection_count"},
                "detection_context": {1: "detection_count"},
                "detection_mask": {1: "detection_count"},
                "logits": {
                    1: "detection_count",
                    2: "track_count_plus_one",
                },
            },
            opset_version=OPSET_VERSION,
            dynamo=False,
            do_constant_folding=True,
        )
    finally:
        unregister_symbolic = getattr(
            torch.onnx,
            "unregister_custom_op_symbolic",
            None,
        )
        if unregister_symbolic is not None:
            unregister_symbolic("aten::cumprod", OPSET_VERSION)
            unregister_symbolic("aten::__iand_", OPSET_VERSION)
            unregister_symbolic("aten::eye", OPSET_VERSION)
    check_onnx_model(temporary_output_path)

    manifest = {
        "model": "coordinate_association",
        "capacity": "dynamic",
        "inputs": {
            "history": [1, "track_count", history_length, 3],
            "track_mask": [1, "track_count"],
            "detections": [1, "detection_count", 6],
            "detection_context": [
                1,
                "detection_count",
                context_length,
                8,
            ],
            "detection_mask": [1, "detection_count"],
        },
        "output": {
            "logits": [
                1,
                "detection_count",
                "track_count + 1",
            ],
            "last_class": "NEW",
        },
        "training_checkpoint_capacity": {
            "max_tracks": max_tracks,
            "max_detections": max_detections,
        },
        "runtime_contract": {
            "minimum_onnx_track_count": 1,
            "zero_real_tracks": (
                "Feed one zero history row with track_mask=false, then remove "
                "the dummy track logit before Hungarian assignment."
            ),
            "zero_detections": "Skip the ONNX call for that frame.",
        },
    }
    manifest_path = ONNX_DIR / "coordinate_association.json"

    if validate_runtime:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "请先安装 onnxruntime：pip install onnxruntime"
            ) from error

        session = ort.InferenceSession(
            str(temporary_output_path),
            providers=["CPUExecutionProvider"],
        )
        validation_cases = [
            # (输入轨迹槽数, 检测数, 是否存在有效轨迹)
            (1, 1, False),
            (1, 12, False),
            (5, 3, True),
            (12, 12, True),
            (16, 16, True),
        ]
        for case_index, (
            case_track_count,
            case_detection_count,
            has_valid_tracks,
        ) in enumerate(validation_cases):
            case_generator = torch.Generator(device="cpu").manual_seed(
                1000 + case_index
            )
            case_history = torch.rand(
                1,
                case_track_count,
                history_length,
                3,
                generator=case_generator,
                dtype=torch.float32,
            )
            case_track_mask = torch.full(
                (1, case_track_count),
                has_valid_tracks,
                dtype=torch.bool,
            )
            case_detections = torch.rand(
                1,
                case_detection_count,
                6,
                generator=case_generator,
                dtype=torch.float32,
            )
            case_context = torch.rand(
                1,
                case_detection_count,
                context_length,
                8,
                generator=case_generator,
                dtype=torch.float32,
            )
            case_detection_mask = torch.ones(
                1,
                case_detection_count,
                dtype=torch.bool,
            )
            case_inputs = (
                case_history,
                case_track_mask,
                case_detections,
                case_context,
                case_detection_mask,
            )
            with torch.inference_mode():
                expected = model(*case_inputs).cpu().numpy()
            actual = session.run(
                ["logits"],
                {
                    "history": case_history.numpy(),
                    "track_mask": case_track_mask.numpy(),
                    "detections": case_detections.numpy(),
                    "detection_context": case_context.numpy(),
                    "detection_mask": case_detection_mask.numpy(),
                },
            )[0]
            expected_shape = (
                1,
                case_detection_count,
                case_track_count + 1,
            )
            if actual.shape != expected_shape:
                raise AssertionError(
                    f"动态输出shape错误：期望{expected_shape}，实际{actual.shape}"
                )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-3,
                atol=1e-4,
            )
            if not has_valid_tracks:
                # ONNX Runtime不接收零长度轨迹轴；确认一条mask=false的占位
                # 轨迹所产生的NEW分数，与PyTorch真正零轨迹分支完全等价。
                zero_history = torch.zeros(
                    1,
                    0,
                    history_length,
                    3,
                    dtype=torch.float32,
                )
                zero_track_mask = torch.zeros(1, 0, dtype=torch.bool)
                with torch.inference_mode():
                    zero_track_expected = model(
                        zero_history,
                        zero_track_mask,
                        case_detections,
                        case_context,
                        case_detection_mask,
                    ).cpu().numpy()
                np.testing.assert_allclose(
                    actual[:, :, -1:],
                    zero_track_expected,
                    rtol=1e-3,
                    atol=1e-4,
                )
            print(
                "动态shape验证通过："
                f"tracks={case_track_count}, "
                f"detections={case_detection_count}, "
                f"active_tracks={has_valid_tracks}"
            )
        print("坐标关联模型全部动态shape PyTorch / ONNX Runtime对齐通过")

    # 只有结构检查及可选的多shape数值验证全部通过，才覆盖正式模型。
    temporary_output_path.replace(output_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"坐标关联模型导出完成：{output_path}")
    print(f"接口说明：{manifest_path}")
    return output_path


class TAPNextExportRGLRU(nn.Module):
    """带显式首帧标志的 RG-LRU，仅用于逐帧 ONNX 导出。

    官方实现以 cache is None 区分首帧；ONNX 输入不能用 Python None 动态
    切换，因此在 cache 最后一列保存 reset 标志。首帧为1，输出自动变为0。
    """

    def __init__(self, original):
        super().__init__()
        self.width = int(original.width)
        self.input_gate = original.input_gate
        self.a_gate = original.a_gate
        self.a_param = original.a_param

    def forward(self, x, cache, use_linear_scan=True):
        del use_linear_scan
        previous_state = cache[..., : self.width]
        reset_flag = cache[..., self.width : self.width + 1]
        reset_flag = reset_flag.to(x.dtype).clamp(0.0, 1.0).unsqueeze(1)

        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * gate_a * torch.nn.functional.softplus(self.a_param)
        a = torch.exp(log_a)
        a_square = torch.exp(2.0 * log_a)

        multiplier = torch.sqrt(1.0 - a_square)
        multiplier = reset_flag + (1.0 - reset_flag) * multiplier
        normalized_x = x * gate_x * multiplier.to(x.dtype)

        # 本 ONNX 接口固定每次输入一帧，即时间长度始终为1。
        y_float = (
            a.to(torch.float32) * previous_state.to(torch.float32).unsqueeze(1)
            + normalized_x.to(torch.float32)
        )
        y = y_float.to(x.dtype)
        next_reset_flag = torch.zeros_like(
            reset_flag[:, 0, :],
            dtype=torch.float32,
        )
        next_cache = torch.cat(
            [y_float[:, -1, :], next_reset_flag],
            dim=-1,
        )
        return y, next_cache


def install_tapnext_export_rglru(model, lru_module) -> int:
    """替换全部 RG-LRU，同时复用原模型的参数与子层。"""
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, lru_module.RGLRU)
    ]
    for name, module in replacements:
        model.set_submodule(name, TAPNextExportRGLRU(module))

    expected_count = len(model.blocks)
    if len(replacements) != expected_count:
        raise RuntimeError(
            "TAPNext RG-LRU 层数异常："
            f"期望{expected_count}层，实际找到{len(replacements)}层"
        )
    return len(replacements)


class TAPNextOnnxWrapper(nn.Module):
    """把 TAPNextTrackingState 展平为 ONNX 可见的纯 Tensor 接口。"""

    def __init__(self, model, state_type, cache_type):
        super().__init__()
        self.model = model
        self.state_type = state_type
        self.cache_type = cache_type

    def forward(self, video, step, query_points, *flat_caches):
        hidden_state = []
        for index in range(0, len(flat_caches), 2):
            hidden_state.append(
                self.cache_type(
                    rg_lru_state=flat_caches[index],
                    conv1d_state=flat_caches[index + 1],
                )
            )

        state = self.state_type(
            step=step,
            query_points=query_points,
            hidden_state=hidden_state,
        )
        # TAPNext 的公开调用约定是：首帧传 query_points，后续帧只传 state。
        # 这里已经显式构造了 state，因此不能再次把 query_points 当成新点传入。
        tracks, track_logits, visible_logits, new_state = self.model(
            video=video,
            state=state,
        )

        outputs = [
            tracks,
            track_logits,
            visible_logits,
            new_state.step,
            new_state.query_points,
        ]
        for cache in new_state.hidden_state:
            outputs.extend([cache.rg_lru_state, cache.conv1d_state])
        return tuple(outputs)


def load_tapnext_model(variant: str, device: torch.device):
    from tapnet.tapnext import tapnext_lru_modules, tapnext_torch

    if variant == "stable":
        require_file(TAPNEXTPP_CHECKPOINT, "TAPNext++ 512 checkpoint")
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        load_kwargs = {
            "device": device,
            "input_resolution": 512,
        }
        # 不同提交版本的 TAPNext++ 参数略有差异；存在这些开关时明确
        # 关闭半精度与 torch.compile，确保拿到适合 ONNX 导出的原始 FP32 图。
        load_parameters = inspect.signature(
            TAPNextPP.from_checkpoint
        ).parameters
        if "half_precision" in load_parameters:
            load_kwargs["half_precision"] = False
        if "compile_model" in load_parameters:
            load_kwargs["compile_model"] = False
        wrapper = TAPNextPP.from_checkpoint(
            str(TAPNEXTPP_CHECKPOINT),
            **load_kwargs,
        )
        model = wrapper._model
        input_resolution = 512
        output_name = "tapnextpp_512.onnx"
    elif variant == "fast":
        require_file(TAPNEXT_FAST_CHECKPOINT, "BootsTAPNext checkpoint")
        from tapnet.tapnext.tapnext_torch_utils import (
            restore_model_from_jax_checkpoint,
        )

        model = tapnext_torch.TAPNext(image_size=(256, 256))
        model = restore_model_from_jax_checkpoint(
            model,
            str(TAPNEXT_FAST_CHECKPOINT),
        )
        model = model.to(device)
        input_resolution = 256
        output_name = "tapnext_fast_256.onnx"
    else:
        raise ValueError(f"未知 TAPNext 版本：{variant}")

    model.eval()
    return (
        model,
        tapnext_torch.TAPNextTrackingState,
        tapnext_lru_modules.RecurrentBlockCache,
        input_resolution,
        output_name,
    )

 
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="导出项目中的神经网络为 ONNX",
    )
    parser.add_argument(
        "--model",
        choices=("all", "ring", "coordinate", "tapnext"),
        default="all",
    )
    parser.add_argument(
        "--tap-variant",
        choices=("stable", "fast"),
        default="stable",
        help="stable=TAPNext++ 512，fast=BootsTAPNext 256",
    )
    parser.add_argument(
        "--tap-points",
        type=int,
        default=4,
        help="TAPNext 导出时用于追踪动态点数的示例点数",
    )
    parser.add_argument(
        "--skip-coordinate-runtime-check",
        action="store_true",
    )
    parser.add_argument(
        "--validate-tap-runtime",
        action="store_true",
        help="使用 ONNX Runtime 检查大模型；需要额外内存和时间",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"ONNX 输出目录：{ONNX_DIR}")

    if args.model in ("all", "ring"):
        export_ring_pose()
        gc.collect()

    if args.model in ("all", "coordinate"):
        export_coordinate_association(
            validate_runtime=not args.skip_coordinate_runtime_check,
        )
        gc.collect()
 

    print("全部指定模型导出完成")


if __name__ == "__main__":
    main()
