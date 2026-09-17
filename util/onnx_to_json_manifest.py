from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从任意ONNX模型生成结构接口JSON（不执行模型）"
    )
    parser.add_argument("model", type=Path, help="输入.onnx文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出JSON；默认与ONNX同目录、同名.json",
    )
    parser.add_argument(
        "--include-value-info",
        action="store_true",
        help="同时导出模型内部张量信息，可能产生较大的JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="额外运行onnx.checker；外部权重文件缺失时可能失败",
    )
    return parser.parse_args()


def dtype_name(element_type: int) -> str:
    try:
        return TensorProto.DataType.Name(element_type).lower()
    except ValueError:
        return f"unknown({element_type})"


def tensor_shape(tensor_type: Any) -> list[int | str | None]:
    result: list[int | str | None] = []
    if not tensor_type.HasField("shape"):
        return result
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def describe_type(type_proto: Any) -> dict[str, Any]:
    """把ONNX TypeProto转换成JSON可序列化结构。"""
    kind = type_proto.WhichOneof("value")
    if kind == "tensor_type":
        tensor = type_proto.tensor_type
        return {
            "kind": "tensor",
            "dtype": dtype_name(tensor.elem_type),
            "shape": tensor_shape(tensor),
        }
    if kind == "sparse_tensor_type":
        tensor = type_proto.sparse_tensor_type
        return {
            "kind": "sparse_tensor",
            "dtype": dtype_name(tensor.elem_type),
            "shape": tensor_shape(tensor),
        }
    if kind == "sequence_type":
        return {
            "kind": "sequence",
            "element": describe_type(type_proto.sequence_type.elem_type),
        }
    if kind == "optional_type":
        return {
            "kind": "optional",
            "element": describe_type(type_proto.optional_type.elem_type),
        }
    if kind == "map_type":
        return {
            "kind": "map",
            "key_dtype": dtype_name(type_proto.map_type.key_type),
            "value": describe_type(type_proto.map_type.value_type),
        }
    return {"kind": kind or "unknown"}


def describe_value(value_info: Any) -> dict[str, Any]:
    result = describe_type(value_info.type)
    if value_info.doc_string:
        result["description"] = value_info.doc_string
    return result


def initializer_summary(graph: Any) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    total_values = 0
    external_count = 0
    for initializer in graph.initializer:
        dtype_counts[dtype_name(initializer.data_type)] += 1
        values = 1
        for dimension in initializer.dims:
            values *= int(dimension)
        total_values += values
        if initializer.data_location == TensorProto.EXTERNAL or initializer.external_data:
            external_count += 1
    return {
        "count": len(graph.initializer),
        "tensor_count_by_dtype": dict(sorted(dtype_counts.items())),
        "total_values": total_values,
        "external_tensor_count": external_count,
    }


def make_manifest(model_path: Path, include_value_info: bool) -> dict[str, Any]:
    # 不载入外部权重字节；即使模型使用.onnx.data，也能读取图结构。
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    real_inputs = [item for item in graph.input if item.name not in initializer_names]
    operator_counts = Counter(node.op_type for node in graph.node)

    manifest: dict[str, Any] = {
        "model": graph.name or model_path.stem,
        "file": model_path.name,
        "format": "ONNX",
        "ir_version": int(model.ir_version),
        "producer": {
            "name": model.producer_name,
            "version": model.producer_version,
        },
        "domain": model.domain,
        "model_version": int(model.model_version),
        "opset_imports": [
            {
                "domain": item.domain or "ai.onnx",
                "version": int(item.version),
            }
            for item in model.opset_import
        ],
        "inputs": {item.name: describe_value(item) for item in real_inputs},
        "outputs": {item.name: describe_value(item) for item in graph.output},
        "graph": {
            "node_count": len(graph.node),
            "operator_count": dict(sorted(operator_counts.items())),
            "initializers": initializer_summary(graph),
        },
        "metadata": {item.key: item.value for item in model.metadata_props},
        "semantic_limit": (
            "本文件自动提取结构信息。归一化、颜色顺序、标签、坐标含义、"
            "阈值和输出类别语义只有在ONNX metadata中存在时才能自动保留。"
        ),
    }
    if model.doc_string:
        manifest["description"] = model.doc_string
    if include_value_info:
        manifest["value_info"] = {
            item.name: describe_value(item) for item in graph.value_info
        }
    return manifest


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"输入文件扩展名不是.onnx：{model_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_path.with_suffix(".json")
    )
    manifest = make_manifest(model_path, args.include_value_info)
    if args.validate:
        # 使用路径检查，以便checker在存在外部数据时从模型目录解析它们。
        onnx.checker.check_model(str(model_path), full_check=False)
        manifest["onnx_checker"] = "passed"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("已生成：", output_path)


if __name__ == "__main__":
    main()
