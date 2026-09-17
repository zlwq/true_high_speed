import csv
import gc
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_bases import MouseButton


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "1.mp4"
MODEL_PATH = PROJECT_ROOT / "model" / "trackon2_dinov2_pure_torchscript.pt"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "output"

MODEL_HEIGHT = 384
MODEL_WIDTH = 512
WARMUP_INITIALIZE_COUNT = 3
WARMUP_STEP_COUNT = 10


def read_first_frame(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError("无法读取视频第一帧")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    return frame, fps, declared_total


def select_points_matplotlib(first_frame):
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    points = []
    confirmed = {"value": False}

    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        fig.canvas.manager.set_window_title("Track-On2 TorchScript 首帧点选")
    except Exception:
        pass

    def redraw():
        ax.clear()
        ax.imshow(rgb)
        ax.set_title("左键添加点 | 右键删除最近点 | Enter确认 | Esc取消")
        ax.set_axis_off()

        for point_id, (x, y) in enumerate(points, start=1):
            ax.scatter(
                [x],
                [y],
                s=70,
                c="red",
                edgecolors="white",
                linewidths=1.2,
            )
            ax.text(
                x + 7,
                y - 7,
                str(point_id),
                color="yellow",
                fontsize=11,
                weight="bold",
            )

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if event.button == MouseButton.LEFT:
            points.append((x, y))
        elif event.button == MouseButton.RIGHT and points:
            distances = [
                (point_x - x) ** 2 + (point_y - y) ** 2
                for point_x, point_y in points
            ]
            del points[int(np.argmin(distances))]

        redraw()

    def on_key(event):
        if event.key == "enter":
            if not points:
                ax.set_title("请至少选择一个点，再按 Enter 确认")
                fig.canvas.draw_idle()
                return
            confirmed["value"] = True
            plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    if not confirmed["value"]:
        raise RuntimeError("用户取消了点选")

    return np.asarray(points, dtype=np.float32)


def scale_points_to_model(points_xy, original_width, original_height):
    model_points = points_xy.astype(np.float32, copy=True)
    model_points[:, 0] *= MODEL_WIDTH / float(original_width)
    model_points[:, 1] *= MODEL_HEIGHT / float(original_height)
    return model_points


def scale_points_to_original(points_xy, original_width, original_height):
    original_points = points_xy.astype(np.float32, copy=True)
    original_points[:, 0] *= original_width / float(MODEL_WIDTH)
    original_points[:, 1] *= original_height / float(MODEL_HEIGHT)
    return original_points


def frame_to_cuda_tensor(frame_bgr, device):
    resized = cv2.resize(
        frame_bgr,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb)
    tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def load_model(model_path: Path, device):
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 TorchScript 模型：{model_path}")

    print(f"正在加载 TorchScript：{model_path}")
    model = torch.jit.load(str(model_path), map_location=device).eval()

    if not hasattr(model, "initialize") or not hasattr(model, "step"):
        raise RuntimeError("TorchScript 中没有 initialize 或 step 方法")

    for parameter in model.parameters():
        if not parameter.is_cuda:
            raise RuntimeError("模型参数没有全部加载到 CUDA")

    return model


def warmup(model, first_frame_tensor, query_points_tensor):
    print(
        f"开始冷启动预热：{WARMUP_INITIALIZE_COUNT} 次 initialize + "
        f"{WARMUP_STEP_COUNT} 次 step"
    )

    with torch.inference_mode():
        state = None
        for _ in range(WARMUP_INITIALIZE_COUNT):
            state = model.initialize(first_frame_tensor, query_points_tensor)

        positions, visible, query_features, temporal_mask, point_memory = state

        for _ in range(WARMUP_STEP_COUNT):
            positions, visible, temporal_mask, point_memory = model.step(
                first_frame_tensor,
                query_features,
                temporal_mask,
                point_memory,
            )

    torch.cuda.synchronize()
    del state
    del positions, visible, query_features, temporal_mask, point_memory
    gc.collect()
    torch.cuda.empty_cache()
    print("预热完成，开始正式视频。")


def output_paths(video_path: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_video = OUTPUT_DIR / (
        video_path.stem + "_track_on2_dinov2_annotated.mp4"
    )
    output_csv = OUTPUT_DIR / (
        video_path.stem + "_track_on2_dinov2_raw.csv"
    )
    return output_video, output_csv


def csv_fieldnames(point_count):
    fieldnames = ["frame_index", "timestamp_sec"]
    for point_id in range(1, point_count + 1):
        fieldnames.extend(
            [
                f"id_{point_id}_x_px",
                f"id_{point_id}_y_px",
                f"id_{point_id}_visible",
            ]
        )
    return fieldnames


def draw_and_build_csv_row(
    frame,
    frame_index,
    fps,
    positions,
    visibility,
):
    row = {
        "frame_index": frame_index,
        "timestamp_sec": f"{frame_index / fps:.9f}",
    }

    for point_index, ((x, y), is_visible) in enumerate(
        zip(positions, visibility), start=1
    ):
        visible = bool(is_visible)
        row[f"id_{point_index}_visible"] = int(visible)

        if visible:
            row[f"id_{point_index}_x_px"] = f"{float(x):.6f}"
            row[f"id_{point_index}_y_px"] = f"{float(y):.6f}"

        draw_x = int(round(float(x)))
        draw_y = int(round(float(y)))
        color = (0, 0, 255) if visible else (0, 255, 255)
        thickness = -1 if visible else 2

        cv2.circle(
            frame,
            (draw_x, draw_y),
            6,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(point_index),
            (draw_x + 8, draw_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return row


def run_video_inference(
    model,
    video_path,
    output_video,
    output_csv,
    query_points_model,
    fps,
    original_width,
    original_height,
    declared_total,
    device,
):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频：{video_path}")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (original_width, original_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_video}")

    query_points_tensor = torch.from_numpy(query_points_model).to(
        device=device,
        dtype=torch.float32,
    )
    fieldnames = csv_fieldnames(len(query_points_model))

    frame_index = 0
    query_features = None
    temporal_mask = None
    point_memory = None
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    try:
        with output_csv.open(
            "w", newline="", encoding="utf-8-sig"
        ) as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_tensor = frame_to_cuda_tensor(frame, device)

                try:
                    with torch.inference_mode():
                        if frame_index == 0:
                            (
                                positions_tensor,
                                visible_tensor,
                                query_features,
                                temporal_mask,
                                point_memory,
                            ) = model.initialize(
                                frame_tensor,
                                query_points_tensor,
                            )
                        else:
                            (
                                positions_tensor,
                                visible_tensor,
                                temporal_mask,
                                point_memory,
                            ) = model.step(
                                frame_tensor,
                                query_features,
                                temporal_mask,
                                point_memory,
                            )
                except torch.OutOfMemoryError as error:
                    completed = frame_index
                    del frame_tensor
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA 显存不足：发生在全局帧 "
                        f"{frame_index}，累计已完成 {completed} 帧。"
                        "该TorchScript是逐帧模型，显存不会随视频长度增长；"
                        "请关闭其他GPU程序或减少跟踪点。"
                    ) from error

                positions_model = (
                    positions_tensor.detach().float().cpu().numpy()
                )
                visibility = visible_tensor.detach().bool().cpu().numpy()
                positions_original = scale_points_to_original(
                    positions_model,
                    original_width,
                    original_height,
                )

                row = draw_and_build_csv_row(
                    frame,
                    frame_index,
                    fps,
                    positions_original,
                    visibility,
                )
                csv_writer.writerow(row)
                writer.write(frame)

                frame_index += 1
                del frame_tensor, positions_tensor, visible_tensor

                if frame_index == 1 or frame_index % 10 == 0:
                    if declared_total > 0:
                        progress = f"{frame_index}/{declared_total}"
                    else:
                        progress = str(frame_index)
                    print(f"\rTrack-On2 已处理 {progress} 帧", end="")

    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError("输入视频没有可用帧")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    print()
    print(
        f"推理完成：{frame_index} 帧，{elapsed:.3f} 秒，"
        f"平均 {frame_index / elapsed:.2f} FPS，"
        f"CUDA峰值 {peak_gib:.2f} GiB"
    )

    if declared_total > 0 and frame_index != declared_total:
        print(
            f"提示：视频声明 {declared_total} 帧，实际读取 {frame_index} 帧。"
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，已拒绝回退CPU")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")

    video_path = VIDEO_PATH.expanduser().resolve()
    model_path = MODEL_PATH.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到输入视频：{video_path}")

    first_frame, fps, declared_total = read_first_frame(video_path)
    original_height, original_width = first_frame.shape[:2]
    selected_points = select_points_matplotlib(first_frame)
    query_points_model = scale_points_to_model(
        selected_points,
        original_width,
        original_height,
    )

    print(f"PyTorch：{torch.__version__}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(
        f"视频：{video_path}，{original_width}×{original_height}，"
        f"{fps:.6f} FPS，{len(selected_points)} 个跟踪点"
    )

    model = load_model(model_path, device)
    first_frame_tensor = frame_to_cuda_tensor(first_frame, device)
    query_points_tensor = torch.from_numpy(query_points_model).to(
        device=device,
        dtype=torch.float32,
    )
    warmup(model, first_frame_tensor, query_points_tensor)
    del first_frame_tensor, query_points_tensor

    output_video, output_csv = output_paths(video_path)
    run_video_inference(
        model=model,
        video_path=video_path,
        output_video=output_video,
        output_csv=output_csv,
        query_points_model=query_points_model,
        fps=fps,
        original_width=original_width,
        original_height=original_height,
        declared_total=declared_total,
        device=device,
    )

    print(f"标注视频：{output_video}")
    print(f"原始CSV：{output_csv}")


if __name__ == "__main__":
    main()
