from pathlib import Path
import csv
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
import torch 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "torchscript"
    / "tapnextpp_stable_512_q4_fp16_ts.pt"
)
VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "27.mp4"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "assets"
    / "output"
    / "27_test.mp4"
)
CSV_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_points.csv")

INPUT_RESOLUTION = 512
MODEL_COORDINATE_SIZE = 256.0
JIT_WARMUP_INITIALIZES = 3
JIT_WARMUP_STEPS = 10
STEADY_SKIP_STEPS = 10


def select_points(frame):
    figure, axes = plt.subplots()
    axes.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes.set_title("Left: add points | Right: undo | Enter: start")
    axes.set_axis_off()
    plt.show(block=False)
    plt.pause(0.05)

    manager = figure.canvas.manager
    window = getattr(manager, "window", None)
    if window is not None and hasattr(window, "showMaximized"):
        window.showMaximized()
    else:
        try:
            manager.full_screen_toggle()
        except Exception:
            pass

    points = plt.ginput(
        -1,
        timeout=0,
        show_clicks=True,
        mouse_add=MouseButton.LEFT,
        mouse_pop=MouseButton.RIGHT,
        mouse_stop=MouseButton.MIDDLE,
    )
    plt.close(figure)
    if not points:
        raise RuntimeError("至少选择1个点")
    return np.asarray(points, dtype=np.float32)


def prepare_video_tensor(frame, device):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_RESOLUTION, INPUT_RESOLUTION))
    array = np.ascontiguousarray(rgb.astype(np.float32) / 127.5 - 1.0)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def prepare_queries(points_xy, width, height, device):
    queries = np.zeros((1, len(points_xy), 3), dtype=np.float32)
    queries[0, :, 1] = points_xy[:, 1] / height * MODEL_COORDINATE_SIZE
    queries[0, :, 2] = points_xy[:, 0] / width * MODEL_COORDINATE_SIZE
    return torch.from_numpy(queries).to(device)


def decode_outputs(outputs, width, height):
    tracks = outputs[0][0, 0].detach().float().cpu().numpy()
    visible_logits = outputs[2][0, 0].detach().float().cpu().numpy()
    if visible_logits.ndim == 2:
        visible_logits = visible_logits[:, 0]
    visible = visible_logits > 0

    positions = np.empty((len(tracks), 2), dtype=np.float32)
    positions[:, 0] = tracks[:, 1] / MODEL_COORDINATE_SIZE * width
    positions[:, 1] = tracks[:, 0] / MODEL_COORDINATE_SIZE * height
    return positions, visible


def draw_points(frame, positions, visible):
    result = frame.copy()
    for point_id, ((x, y), is_visible) in enumerate(
        zip(positions, visible), start=1
    ):
        x, y = int(round(x)), int(round(y))
        color = (0, 0, 255) if bool(is_visible) else (0, 255, 255)
        cv2.circle(result, (x, y), 6, color, -1 if is_visible else 2)
        cv2.putText(
            result,
            str(point_id),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return result


def write_csv_row(writer, frame_index, timestamp, positions, visible):
    row = {
        "frame_index": frame_index,
        "timestamp_sec": f"{timestamp:.9f}",
    }
    for point_id, ((x, y), is_visible) in enumerate(
        zip(positions, visible), start=1
    ):
        if bool(is_visible):
            row[f"id_{point_id}_x_px"] = int(round(x))
            row[f"id_{point_id}_y_px"] = int(round(y))
    writer.writerow(row)


def assert_cuda_outputs(outputs):
    for index, tensor in enumerate(outputs):
        if index == 3:
            # next_step只是一个int64零维帧计数器。Trace可能把它常量化到
            # CPU；它只有8字节，允许留在CPU，不影响GPU模型和大缓存。
            if tensor.ndim != 0 or tensor.dtype != torch.int64:
                raise RuntimeError("outputs[3]不是预期的int64零维step")
            continue
        if not tensor.is_cuda:
            raise RuntimeError(f"输出/状态outputs[{index}]不在CUDA上")


def timed_cuda_call(call, device):
    """同时测量CUDA Event时间和包含Python/JIT调度的wall时间。"""
    torch.cuda.synchronize(device)
    started_event = torch.cuda.Event(enable_timing=True)
    finished_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    started_event.record()
    outputs = call()
    finished_event.record()
    finished_event.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000.0
    event_ms = started_event.elapsed_time(finished_event)
    return outputs, event_ms, wall_ms


def print_latency_sequence(title, values):
    formatted = ", ".join(
        f"{index}:{value:.2f}ms" for index, value in enumerate(values)
    )
    print(f"{title}：{formatted}")


def fps_from_ms(values):
    if not values:
        return None
    return 1000.0 * len(values) / sum(values)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行GPU测试")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(VIDEO_PATH)

    device = torch.device("cuda:0")
    print("GPU：", torch.cuda.get_device_name(0))
    print("加载TorchScript：", MODEL_PATH)
    model = torch.jit.load(str(MODEL_PATH), map_location=device).eval()

    model = torch.jit.freeze(model,preserved_attrs=["initialize", "step"],)
        
    parameters = list(model.parameters())
    if parameters and not all(parameter.is_cuda for parameter in parameters):
        raise RuntimeError("模型参数没有全部加载到CUDA")
    print("模型参数设备：CUDA")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"无法读取视频：{VIDEO_PATH}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("无法读取第一帧")
    height, width = first_frame.shape[:2]
    points = select_points(first_frame)
    point_count = len(points)
    print(f"本次选择点数：{point_count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not video_writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{OUTPUT_PATH}")

    fields = ["frame_index", "timestamp_sec"]
    for point_id in range(1, point_count + 1):
        fields.extend([f"id_{point_id}_x_px", f"id_{point_id}_y_px"])

    queries = prepare_queries(points, width, height, device)
    # 先用第一帧创建一条独立的递归序列，只用于触发TorchScript profiling、
    # 融合和CUDA内核编译。预热状态随后丢弃，正式跟踪会重新initialize。
    warmup_video = prepare_video_tensor(first_frame, device)
    warmup_initialize_event_ms = []
    warmup_initialize_wall_ms = []
    warmup_step_event_ms = []
    warmup_step_wall_ms = []
    print(
        f"开始JIT冷启动预热：{JIT_WARMUP_INITIALIZES}次initialize + "
        f"{JIT_WARMUP_STEPS}个step……"
    )
    with torch.inference_mode():
        # initialize和step是两张独立TorchScript图。initialize也要重复执行，
        # 否则它的第二阶段JIT优化可能落到正式第一帧，造成约1.6秒延迟。
        for _ in range(JIT_WARMUP_INITIALIZES):
            warm_outputs, event_ms, wall_ms = timed_cuda_call(
                lambda: model.initialize(warmup_video, queries), device
            )
            warmup_initialize_event_ms.append(event_ms)
            warmup_initialize_wall_ms.append(wall_ms)
            assert_cuda_outputs(warm_outputs)

        # 使用最后一次initialize产生的状态预热递归step图。
        for _ in range(JIT_WARMUP_STEPS):
            previous_outputs = warm_outputs
            warm_outputs, event_ms, wall_ms = timed_cuda_call(
                lambda previous=previous_outputs: model.step(
                    warmup_video, *previous[3:]
                ),
                device,
            )
            del previous_outputs
            warmup_step_event_ms.append(event_ms)
            warmup_step_wall_ms.append(wall_ms)
    print_latency_sequence(
        "预热initialize CUDA Event", warmup_initialize_event_ms
    )
    print_latency_sequence("预热initialize wall", warmup_initialize_wall_ms)
    print_latency_sequence("预热step CUDA Event", warmup_step_event_ms)
    print_latency_sequence("预热step wall", warmup_step_wall_ms)
    del warm_outputs, warmup_video
    torch.cuda.synchronize(device)

    frame_event_ms = []
    frame_wall_ms = []
    processed_frames = 0
    torch.cuda.reset_peak_memory_stats(device)

    try:
        with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()

            frame = first_frame
            outputs = None
            with torch.inference_mode():
                while True:
                    video = prepare_video_tensor(frame, device)
                    if outputs is None:
                        outputs, event_ms, wall_ms = timed_cuda_call(
                            lambda: model.initialize(video, queries), device
                        )
                    else:
                        previous_outputs = outputs
                        outputs, event_ms, wall_ms = timed_cuda_call(
                            lambda previous=previous_outputs: model.step(
                                video, *previous[3:]
                            ),
                            device,
                        )
                        del previous_outputs
                    frame_event_ms.append(event_ms)
                    frame_wall_ms.append(wall_ms)

                    assert_cuda_outputs(outputs)
                    positions, visible = decode_outputs(outputs, width, height)
                    if len(positions) != point_count:
                        raise RuntimeError(
                            f"模型输出点数{len(positions)}与选择点数{point_count}不符"
                        )
                    video_writer.write(draw_points(frame, positions, visible))
                    timestamp = processed_frames / fps
                    write_csv_row(
                        csv_writer,
                        processed_frames,
                        timestamp,
                        positions,
                        visible,
                    )
                    processed_frames += 1
                    if processed_frames == 1:
                        print(
                            "动态点数首帧通过；模型输出、query_points及"
                            "24个大缓存全部位于CUDA"
                        )
                        print("step标量设备：", outputs[3].device)
                    if processed_frames % 10 == 0:
                        print(
                            f"\r已处理 {processed_frames}/{total_frames} 帧",
                            end="",
                        )

                    ok, frame = capture.read()
                    if not ok:
                        break
    finally:
        capture.release()
        video_writer.release()

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    step_event_ms = frame_event_ms[1:]
    step_wall_ms = frame_wall_ms[1:]
    steady_event_ms = step_event_ms[STEADY_SKIP_STEPS:]
    steady_wall_ms = step_wall_ms[STEADY_SKIP_STEPS:]
    print()
    print(f"正式initialize CUDA Event：{frame_event_ms[0]:.2f} ms")
    print(f"正式initialize wall：{frame_wall_ms[0]:.2f} ms")
    print_latency_sequence(
        "正式前10个step CUDA Event", step_event_ms[:STEADY_SKIP_STEPS]
    )
    print_latency_sequence("正式前10个step wall", step_wall_ms[:STEADY_SKIP_STEPS])
    print(f"全程模型CUDA Event：{fps_from_ms(frame_event_ms):.2f} FPS")
    print(f"全程模型wall：{fps_from_ms(frame_wall_ms):.2f} FPS")
    if steady_event_ms:
        print(
            f"跳过前{STEADY_SKIP_STEPS}个step后的稳定CUDA Event："
            f"{fps_from_ms(steady_event_ms):.2f} FPS"
        )
        print(
            f"跳过前{STEADY_SKIP_STEPS}个step后的稳定wall："
            f"{fps_from_ms(steady_wall_ms):.2f} FPS"
        )
    else:
        print(
            f"视频step不足{STEADY_SKIP_STEPS + 1}个，无法计算稳定段FPS"
        )
    print(f"CUDA峰值已分配显存：{peak_gib:.2f} GiB")
    print("输出视频：", OUTPUT_PATH)
    print("输出CSV：", CSV_PATH)


if __name__ == "__main__":
    main()