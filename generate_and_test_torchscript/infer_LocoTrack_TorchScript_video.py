import csv
import gc
import os
import time
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_bases import MouseButton


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_SIZE = "base"
VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "1.mp4"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "output"

MODEL_RESOLUTION = 256
MAX_POINTS = 64
USE_FP16 = True
FRAME_PROFILES = (64, 32, 16, 8, 4, 2)


def model_path(frame_profile):
    return PROJECT_ROOT / "model" / (
        f"locotrack_{MODEL_SIZE}_t{frame_profile}_torchscript.pt"
    )


def read_first_frame(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError("无法读取视频第一帧")
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    return frame, float(fps), total_frames


def select_points_matplotlib(first_frame):
    rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    points = []
    state = {"confirmed": False}

    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        fig.canvas.manager.set_window_title(
            f"LocoTrack-{MODEL_SIZE.upper()} TorchScript 首帧点选"
        )
    except Exception:
        pass

    def redraw(message=None):
        ax.clear()
        ax.imshow(rgb)
        ax.set_title(
            message
            or "左键添加点 | 右键删除最近点 | Enter确认 | Esc取消"
        )
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
            if len(points) >= MAX_POINTS:
                redraw(f"最多选择 {MAX_POINTS} 个点")
                return
            points.append((x, y))
        elif event.button == MouseButton.RIGHT and points:
            distances = [
                (px - x) ** 2 + (py - y) ** 2
                for px, py in points
            ]
            del points[int(np.argmin(distances))]

        redraw()

    def on_key(event):
        if event.key == "enter":
            if not points:
                redraw("请至少选择一个点，再按 Enter 确认")
                return
            state["confirmed"] = True
            plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()

    if not state["confirmed"]:
        raise RuntimeError("用户取消了点选")

    return np.asarray(points, dtype=np.float32)


def load_video_256(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        resized = cv2.resize(
            frame,
            (MODEL_RESOLUTION, MODEL_RESOLUTION),
            interpolation=cv2.INTER_AREA,
        )
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        if len(frames) % 100 == 0:
            print(f"\r已读取并缩放 {len(frames)} 帧", end="")

    capture.release()
    print()

    if not frames:
        raise RuntimeError("视频中没有可用帧")

    return np.stack(frames, axis=0)[None]


def prepare_query_points(points_xy, width, height):
    queries = np.zeros((1, len(points_xy), 3), dtype=np.float32)
    queries[0, :, 0] = 0.0
    queries[0, :, 1] = points_xy[:, 1] / height * MODEL_RESOLUTION
    queries[0, :, 2] = points_xy[:, 0] / width * MODEL_RESOLUTION
    return queries


def clear_cuda_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def is_cuda_oom(error):
    if isinstance(error, torch.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def run_inference(model, video, query_points, width, height):
    video_array = np.ascontiguousarray(video)
    query_array = np.ascontiguousarray(query_points)
    video_tensor = torch.from_numpy(video_array).cuda()
    query_tensor = torch.from_numpy(query_array).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=USE_FP16,
    ):
        tracks_tensor, occluded_tensor = model(
            video_tensor,
            query_tensor,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not tracks_tensor.is_cuda or not occluded_tensor.is_cuda:
        raise RuntimeError("TorchScript 输出不在 CUDA，已停止")

    tracks = tracks_tensor[0].float().cpu().numpy()
    occluded = occluded_tensor[0].cpu().numpy().astype(bool)
    tracks[..., 0] *= width / MODEL_RESOLUTION
    tracks[..., 1] *= height / MODEL_RESOLUTION

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    frame_count = video.shape[1]
    print(f"本段推理耗时：{elapsed:.3f} 秒")
    print(f"本段平均速度：{frame_count / elapsed:.2f} FPS")
    print(f"CUDA 峰值已分配显存：{peak_gib:.2f} GiB")

    del video_tensor
    del query_tensor
    del tracks_tensor
    del occluded_tensor
    return tracks, occluded


def try_run_inference(
    model,
    video,
    query_points,
    width,
    height,
):
    try:
        return run_inference(
            model,
            video,
            query_points,
            width,
            height,
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not is_cuda_oom(error):
            raise
        return None


def choose_profile(profiles, maximum_index, remaining_frames):
    maximum_profile = profiles[maximum_index]
    if remaining_frames >= maximum_profile:
        return maximum_index

    required_frames = max(2, remaining_frames)
    candidates = [
        index
        for index in range(maximum_index, len(profiles))
        if profiles[index] >= required_frames
    ]
    if candidates:
        return candidates[-1]
    return len(profiles) - 1


def load_profile_model(profile):
    path = model_path(profile).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {profile} 帧模型：{path}")
    print(f"加载固定 {profile} 帧 TorchScript：{path.name}")
    return torch.jit.load(str(path), map_location="cuda").eval()


def run_inference_segmented(
    video,
    selected_points,
    width,
    height,
    profiles,
):
    total_frames = video.shape[1]
    segment_start = 0
    maximum_profile_index = 0
    loaded_profile = None
    model = None
    current_points = selected_points.astype(np.float32, copy=True)
    track_parts = []
    occlusion_parts = []
    oom_count = 0
    segment_count = 0

    while segment_start < total_frames:
        remaining_frames = total_frames - segment_start
        attempt_profile_index = choose_profile(
            profiles,
            maximum_profile_index,
            remaining_frames,
        )

        while True:
            profile = profiles[attempt_profile_index]
            real_segment_frames = min(profile, remaining_frames)
            segment_end = segment_start + real_segment_frames
            segment_video = video[:, segment_start:segment_end]

            if real_segment_frames < profile:
                padding = np.repeat(
                    segment_video[:, -1:],
                    profile - real_segment_frames,
                    axis=1,
                )
                segment_video = np.concatenate(
                    [segment_video, padding],
                    axis=1,
                )

            if loaded_profile != profile:
                if model is not None:
                    del model
                    clear_cuda_memory()
                model = load_profile_model(profile)
                loaded_profile = profile

            query_points = prepare_query_points(
                current_points,
                width,
                height,
            )

            print(
                f"尝试推理全局帧 {segment_start}~{segment_end - 1} "
                f"（固定 {profile} 帧档位，实际 {real_segment_frames} 帧，"
                f"累计已完成 {segment_start}/{total_frames} 帧）"
            )

            inference_result = try_run_inference(
                model,
                segment_video,
                query_points,
                width,
                height,
            )

            if inference_result is None:
                oom_count += 1
                del segment_video
                del query_points
                del model
                model = None
                loaded_profile = None
                clear_cuda_memory()

                print(
                    f"第 {oom_count} 次 CUDA 显存不足：累计已完成 "
                    f"{segment_start}/{total_frames} 帧；刚才使用的是"
                    f"固定 {profile} 帧档位。"
                )

                if attempt_profile_index + 1 >= len(profiles):
                    raise RuntimeError(
                        "即使固定 2 帧模型仍然显存不足；请关闭其他"
                        "占用 GPU 的程序，或改用 LocoTrack-S。"
                    ) from None

                attempt_profile_index += 1
                maximum_profile_index = max(
                    maximum_profile_index,
                    attempt_profile_index,
                )
                print(
                    f"自动切换到固定 "
                    f"{profiles[attempt_profile_index]} 帧模型。"
                )
                continue

            segment_tracks, segment_occluded = inference_result
            segment_tracks = segment_tracks[:, :real_segment_frames]
            segment_occluded = segment_occluded[:, :real_segment_frames]
            del inference_result
            del segment_video
            del query_points
            segment_count += 1
            break

        if segment_end == total_frames:
            track_parts.append(segment_tracks)
            occlusion_parts.append(segment_occluded)
            segment_start = total_frames
            print(
                f"第 {segment_count} 段完成：已累计处理 "
                f"{segment_start}/{total_frames} 帧。"
            )
            break

        all_points_visible = np.all(~segment_occluded, axis=0)
        continuation_candidates = np.flatnonzero(all_points_visible)
        continuation_candidates = continuation_candidates[
            continuation_candidates > 0
        ]

        if continuation_candidates.size > 0:
            continuation_local = int(continuation_candidates[-1])
            visible_count = len(current_points)
            continuation_reason = "全部跟踪点可见"
        else:
            visible_counts = np.sum(~segment_occluded, axis=0)
            best_visible_count = int(np.max(visible_counts[1:]))
            best_candidates = np.flatnonzero(
                visible_counts == best_visible_count
            )
            best_candidates = best_candidates[best_candidates > 0]
            continuation_local = int(best_candidates[-1])
            visible_count = best_visible_count
            continuation_reason = "可见点数量最多"
            print(
                "提示：本段除起点外没有所有跟踪点同时可见的帧；"
                f"改用可见点最多的帧继续（{visible_count}/"
                f"{len(current_points)} 个点可见）。"
            )

        continuation_global = segment_start + continuation_local
        track_parts.append(segment_tracks[:, :continuation_local])
        occlusion_parts.append(segment_occluded[:, :continuation_local])
        current_points = segment_tracks[
            :,
            continuation_local,
        ].astype(np.float32, copy=True)

        print(
            f"第 {segment_count} 段完成：选择全局帧 "
            f"{continuation_global} 作为下一段起点（"
            f"{continuation_reason}，{visible_count}/"
            f"{len(current_points)}）；累计已确定 "
            f"{continuation_global}/{total_frames} 帧。"
        )

        segment_start = continuation_global
        del segment_tracks
        del segment_occluded
        clear_cuda_memory()

    if model is not None:
        del model
        clear_cuda_memory()

    tracks = np.concatenate(track_parts, axis=1)
    occluded = np.concatenate(occlusion_parts, axis=1)

    if tracks.shape[1] != total_frames:
        raise RuntimeError(
            f"分段合并后得到 {tracks.shape[1]} 帧，"
            f"但输入视频共有 {total_frames} 帧。"
        )

    print(
        f"分段推理及合并完成：共 {segment_count} 段，"
        f"发生 {oom_count} 次显存不足，最终 {total_frames} 帧连续。"
    )
    return tracks, occluded


def output_paths(video_path: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"locotrack_{MODEL_SIZE}_torchscript"
    output_video = OUTPUT_DIR / f"{video_path.stem}_{tag}_annotated.mp4"
    output_csv = OUTPUT_DIR / f"{video_path.stem}_{tag}_raw.csv"
    return output_video, output_csv


def write_results(
    video_path,
    output_video,
    output_csv,
    tracks,
    occluded,
    fps,
    width,
    height,
):
    point_count, frame_count, _ = tracks.shape
    if occluded.shape != (point_count, frame_count):
        raise RuntimeError("轨迹和遮挡输出形状不一致")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("无法重新打开输入视频以写入结果")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出视频：{output_video}")

    fieldnames = ["frame_index", "timestamp_sec"]
    for point_id in range(1, point_count + 1):
        fieldnames.extend(
            [
                f"id_{point_id}_x_px",
                f"id_{point_id}_y_px",
                f"id_{point_id}_visible",
            ]
        )

    with output_csv.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                writer.release()
                capture.release()
                raise RuntimeError(
                    f"输出阶段无法读取第 {frame_index} 帧"
                )

            row = {
                "frame_index": frame_index,
                "timestamp_sec": f"{frame_index / fps:.9f}",
            }

            for point_index in range(point_count):
                point_id = point_index + 1
                x, y = tracks[point_index, frame_index]
                visible = not bool(occluded[point_index, frame_index])

                row[f"id_{point_id}_visible"] = int(visible)
                if visible:
                    row[f"id_{point_id}_x_px"] = f"{x:.6f}"
                    row[f"id_{point_id}_y_px"] = f"{y:.6f}"

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
                    str(point_id),
                    (draw_x + 8, draw_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            csv_writer.writerow(row)
            writer.write(frame)

            if frame_index == 0 or (frame_index + 1) % 10 == 0:
                print(
                    f"\r正在写入结果 {frame_index + 1}/{frame_count} 帧",
                    end="",
                )

    capture.release()
    writer.release()
    print()


def main():
    if MODEL_SIZE not in ("small", "base"):
        raise ValueError('MODEL_SIZE 只能填写 "small" 或 "base"')
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，已拒绝回退 CPU")

    video_path = VIDEO_PATH.expanduser().resolve()
    available_profiles = [
        profile
        for profile in FRAME_PROFILES
        if model_path(profile).expanduser().resolve().is_file()
    ]

    if not video_path.is_file():
        raise FileNotFoundError(f"找不到输入视频：{video_path}")
    if not available_profiles:
        raise FileNotFoundError(
            f"没有找到 LocoTrack-{MODEL_SIZE.upper()} 的任何"
            "固定帧数 TorchScript 模型"
        )

    first_frame, fps, declared_total = read_first_frame(video_path)
    height, width = first_frame.shape[:2]
    selected_points = select_points_matplotlib(first_frame)

    video = load_video_256(video_path)
    if declared_total > 0 and video.shape[1] != declared_total:
        print(
            f"提示：视频声明 {declared_total} 帧，实际读取 "
            f"{video.shape[1]} 帧"
        )

    print(
        f"开始在 CUDA 上推理：{video.shape[1]} 帧，"
        f"{len(selected_points)} 个跟踪点"
    )
    print(f"可用固定帧数档位：{available_profiles}")
    tracks, occluded = run_inference_segmented(
        video,
        selected_points,
        width,
        height,
        available_profiles,
    )

    output_video, output_csv = output_paths(video_path)
    write_results(
        video_path,
        output_video,
        output_csv,
        tracks,
        occluded,
        fps,
        width,
        height,
    )

    print(f"标注视频：{output_video}")
    print(f"原始 CSV：{output_csv}")


if __name__ == "__main__":
    main()