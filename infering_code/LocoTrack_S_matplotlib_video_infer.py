import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_bases import MouseButton

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "assets/input/1.mp4"
OUTPUT_PATH = PROJECT_ROOT / "assets/output" 
MODEL_RESOLUTION = 512
QUERY_CHUNK_SIZE = 64
USE_FP16 = True

LOCO_REPO_URL = (
    "https://github.com/cvlab-kaist/locotrack/"
    "archive/refs/heads/main.zip"
)


def ensure_locotrack_source() -> Path:
    script_dir = Path(__file__).resolve().parent
    third_party_dir = script_dir / "third_party"
    repo_dir = third_party_dir / "locotrack-main"
    module_file = (
        repo_dir
        / "locotrack_pytorch"
        / "models"
        / "locotrack_model.py"
    )

    if module_file.is_file():
        return repo_dir / "locotrack_pytorch"

    third_party_dir.mkdir(parents=True, exist_ok=True)
    archive_path = third_party_dir / "locotrack-main.zip.part"

    if archive_path.exists():
        archive_path.unlink()
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print("首次运行，正在下载 LocoTrack 官方源码……")
    try:
        torch.hub.download_url_to_file(
            LOCO_REPO_URL,
            str(archive_path),
            progress=True,
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(third_party_dir)
    except Exception as error:
        if archive_path.exists():
            archive_path.unlink()
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        raise RuntimeError("LocoTrack 官方源码下载或解压失败") from error
    finally:
        if archive_path.exists():
            archive_path.unlink()

    if not module_file.is_file():
        raise RuntimeError("下载完成，但没有找到 LocoTrack PyTorch 模型文件")

    return repo_dir / "locotrack_pytorch"


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
        fig.canvas.manager.set_window_title("LocoTrack-S 首帧点选")
    except Exception:
        pass

    def redraw():
        ax.clear()
        ax.imshow(rgb)
        ax.set_title(
            "左键添加点 | 右键删除最近点 | Enter确认 | Esc取消"
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
                ax.set_title("请至少选择一个点，再按 Enter 确认")
                fig.canvas.draw_idle()
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
    queries[0, :, 1] = (
        points_xy[:, 1] / height * MODEL_RESOLUTION
    )
    queries[0, :, 2] = (
        points_xy[:, 0] / width * MODEL_RESOLUTION
    )
    return torch.from_numpy(queries)


def run_inference(model, video, query_points, width, height):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=USE_FP16,
    ):
        output = model.inference(
            video,
            query_points,
            query_chunk_size=QUERY_CHUNK_SIZE,
            resolution=(MODEL_RESOLUTION, MODEL_RESOLUTION),
            query_format="tyx",
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not output["tracks"].is_cuda or not output["occlusion"].is_cuda:
        raise RuntimeError("LocoTrack-S 输出不在 CUDA，已停止")

    tracks = output["tracks"][0].float().cpu().numpy()
    occluded = output["occlusion"][0].cpu().numpy().astype(bool)

    tracks[..., 0] *= width / MODEL_RESOLUTION
    tracks[..., 1] *= height / MODEL_RESOLUTION

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    frame_count = video.shape[1]
    print(f"LocoTrack-S 模型推理耗时：{elapsed:.3f} 秒")
    print(f"模型平均速度：{frame_count / elapsed:.2f} FPS")
    print(f"CUDA 峰值已分配显存：{peak_gib:.2f} GiB")

    return tracks, occluded


def output_paths(video_path: Path):
    output_video = OUTPUT_PATH / "_locotrack_s_annotated.mp4" 
    output_csv = OUTPUT_PATH /  "_locotrack_s_raw.csv"
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
        raise RuntimeError("LocoTrack-S 轨迹和遮挡输出形状不一致")

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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，已拒绝回退 CPU")

    video_path = VIDEO_PATH.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到输入视频：{video_path}")

    first_frame, fps, declared_total = read_first_frame(video_path)
    height, width = first_frame.shape[:2]
    selected_points = select_points_matplotlib(first_frame)

    locotrack_root = ensure_locotrack_source()
    sys.path.insert(0, str(locotrack_root))
    from models.locotrack_model import load_model

    checkpoint_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / "locotrack_small.ckpt"
    )
    if checkpoint_path.is_file():
        print(f"使用已有 LocoTrack-S 权重：{checkpoint_path}")
    else:
        print(f"首次运行，将自动下载 LocoTrack-S 权重到：{checkpoint_path}")

    model = load_model(model_size="small").to("cuda").eval()
    if not all(parameter.is_cuda for parameter in model.parameters()):
        raise RuntimeError("LocoTrack-S 模型参数未全部加载到 CUDA")

    video = load_video_256(video_path)
    if declared_total > 0 and video.shape[1] != declared_total:
        print(
            f"提示：视频声明 {declared_total} 帧，实际读取 {video.shape[1]} 帧"
        )

    query_points = prepare_query_points(selected_points, width, height)
    print(
        f"开始在 CUDA 上推理：{video.shape[1]} 帧，"
        f"{len(selected_points)} 个跟踪点"
    )
    tracks, occluded = run_inference(
        model,
        video,
        query_points,
        width,
        height,
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
