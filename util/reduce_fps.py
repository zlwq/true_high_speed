from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_VIDEO = PROJECT_ROOT / "assets/input/4.mp4"
CRF = 18
PRESET = "medium" 

def read_video_fps(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"视频中没有画面流：{video_path}")
    fps = float(Fraction(streams[0]["avg_frame_rate"]))
    frame_count_text = streams[0].get("nb_frames", "0")
    frame_count = int(frame_count_text) if str(frame_count_text).isdigit() else 0
    if fps <= 0:
        raise RuntimeError("无法读取原视频帧率")
    return fps, frame_count


def make_output_path(video_path, target_fps):
    fps_text = f"{target_fps:g}".replace(".", "p")
    output_path = video_path.with_name(f"{video_path.stem}_{fps_text}fps.mp4")
    index = 2
    while output_path.exists():
        output_path = video_path.with_name(
            f"{video_path.stem}_{fps_text}fps_{index}.mp4"
        )
        index += 1
    return output_path


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"找不到输入视频：{INPUT_VIDEO}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("找不到 ffmpeg，请先安装：sudo apt install ffmpeg")

    source_fps, frame_count = read_video_fps(INPUT_VIDEO)
    print(f"输入视频：{INPUT_VIDEO}")
    print(f"原始帧率：{source_fps:.3f} FPS")
    print(f"原始帧数：{frame_count}")

    try:
        target_fps = float(input("请输入目标帧率：").strip())
    except ValueError as error:
        raise ValueError("目标帧率必须是数字，例如 15 或 7.5") from error

    if target_fps <= 0:
        raise ValueError("目标帧率必须大于 0")
    if target_fps >= source_fps:
        raise ValueError(
            f"目标帧率必须低于原始帧率 {source_fps:.3f} FPS"
        )

    output_path = make_output_path(INPUT_VIDEO, target_fps)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(INPUT_VIDEO),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        f"fps={target_fps:g}:round=near",
        "-c:v",
        "libx264",
        "-preset",
        PRESET,
        "-crf",
        str(CRF),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart",
        "-n",
        str(output_path),
    ]

    print(f"输出视频：{output_path}")
    print("开始抽帧，视频总时长保持不变……")
    subprocess.run(command, check=True)
    print(f"完成：{output_path}")


if __name__ == "__main__":
    main()