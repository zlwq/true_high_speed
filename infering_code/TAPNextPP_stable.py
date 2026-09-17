import cv2
import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from pathlib import Path
from matplotlib.backend_bases import MouseButton
from tapnet.tapnextpp.votsp2026.model import TAPNextPP
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "27.mp4"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "output" / "27.mp4"
CSV_OUTPUT_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_points.csv")
SELECTION_WINDOW = "Select points"

CHECKPOINT = os.path.expanduser(
    "~/.cache/tapnextpp/tapnextpp_512.ckpt"
)

CHUNK_SIZE = 50
SCALE = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"


model = TAPNextPP.from_checkpoint(
    CHECKPOINT,
    device=device,
    input_resolution=512
)


cap = cv2.VideoCapture(str(VIDEO_PATH))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


ok, first_frame = cap.read()

if not ok:
    raise RuntimeError("无法读取视频")


if SCALE != 1.0:
    first_frame = cv2.resize(
        first_frame,
        None,
        fx=SCALE,
        fy=SCALE
    )


h, w = first_frame.shape[:2]

def maximize_figure_window(figure):

    manager = figure.canvas.manager
    window = getattr(manager, "window", None)

    if window is not None and hasattr(window, "showMaximized"):
        window.showMaximized()
        return

    if window is not None and hasattr(window, "state"):
        try:
            window.state("zoomed")
            return
        except Exception:
            pass

    if window is not None and hasattr(window, "attributes"):
        try:
            window.attributes("-zoomed", True)
            return
        except Exception:
            pass

    try:
        manager.full_screen_toggle()
    except Exception:
        pass


def select_points(frame):

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]
    selected_points = []
    selection_state = {
        "confirmed": False,
        "cancelled": False,
    }

    figure, axes = plt.subplots()
    figure.canvas.manager.set_window_title(SELECTION_WINDOW)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94)

    def redraw():
        axes.clear()
        axes.imshow(rgb_frame)
        axes.set_title(
            "Left click: add | Right click: undo | "
            "Enter: start | Esc: cancel"
        )
        axes.set_axis_off()

        for point_id, (x, y) in enumerate(selected_points, start=1):
            axes.scatter(x, y, s=70, c="red", edgecolors="white", linewidths=0.8)
            axes.text(
                x + 8,
                y - 8,
                str(point_id),
                color="red",
                fontsize=11,
                weight="bold",
            )

        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == MouseButton.LEFT:
            x = int(np.clip(round(event.xdata), 0, frame_width - 1))
            y = int(np.clip(round(event.ydata), 0, frame_height - 1))
            selected_points.append((x, y))
            redraw()
        elif event.button == MouseButton.RIGHT and selected_points:
            selected_points.pop()
            redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            selection_state["confirmed"] = True
            plt.close(figure)
        elif event.key == "escape":
            selection_state["cancelled"] = True
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    redraw()

    plt.show(block=False)
    plt.pause(0.05)
    maximize_figure_window(figure)
    plt.show()

    if selection_state["cancelled"]:
        raise SystemExit("已取消点选择")
    if not selection_state["confirmed"]:
        raise RuntimeError("点选择窗口已关闭，但没有按Enter确认")
    if not selected_points:
        raise RuntimeError("没有选择任何跟踪点")

    return selected_points


try:
    points = select_points(first_frame)
except BaseException:
    cap.release()
    raise


query_points = np.array(
    points,
    dtype=np.float32
)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"无法创建输出视频：{OUTPUT_PATH}")


def visible_point_coordinates(positions, visible):

    return {
        point_id: (int(round(x)), int(round(y)))
        for point_id, ((x, y), is_visible) in enumerate(
            zip(positions, visible),
            start=1
        )
        if bool(is_visible)
    }


def write_point_csv(csv_path, frame_records, point_count):

    fieldnames = ["frame_index", "timestamp_sec"]
    for point_id in range(1, point_count + 1):
        fieldnames.extend(
            [f"id_{point_id}_x_px", f"id_{point_id}_y_px"]
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for frame_index, timestamp_sec, visible_points in frame_records:
            row = {
                "frame_index": frame_index,
                "timestamp_sec": f"{timestamp_sec:.9f}",
            }
            for point_id, (x, y) in visible_points.items():
                row[f"id_{point_id}_x_px"] = x
                row[f"id_{point_id}_y_px"] = y
            csv_writer.writerow(row)


def draw_tracks(
    frame,
    positions,
    visible
):

    img = frame.copy()

    for i, ((x, y), vis) in enumerate(
        zip(
            positions,
            visible
        )
    ):

        if vis:

            x = int(round(x))
            y = int(round(y))

            cv2.circle(
                img,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                img,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        else:

            x = int(round(x))
            y = int(round(y))

            cv2.circle(
                img,
                (x, y),
                6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                img,
                str(i + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    return img


positions, visible, state = model.track_frame(
    first_frame,
    query_points_xy=query_points
)


writer.write(
    draw_tracks(
        first_frame,
        positions,
        visible
    )
)


processed_frames = 1
frame_records = [
    (0, 0.0, visible_point_coordinates(positions, visible))
]


while True:

    chunk = []

    for _ in range(CHUNK_SIZE):

        ok, frame = cap.read()

        if not ok:
            break

        if SCALE != 1.0:

            frame = cv2.resize(
                frame,
                None,
                fx=SCALE,
                fy=SCALE
            )

        chunk.append(frame)


    if len(chunk) == 0:
        break


    for frame in chunk:

        positions, visible, state = model.track_frame(
            frame,
            state=state
        )

        output_frame = draw_tracks(
            frame,
            positions,
            visible
        )

        writer.write(
            output_frame
        )


        frame_index = processed_frames
        frame_records.append(
            (
                frame_index,
                frame_index / fps,
                visible_point_coordinates(positions, visible),
            )
        )

        processed_frames += 1


    print(
        f"\r已处理 {processed_frames}/{total_frames} 帧",
        end=""
    )


    if not ok:
        break


cap.release()
writer.release()
write_point_csv(CSV_OUTPUT_PATH, frame_records, len(points))

print()
print(
    "完成:",
    OUTPUT_PATH
)

print(
    "坐标CSV:",
    CSV_OUTPUT_PATH
)