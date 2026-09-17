import cv2
import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import urllib.request
from pathlib import Path
from matplotlib.backend_bases import MouseButton

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "2.mp4"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "output" / "2.mp4" 
CSV_OUTPUT_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_points.csv")
SELECTION_WINDOW = "Select points"

CHUNK_SIZE = 50
SCALE = 1.0

MODEL_SIZE = 256

CHECKPOINT = os.path.expanduser(
    "~/.cache/tapnext/bootstapnext_ckpt.npz"
)

CHECKPOINT_URL = (
    "https://storage.googleapis.com/dm-tapnet/"
    "tapnext/bootstapnext_ckpt.npz"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


os.makedirs(
    os.path.dirname(CHECKPOINT),
    exist_ok=True
)


if not os.path.exists(CHECKPOINT):

    print("正在下载 BootsTAPNext 模型...")

    urllib.request.urlretrieve(
        CHECKPOINT_URL,
        CHECKPOINT
    )

    print("模型下载完成")


print("计算设备:", device)


model = TAPNext(
    image_size=(MODEL_SIZE, MODEL_SIZE)
)

model = restore_model_from_jax_checkpoint(
    model,
    CHECKPOINT
)

model = model.to(device)

model.eval()


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


def prepare_frame(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb = cv2.resize(
        rgb,
        (MODEL_SIZE, MODEL_SIZE)
    )

    tensor = torch.from_numpy(
        rgb
    ).float()

    tensor = (
        tensor / 255.0
    ) * 2.0 - 1.0

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def make_queries(points):

    queries = []

    for x, y in points:

        model_x = (
            x / w
            * MODEL_SIZE
        )

        model_y = (
            y / h
            * MODEL_SIZE
        )

        queries.append(
            [
                0,
                model_y,
                model_x
            ]
        )

    return torch.tensor(
        [queries],
        dtype=torch.float32,
        device=device
    )


def convert_positions(tracks):

    tracks = (
        tracks[0, 0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    result = []

    for y, x in tracks:

        real_x = (
            x
            / MODEL_SIZE
            * w
        )

        real_y = (
            y
            / MODEL_SIZE
            * h
        )

        result.append(
            (real_x, real_y)
        )

    return np.array(
        result,
        dtype=np.float32
    )


def convert_visibility(
    visible_logits
):

    visible = (
        visible_logits[0, 0, :, 0]
        > 0
    )

    return (
        visible
        .detach()
        .cpu()
        .numpy()
    )


def draw_tracks(
    frame,
    positions,
    visible
):

    img = frame.copy()

    for i, (
        (x, y),
        vis
    ) in enumerate(
        zip(
            positions,
            visible
        )
    ):

        x = int(round(x))
        y = int(round(y))

        if vis:

            cv2.circle(
                img,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            color = (
                0,
                0,
                255
            )

        else:

            cv2.circle(
                img,
                (x, y),
                7,
                (0, 255, 255),
                2
            )

            color = (
                0,
                255,
                255
            )

        cv2.putText(
            img,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return img


queries = make_queries(
    points
)


first_tensor = prepare_frame(
    first_frame
)


autocast_enabled = (
    device.type == "cuda"
)


with torch.inference_mode():

    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=autocast_enabled
    ):

        (
            tracks,
            track_logits,
            visible_logits,
            state
        ) = model(
            video=first_tensor,
            query_points=queries
        )


positions = convert_positions(
    tracks
)

visible = convert_visibility(
    visible_logits
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


    for _ in range(
        CHUNK_SIZE
    ):

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


        chunk.append(
            frame
        )


    if len(chunk) == 0:
        break


    for frame in chunk:

        frame_tensor = prepare_frame(
            frame
        )


        with torch.inference_mode():

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled
            ):

                (
                    tracks,
                    track_logits,
                    visible_logits,
                    state
                ) = model(
                    video=frame_tensor,
                    state=state
                )


        positions = convert_positions(
            tracks
        )

        visible = convert_visibility(
            visible_logits
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
        f"\r已处理 "
        f"{processed_frames}/"
        f"{total_frames} 帧",
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