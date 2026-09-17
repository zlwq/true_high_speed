from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


INPUT_VIDEO = Path("../assets/2.mp4")
OUTPUT_VIDEO = Path("../assets/2_yoloe_multi.mp4")

REFERENCE_FRAME_INDEX = 0
MODEL_NAME = "yoloe-26s-seg.pt"
IMAGE_SIZE = 960
CONFIDENCE = 0.001
IOU = 0.50
DEVICE = None

def read_reference_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_index < 0 or frame_index >= frame_count:
        cap.release()
        raise RuntimeError(f"参考帧超出范围: {frame_index}, 总帧数: {frame_count}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"无法读取参考帧: {frame_index}")

    return frame, fps if fps > 0 else 30.0, width, height


def select_prompt_boxes(frame):
    height, width = frame.shape[:2]
    boxes = []
    patches = []
    confirmed = False

    figure, axis = plt.subplots(figsize=(8, 10))
    axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axis.axis("off")

    try:
        figure.canvas.manager.set_window_title("Select rings")
    except AttributeError:
        pass

    def update_title():
        axis.set_title(
            f"Left drag: add | Right click: undo | Enter: run | C: clear | Esc: exit | boxes: {len(boxes)}"
        )
        figure.canvas.draw_idle()

    def add_box(eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x1, x2 = sorted((int(round(eclick.xdata)), int(round(erelease.xdata))))
        y1, y2 = sorted((int(round(eclick.ydata)), int(round(erelease.ydata))))

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width - 1))
        y2 = max(y1 + 1, min(y2, height - 1))

        if x2 - x1 < 6 or y2 - y1 < 6:
            return

        boxes.append([x1, y1, x2, y2])
        patch = Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        patches.append(patch)
        axis.add_patch(patch)
        update_title()

    def undo_box(event):
        if event.button != 3 or not boxes:
            return

        boxes.pop()
        patches.pop().remove()
        update_title()

    def clear_boxes():
        boxes.clear()
        while patches:
            patches.pop().remove()
        update_title()

    def key_handler(event):
        nonlocal confirmed

        if event.key in ("enter", " ", "space") and boxes:
            confirmed = True
            plt.close(figure)
        elif event.key in ("c", "C"):
            clear_boxes()
        elif event.key == "escape":
            plt.close(figure)

    selector = RectangleSelector(
        axis,
        add_box,
        useblit=False,
        button=[1],
        minspanx=6,
        minspany=6,
        spancoords="pixels",
        interactive=False,
    )

    figure.canvas.mpl_connect("button_press_event", undo_box)
    figure.canvas.mpl_connect("key_press_event", key_handler)
    update_title()
    plt.tight_layout()
    plt.show()
    selector.set_active(False)

    if not confirmed:
        raise RuntimeError("已取消框选")

    return np.array(boxes, dtype=np.float32)


def draw_detection(frame, xyxy, confidence):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = xyxy.round().astype(int)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"ring {confidence:.2f}",
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    reference_frame, fps, width, height = read_reference_frame(
        INPUT_VIDEO,
        REFERENCE_FRAME_INDEX,
    )

    prompt_boxes = select_prompt_boxes(reference_frame)
    print("已选择参考框:")
    for box in prompt_boxes.astype(int):
        print(box.tolist())

    prompts = {
        "bboxes": prompt_boxes,
        "cls": np.zeros(len(prompt_boxes), dtype=np.int32),
    }

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    model = YOLOE(MODEL_NAME)
    results = model.predict(
        source=str(INPUT_VIDEO),
        refer_image=reference_frame,
        visual_prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        stream=True,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU,
        agnostic_nms=True,
        max_det=20,
        device=DEVICE,
        verbose=False,
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {OUTPUT_VIDEO}")

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                draw_detection(frame, box, float(score))

        writer.write(frame)

    writer.release()
    print(f"完成: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()