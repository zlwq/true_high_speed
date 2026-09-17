import cv2
import numpy as np
from pathlib import Path


VIDEO = Path("../assets/2.mp4")
OUTPUT = Path("../assets/2_lightflow.mp4")


MAX_CORNERS = 200
LK_WINDOW = 60
MAX_LEVEL = 3

def detect_circle(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        160,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circles = []

    for c in contours:

        if len(c) < 5:
            continue

        area = cv2.contourArea(c)

        if area < 100:
            continue

        ellipse = cv2.fitEllipse(c)

        (x, y), (w, h), angle = ellipse

        if max(w, h) / min(w, h) > 3:
            continue

        circles.append(
            np.array(
                [x, y],
                np.float32
            )
        )

    return circles



def choose_points(frame):

    circles = detect_circle(frame)


    # 去除距离太近的候选
    filtered = []

    for p in circles:

        if all(
            np.linalg.norm(
                p-q
            ) > 30
            for q in filtered
        ):
            filtered.append(p)


    circles = filtered


    selected = []


    def redraw():

        show = frame.copy()


        # 所有候选点
        for i,p in enumerate(circles):

            x,y = p.astype(int)

            color = (
                0,
                0,
                255
            )


            # 如果已经选择
            for s in selected:

                if np.linalg.norm(
                    p-s
                ) < 5:

                    color = (
                        0,
                        255,
                        0
                    )


            cv2.circle(
                show,
                (x,y),
                8,
                color,
                -1
            )


        cv2.imshow(
            "choose",
            show
        )


    def mouse(event,x,y,flags,param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return


        click = np.array(
            [
                x,
                y
            ],
            np.float32
        )


        if not circles:
            return


        nearest = min(
            circles,
            key=lambda p:
            np.linalg.norm(
                p-click
            )
        )


        distance = np.linalg.norm(
            nearest-click
        )


        if distance > 40:
            return



        # 已经选中 -> 取消
        for i,s in enumerate(selected):

            if np.linalg.norm(
                s-nearest
            ) < 5:

                selected.pop(i)

                redraw()

                return


        # 没选中 -> 添加

        selected.append(
            nearest.copy()
        )


        redraw()



    cv2.namedWindow(
        "choose"
    )


    cv2.setMouseCallback(
        "choose",
        mouse
    )


    redraw()


    while True:

        key = cv2.waitKey(20)


        if key == 13:
            break


        if key == 27:

            selected.clear()
            break


    cv2.destroyWindow(
        "choose"
    )


    return selected

def create_feature_points(
        gray,
        centers
):

    points = []

    for c in centers:

        x,y = c.astype(int)

        mask = np.zeros(
            gray.shape,
            np.uint8
        )

        cv2.circle(
            mask,
            (x,y),
            30,
            255,
            -1
        )

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )

        if pts is not None:

            points.append(
                pts
            )

        else:

            points.append(
                np.array(
                    [[[x,y]]],
                    np.float32
                )
            )


    return points



def main():

    cap = cv2.VideoCapture(
        str(VIDEO)
    )

    ret, frame = cap.read()

    if not ret:
        return


    centers = choose_points(
        frame
    )


    gray_old = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    feature_points = create_feature_points(
        gray_old,
        centers
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    w = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    h = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (w,h)
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        new_centers = []


        for pts in feature_points:

            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                gray_old,
                gray,
                pts,
                None,
                winSize=(
                    LK_WINDOW,
                    LK_WINDOW
                ),
                maxLevel=MAX_LEVEL
            )


            good = next_pts[
                status == 1
            ]


            if len(good):

                center = np.mean(
                    good,
                    axis=0
                )

            else:

                center = np.mean(
                    pts.reshape(-1,2),
                    axis=0
                )


            new_centers.append(
                center
            )


            for p in good:

                x,y = p.astype(int)

                cv2.circle(
                    frame,
                    (x,y),
                    2,
                    (255,0,0),
                    -1
                )


            cv2.circle(
                frame,
                tuple(
                    center.astype(int)
                ),
                8,
                (0,0,255),
                -1
            )


        feature_points = [
            p.reshape(-1,1,2)
            for p in new_centers
        ]


        gray_old = gray


        writer.write(
            frame
        )


        cv2.imshow(
            "LK",
            frame
        )

        if cv2.waitKey(1)==27:
            break


    cap.release()
    writer.release()
    cv2.destroyAllWindows()



main()