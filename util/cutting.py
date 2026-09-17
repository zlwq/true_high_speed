import cv2
import os

video_path = "assets/output/22.mp4"

video_dir = os.path.dirname(video_path)
video_name = os.path.splitext(os.path.basename(video_path))[0]

output_dir = "util/output/"+video_name+"_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

index = 1

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output_path = os.path.join(
        output_dir,
        f"{index:03d}.jpg"
    )

    cv2.imwrite(output_path, frame)

    index += 1

cap.release()

print(f"完成，共保存 {index - 1} 帧")
print(f"保存位置：{output_dir}")