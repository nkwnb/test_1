import cv2
import os

# 输入文件夹路径，包含多个 .mov 视频
input_folder = 'D:\chicken data\立华\立华手机拍摄\母鸡视频'
# 输出文件夹路径，用于保存提取的帧
output_folder = 'D:\chicken data\lihua1'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有视频文件
video_files = [f for f in os.listdir(input_folder) if f.endswith('.mov')]

frame_count = 361  # 用于命名输出图像（第一张照片名称）

# 遍历每个视频文件
for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        continue

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 10 帧提取一张图片
        if frame_idx % 50 == 0:
            output_path = os.path.join(output_folder, f"{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1

        frame_idx += 1

    cap.release()

print("帧提取完成！")