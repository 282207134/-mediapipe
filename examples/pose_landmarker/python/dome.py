import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_landmarks_on_image(rgb_image, pose_landmarks):
    annotated_image = np.copy(rgb_image)

    # Check if pose landmarks are detected
    if pose_landmarks:
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks.landmark
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image

# 初始化 Mediapipe Pose 模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 读取摄像头
cap = cv2.VideoCapture(0)
#在上面代码后添加下面代码可更改分辨率
frame_width, frame_height = 1920, 1080  # 设置为你想要的分辨率
cap.set(3, frame_width)
cap.set(4, frame_height)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # 处理图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # 获取姿势关键点
    pose_landmarks = results.pose_landmarks

    # 在图像上绘制姿势关键点
    annotated_frame = draw_landmarks_on_image(rgb_frame, pose_landmarks)

    # 将图像显示出来
    cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
