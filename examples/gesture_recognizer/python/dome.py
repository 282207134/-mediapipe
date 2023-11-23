import cv2
import mediapipe as mp
import numpy as np

# 初始化 Mediapipe Hands 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# 读取摄像头
cap = cv2.VideoCapture(0)

# 初始化上一帧图像
annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 使用适当的图像尺寸

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # 处理图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 初始化手部关键点列表
    hand_landmarks_list = []

    # 提取手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 在这里处理每只手的关键点逻辑
            hand_landmarks_list.append(hand_landmarks)
            # 例如，输出手腕的坐标
            print(f"Wrist coordinates: {hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x}, {hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y}")

    # 在图像上绘制手部关键点和连接
    annotated_frame = frame.copy()
    for hand_landmarks in hand_landmarks_list:
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 如果没有检测到手，依然显示上一帧的图像
    cv2.imshow("Hand Tracking", annotated_frame)

    # 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
