import cv2
import mediapipe as mp
import numpy as np
# 初始化 Mediapipe Hands 模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# 读取摄像头
cap = cv2.VideoCapture(0)
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


# 初始化上一帧图像
annotated_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

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
    annotated_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # 清空上一帧图像
    for hand_landmarks in hand_landmarks_list:
        for i, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
            cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)  # 绘制关键点
            if i > 0:
                prev_x, prev_y = int(hand_landmarks.landmark[i - 1].x * frame_width), int(hand_landmarks.landmark[i - 1].y * frame_height)
                cv2.line(annotated_frame, (prev_x, prev_y), (x, y), (255, 0, 0), 2)  # 绘制连接线

    # 如果没有检测到手，依然显示上一帧的图像
    cv2.imshow("Hand Tracking", annotated_frame)

    # 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
