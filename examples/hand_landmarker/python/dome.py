import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green



def draw_landmarks_on_image(rgb_image, hand_landmarks_list, handedness_list):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        handedness_str = handedness.classification[0].label if handedness.classification else "Unknown"
        cv2.putText(annotated_image, f"{handedness_str}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

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

    # 获取手部关键点
    hand_landmarks_list = [hand.landmark for hand in results.multi_hand_landmarks] if results.multi_hand_landmarks else []
    handedness_list = results.multi_handedness

    # 在图像上绘制手部关键点和 handedness
    annotated_frame = draw_landmarks_on_image(rgb_frame, hand_landmarks_list, handedness_list)

    # 将图像显示出来
    cv2.imshow("Hand Tracking", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # 退出循环
    if cv2.waitKey(1) & 0xFF == 27:  # 按下ESC键退出
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
