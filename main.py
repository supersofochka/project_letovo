import mediapipe as mp
import numpy as np
from cv2 import cv2


def get_points(landmark, shape):
    points = list()
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        cv2.drawContours(flippedRGB, [get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)], 0,
                         (255, 0, 0), 2)
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        if 2 * r / ws > 1.3:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 0, 255), 2)
        else:
            cv2.circle(flippedRGB, (int(x), int(y)), int(r), (0, 255, 0), 2)
            print('кулак, то есть камень')
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
