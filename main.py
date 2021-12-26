import mediapipe as mp
import numpy as np
from cv2 import cv2

img1 = cv2.imread("scheme.png")
cv2.imshow("Prompting", img1)
cv2.waitKey(0)


def get_points(landmark, shape):
    points = list()
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)


def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def stone(r, ws):
    if not (2 * r / ws > 1.3):
        return True
    return False


def table():
    return False


def bird():
    return False


def water():
    return False


def gun():
    return False


handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(flippedRGB, 'Show one of 5 gestures (stone, table,', (10, 30), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(flippedRGB, 'water, bird, table)', (350, 60), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(flippedRGB, 'Program recognizes it', (10, 90), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)

    if results.multi_hand_landmarks is not None:
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)

        if stone(r, ws):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flippedRGB, 'stone', (10, 200), font, 4, (255, 0, 0), 2, cv2.LINE_AA)

        if table():
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flippedRGB, 'table', (10, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        if water():
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flippedRGB, 'water', (10, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        if bird():
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flippedRGB, 'bird', (10, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        if gun():
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(flippedRGB, 'gun', (10, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
