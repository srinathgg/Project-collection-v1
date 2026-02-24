import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

canvas = None
prev_point = None
drawing = False

# ---------------- HELPERS ----------------
def pinch(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]
    return math.dist((thumb.x, thumb.y), (index.x, index.y)) < 0.04

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            index_tip = hand_landmarks.landmark[8]
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            if pinch(hand_landmarks.landmark):
                drawing = True
            else:
                drawing = False
                prev_point = None

            if drawing:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (x, y), (255, 255, 255), 4)
                prev_point = (x, y)

            # draw cursor
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # open palm clears
            wrist = hand_landmarks.landmark[0]
            if math.dist((wrist.x, wrist.y), (index_tip.x, index_tip.y)) > 0.25:
                canvas = np.zeros_like(frame)

    # merge drawing with frame
    output = cv2.addWeighted(frame, 1, canvas, 1, 0)

    cv2.imshow("AIR DRAW", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
