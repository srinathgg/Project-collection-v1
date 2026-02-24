import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)

glow_radius = 0
max_glow = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Index fingertip is landmark 8
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            glow_radius = min(glow_radius + 3, max_glow)

            # Create glow effect
            for i in range(5):
                radius = glow_radius + i * 10
                alpha = max(0, 255 - i * 50)
                overlay = frame.copy()
                cv2.circle(overlay, (x, y), radius, (255, 200, 0), -1)
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

    else:
        # Fade out effect
        glow_radius = max(glow_radius - 5, 0)

    cv2.imshow("Finger Glow", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()