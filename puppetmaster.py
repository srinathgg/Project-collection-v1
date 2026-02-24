import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ---------------- DISTORTION FUNCTION ----------------
def blob_distortion(frame, center, strength):
    h, w = frame.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    dx = x - center[0]
    dy = y - center[1]
    dist = np.sqrt(dx**2 + dy**2)

    radius = 200 + strength * 2

    mask = dist < radius
    factor = (radius - dist) / radius
    factor = np.clip(factor, 0, 1)

    x[mask] += dx[mask] * factor[mask] * 0.3
    y[mask] += dy[mask] * factor[mask] * 0.3

    distorted = cv2.remap(frame, x, y, cv2.INTER_LINEAR)
    return distorted

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    center_screen = (w // 2, h // 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left_hand_pos = None

    if result.multi_hand_landmarks and result.multi_handedness:
        for lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label

            if label == "Left":
                wrist = lm.landmark[0]
                x = int(wrist.x * w)
                y = int(wrist.y * h)
                left_hand_pos = (x, y)

    # ---------------- DRAW STRINGS ----------------
    if left_hand_pos:
        # draw strings from corners
        corners = [(0,0), (w,0), (0,h), (w,h)]
        for c in corners:
            cv2.line(frame, c, left_hand_pos, (255,255,255), 2)

        # measure extension distance
        dist = math.dist(left_hand_pos, center_screen)

        if dist > 120:
            strength = min(dist - 120, 150)
            frame = blob_distortion(frame, center_screen, strength)

    cv2.imshow("STRING BLOB FIELD", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
