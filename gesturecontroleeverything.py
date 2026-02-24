import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

prev_x = None
cooldown = 0

def fingers_up(lm):
    tips = [8, 12, 16, 20]
    return [lm[tip].y < lm[tip - 2].y for tip in tips]

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # -------- MOUSE MOVE --------
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        mx = np.interp(ix, [0, w], [0, screen_w])
        my = np.interp(iy, [0, h], [0, screen_h])
        pyautogui.moveTo(mx, my, duration=0.05)

        # -------- PINCH CLICK --------
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)
        distance = np.hypot(ix - tx, iy - ty)

        if distance < 30:
            pyautogui.click()
            time.sleep(0.25)

        fingers = fingers_up(lm)

        # -------- VOLUME --------
        if time.time() > cooldown:
            if fingers == [True, True, True, True]:
                pyautogui.press("volumeup")
                cooldown = time.time() + 0.3

            elif fingers == [False, False, False, False]:
                pyautogui.press("volumedown")
                cooldown = time.time() + 0.3

        # -------- SWIPE MUSIC --------
        if prev_x is not None:
            diff = ix - prev_x

            if diff > 120:
                pyautogui.press("nexttrack")
                time.sleep(0.5)

            elif diff < -120:
                pyautogui.press("prevtrack")
                time.sleep(0.5)

        prev_x = ix

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
