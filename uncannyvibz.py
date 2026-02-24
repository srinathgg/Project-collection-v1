import cv2
import numpy as np

# ---------------- INIT ----------------
cap = cv2.VideoCapture(0)

trail_canvas = None
DECAY = 0.90        # lower = longer trails (try 0.88–0.93)
ADD_STRENGTH = 0.25 # how strong new frames add in

prev_gray = None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if trail_canvas is None:
        trail_canvas = frame.astype(np.float32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------- MOTION DETECTION ----------------
    motion_strength = 0
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion_strength = np.mean(diff)

    prev_gray = gray.copy()

    # ---------------- DECAY OLD TRAILS ----------------
    trail_canvas *= DECAY

    # ---------------- ADD NEW FRAME BASED ON MOTION ----------------
    motion_factor = np.clip(motion_strength / 25.0, 0.05, 1.0)
    trail_canvas += frame.astype(np.float32) * (ADD_STRENGTH * motion_factor)

    # ---------------- FINAL COMPOSITE ----------------
    output = np.clip(trail_canvas, 0, 255).astype(np.uint8)

    # subtle desaturation (aesthetic)
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = (hsv[..., 1] * 0.85).astype(np.uint8)
    output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("TIME TRAILS", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
