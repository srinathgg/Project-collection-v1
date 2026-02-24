import cv2
import mediapipe as mp
import numpy as np
import time

mp_selfie = mp.solutions.selfie_segmentation
segment = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

trail_history = []
max_trails = 30

prev_center = None
shockwave_radius = 0

def calculate_speed(p1, p2):
    if p1 is None or p2 is None:
        return 0
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segment.process(rgb)

    mask = result.segmentation_mask
    mask = (mask > 0.6).astype(np.uint8) * 255

    person = cv2.bitwise_and(frame, frame, mask=mask)

    # --- Find body center for speed detection ---
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        center = (cx, cy)
    else:
        center = None

    speed = calculate_speed(prev_center, center)
    prev_center = center

    trail_history.append(person.copy())
    if len(trail_history) > max_trails:
        trail_history.pop(0)

    output = frame.copy()

    # --- Draw velocity trails ---
    for i, trail in enumerate(trail_history):
        fade = (i + 1) / len(trail_history)
        alpha = fade * 0.5

        # Add slight blur to older trails
        if i < len(trail_history) - 5:
            trail = cv2.GaussianBlur(trail, (15, 15), 0)

        output = cv2.addWeighted(output, 1, trail, alpha, 0)

    # --- Anime Dash Effect ---
    if speed > 25:
        streak = cv2.GaussianBlur(person, (45, 45), 0)
        output = cv2.addWeighted(output, 1, streak, 0.4, 0)

    # --- Shockwave on sudden stop ---
    if speed < 5 and shockwave_radius == 0:
        shockwave_radius = 10

    if shockwave_radius > 0 and center is not None:
        cv2.circle(output, center, shockwave_radius, (255, 255, 255), 2)
        shockwave_radius += 15
        if shockwave_radius > 400:
            shockwave_radius = 0

    cv2.imshow("⚡ VELOCITY HYBRID MODE ⚡", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
