import cv2
import numpy as np
import time

# =========================
# INITIALIZE CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Laptop camera not found")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# LOAD PERSON DETECTOR
# =========================
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# =========================
# VARIABLES
# =========================
autonomous = False
recording = False
writer = None

dx, dy = 0, 0
last_time = time.time()

print("""
CONTROLS:
W A S D  -> Move Drone
F        -> Autonomous Follow
R        -> Record Video
Q        -> Quit
""")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))
    h, w = frame.shape[:2]

    # FPS
    current_time = time.time()
    fps = int(1 / (current_time - last_time))
    last_time = current_time

    # =========================
    # PERSON DETECTION
    # =========================
    boxes, _ = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    target = None
    if len(boxes) > 0:
        target = max(boxes, key=lambda b: b[2] * b[3])
        x, y, bw, bh = target
        cx = x + bw // 2
        cy = y + bh // 2

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        if autonomous:
            error_x = cx - w // 2
            error_y = cy - h // 2
            dx = -error_x // 50
            dy = -error_y // 50

    # =========================
    # MANUAL CONTROL
    # =========================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        dy -= 5
    elif key == ord('s'):
        dy += 5
    elif key == ord('a'):
        dx -= 5
    elif key == ord('d'):
        dx += 5
    elif key == ord('f'):
        autonomous = not autonomous
    elif key == ord('r'):
        recording = not recording
        if recording:
            writer = cv2.VideoWriter(
                "drone_record.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,
                (w, h)
            )
        else:
            writer.release()
    elif key == ord('q'):
        break

    # =========================
    # OVERLAY TELEMETRY
    # =========================
    cv2.putText(frame, f"FPS: {fps}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"Mode: {'AUTO' if autonomous else 'MANUAL'}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Velocity X:{dx} Y:{dy}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if recording:
        cv2.putText(frame, "REC", (w - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        writer.write(frame)

    cv2.imshow("Laptop Drone Vision System", frame)

# =========================
# CLEANUP
# =========================
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
