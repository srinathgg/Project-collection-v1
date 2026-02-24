import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# FORCE HD CAMERA
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

style = 1
particles = []
shake_strength = 0

def enhance_frame(frame):
    # Sharpen
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    frame = cv2.filter2D(frame, -1, kernel)

    # Contrast boost
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    # Noise reduction
    frame = cv2.bilateralFilter(frame, 5, 50, 50)

    return frame

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers += 1
    return fingers

def add_fire_particles(frame, cx, cy):
    global particles
    for _ in range(8):
        particles.append([cx, cy, random.uniform(-2,2),
                          random.uniform(-6,-2),
                          random.randint(6,12)])

    new_particles = []
    for p in particles:
        p[0] += p[2]
        p[1] += p[3]
        p[4] -= 0.4

        if p[4] > 0:
            cv2.circle(frame, (int(p[0]), int(p[1])),
                       int(p[4]), (0,140,255), -1)
            new_particles.append(p)

    particles = new_particles

def branching_lightning(frame):
    h, w = frame.shape[:2]
    for _ in range(3):
        x = random.randint(0, w)
        y = 0
        for _ in range(8):
            nx = x + random.randint(-40,40)
            ny = y + random.randint(40,100)
            cv2.line(frame, (x,y),(nx,ny),(255,255,255),2)
            x, y = nx, ny

def water_ripple(frame, cx, cy):
    rows, cols = frame.shape[:2]
    ripple = frame.copy()
    for i in range(rows):
        for j in range(cols):
            dx = j - cx
            dy = i - cy
            dist = math.sqrt(dx*dx + dy*dy)
            offset = int(5 * math.sin(dist/15 - time.time()*4))
            new_x = min(max(j + offset, 0), cols-1)
            ripple[i,j] = frame[i,new_x]
    return ripple

def apply_screen_shake(frame):
    dx = random.randint(-shake_strength, shake_strength)
    dy = random.randint(-shake_strength, shake_strength)
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    element = ""
    cx, cy = 0, 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(handLms)
            h, w = frame.shape[:2]
            cx = int(handLms.landmark[8].x * w)
            cy = int(handLms.landmark[8].y * h)

            if fingers == 1:
                element = "FIRE"
            elif fingers == 2:
                element = "LIGHTNING"
            elif fingers == 3:
                element = "WATER"
            elif fingers == 0:
                element = "EARTH"

    if element == "FIRE":
        add_fire_particles(frame, cx, cy)

    elif element == "LIGHTNING":
        branching_lightning(frame)

    elif element == "WATER":
        frame = water_ripple(frame, cx, cy)

    elif element == "EARTH":
        shake_strength = 15
        frame = apply_screen_shake(frame)
    else:
        shake_strength = 0

    # Glow effect
    glow = cv2.GaussianBlur(frame, (15,15), 0)
    frame = cv2.addWeighted(frame, 0.8, glow, 0.2, 0)

    cv2.putText(frame, f"{element}", (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0,255,255), 3)

    cv2.imshow("ULTIMATE ELEMENT BENDER HD", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
import mediapipe as mp
import numpy as np

prev_x, prev_y = 0, 0
smooth_factor = 0.7

# smoothing formula
x = int(prev_x * smooth_factor + current_x * (1 - smooth_factor))
y = int(prev_y * smooth_factor + current_y * (1 - smooth_factor))

prev_x, prev_y = x, y
