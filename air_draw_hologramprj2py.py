import cv2
import mediapipe as mp
import numpy as np
import math
import random

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

# ---------- STATE ----------
energy_radius = 55
min_energy = 25
max_energy = 100

ring_base_radius = 90
ring_angle = 0
ring_speed = 2          # will go CRAZY on pinch

prev_x, prev_y = 0, 0
smooth = 0.25

# ---------- PARTICLES ----------
particles = []

class Particle:
    def __init__(self):
        self.angle = random.uniform(0, 2 * math.pi)
        self.radius = random.randint(40, 120)
        self.speed = random.uniform(0.002, 0.01)

    def update(self):
        self.angle += self.speed

    def pos(self, cx, cy):
        return (
            int(cx + math.cos(self.angle) * self.radius),
            int(cy + math.sin(self.angle) * self.radius)
        )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    overlay = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pinched = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        ix = int(hand.landmark[8].x * w)
        iy = int(hand.landmark[8].y * h)
        tx = int(hand.landmark[4].x * w)
        ty = int(hand.landmark[4].y * h)

        ix = int(prev_x + smooth * (ix - prev_x))
        iy = int(prev_y + smooth * (iy - prev_y))
        prev_x, prev_y = ix, iy

        pinch_dist = math.hypot(ix - tx, iy - ty)
        pinched = pinch_dist < 40

        # ---------- PHYSICS ----------
        if pinched:
            energy_radius = max(min_energy, energy_radius - 3)
            ring_speed = min(40, ring_speed + 4)  # FAST AF
        else:
            energy_radius = min(max_energy, energy_radius + 1)
            ring_speed = max(2, ring_speed - 2)

            if len(particles) < 80:
                particles.append(Particle())

        # ---------- CORE: BLACK → PURPLE → WHITE ----------
        # outer white glow
        for i in range(10):
            cv2.circle(
                overlay,
                (ix, iy),
                energy_radius + i * 8,
                (255, 255, 255),
                -1
            )

        # mid purple energy
        for i in range(6):
            cv2.circle(
                overlay,
                (ix, iy),
                int(energy_radius * 0.7) + i * 6,
                (200, 0, 255),
                -1
            )

        # inner BLACK core
        cv2.circle(
            overlay,
            (ix, iy),
            int(energy_radius * 0.45),
            (0, 0, 0),
            -1
        )

        # ---------- MULTIPLE RINGS ----------
        ring_angle += ring_speed

        ring_layers = [
            (ring_base_radius, 0.35, 2),
            (ring_base_radius + 25, 0.45, 1),
            (ring_base_radius + 50, 0.25, 2),
            (ring_base_radius + 75, 0.55, 1),
        ]

        for idx, (r, squash, thick) in enumerate(ring_layers):
            cv2.ellipse(
                overlay,
                (ix, iy),
                (r, int(r * squash)),
                ring_angle * (1 + idx * 1),
                0,
                360,
                (255, 200, 255),
                thick
            )

        # ---------- PARTICLES (FLOATING ENERGY) ----------
        if not pinched:
            for p in particles:
                p.update()
                px, py = p.pos(ix, iy)
                cv2.circle(overlay, (px, py), 2, (220, 220, 255), -1)

            if len(particles) > 4000:
                particles = particles[-3000:]

    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.imshow("Hollow Purple — OVERDRIVE", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
