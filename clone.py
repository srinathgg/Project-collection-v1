import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
from collections import deque

# ================= AUDIO SETUP =================

SAMPLE_RATE = 44100
pygame.mixer.pre_init(SAMPLE_RATE, size=-16, channels=2)
pygame.mixer.init()

def generate_tone(freq, duration=0.12, volume=0.5):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * 2 * np.pi * t)

    audio = wave * (2**15 - 1) * volume
    audio = audio.astype(np.int16)

    # Convert mono → stereo (2D array)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)

# ================= VIDEO SETUP =================

cap = cv2.VideoCapture(0)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

buffer = deque(maxlen=30)
last_note_time = 0
NOTE_INTERVAL = 0.10  # seconds between notes

# ================= MAIN LOOP =================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)

    mask = (result.segmentation_mask > 0.6).astype(np.uint8)
    body = cv2.bitwise_and(frame, frame, mask=mask)
    buffer.append(body.copy())

    # ===== Get body center =====
    ys, xs = np.where(mask == 1)

    if len(xs) > 100:
        cx = int(np.mean(xs))

        # Pitch from horizontal position
        pitch = np.interp(cx, [0, w], [220, 880])

        # Motion energy (better version)
        gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)
        motion_energy = np.mean(gray)
        volume = np.clip(motion_energy / 100, 0.1, 0.7)

        current_time = time.time()

        # ===== Play main note =====
        if current_time - last_note_time > NOTE_INTERVAL:
            sound = generate_tone(pitch, volume=volume)
            sound.play()
            last_note_time = current_time

        # ===== Clone harmony =====
        if len(buffer) > 10:
            clone = buffer[-10]
            cgray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
            clone_energy = np.mean(cgray)

            if clone_energy > 20:
                harmony_pitch = pitch * 1.5
                harmony = generate_tone(harmony_pitch, volume=0.3)
                harmony.play()

    cv2.imshow("Reality Instrument", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
