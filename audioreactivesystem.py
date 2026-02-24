import pygame
import numpy as np
import sounddevice as sd
import random
import math
import os

# =========================
# CONFIG
# =========================
WIDTH, HEIGHT = 1280, 800
FPS = 60
BLOCK = 1024
RATE = 44100

# =========================
# AUDIO
# =========================
audio_buffer = np.zeros(BLOCK)
gain = 1.0
energy_smooth = 0

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0].copy()

stream = sd.InputStream(
    channels=1,
    samplerate=RATE,
    blocksize=BLOCK,
    callback=audio_callback
)

# =========================
# PARTICLES (3D PARALLAX)
# =========================
class Particle:
    def __init__(self, depth):
        self.depth = depth
        self.angle = random.uniform(0, math.pi*2)
        self.radius = random.randint(150, 500)
        self.speed = random.uniform(0.002, 0.01) * (1/depth)
        self.size = max(1, int(5/depth))

    def update(self, high):
        self.angle += self.speed + high * 0.00003

    def draw(self, screen, center, hue):
        x = int(center[0] + math.cos(self.angle) * self.radius)
        y = int(center[1] + math.sin(self.angle) * self.radius)
        c = pygame.Color(0)
        c.hsva = (hue, 80, 100 - self.depth*20, 100)
        pygame.draw.circle(screen, c, (x,y), self.size)

# =========================
# INIT
# =========================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🔥 AUDIO GOD MODE 🔥")
clock = pygame.time.Clock()

stream.start()

particles = []
for d in [1, 1.8, 3]:
    for _ in range(150):
        particles.append(Particle(d))

trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

t = 0
beat_bass = 0
shockwave = 0
recording = False
frame_id = 0
rotation = 0

if not os.path.exists("frames"):
    os.mkdir("frames")

running = True

# =========================
# MAIN LOOP
# =========================
while running:
    clock.tick(FPS)
    t += 1
    rotation += 0.002

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()
            if event.key == pygame.K_r:
                recording = not recording

    trail.fill((0,0,0,25))
    screen.blit(trail, (0,0))

    # ================= AUDIO ANALYSIS =================
    fft = np.abs(np.fft.rfft(audio_buffer))

    bass = np.mean(fft[2:20])
    mids = np.mean(fft[20:80])
    highs = np.mean(fft[80:180])

    energy = bass * 0.006
    energy_smooth = energy_smooth*0.85 + energy*0.15

    if energy_smooth > 1:
        gain *= 0.995
    else:
        gain *= 1.002

    gain = max(0.5, min(gain, 3))
    energy_smooth *= gain

    if bass > 200:
        beat_bass = 8
        shockwave = 0

    center = (WIDTH//2, HEIGHT//2)

    # ================= MOOD COLOR =================
    dominant = max(bass, mids, highs)
    if dominant == bass:
        base_hue = 0
    elif dominant == mids:
        base_hue = 120
    else:
        base_hue = 240

    hue = (base_hue + t) % 360

    # ================= METABALL CORE =================
    points = []
    base_radius = 150 + energy_smooth*250

    for i in range(80):
        angle = i/80 * math.pi*2
        wobble = math.sin(angle*6 + t*0.05) * 30
        r = base_radius + wobble
        x = center[0] + math.cos(angle+rotation)*r
        y = center[1] + math.sin(angle+rotation)*r
        points.append((int(x), int(y)))

    c = pygame.Color(0)
    c.hsva = (hue, 90, 100, 100)
    pygame.draw.polygon(screen, c, points)

    # ================= RADIAL SPECTRUM RING =================
    for i in range(100):
        value = fft[i*2] * 0.01
        length = min(value, 200)

        angle = i/100 * math.pi*2 + rotation
        x1 = int(center[0] + math.cos(angle)*(base_radius+40))
        y1 = int(center[1] + math.sin(angle)*(base_radius+40))
        x2 = int(center[0] + math.cos(angle)*(base_radius+40+length))
        y2 = int(center[1] + math.sin(angle)*(base_radius+40+length))

        rc = pygame.Color(0)
        rc.hsva = ((hue+i*3)%360, 90, 100, 100)
        pygame.draw.line(screen, rc, (x1,y1), (x2,y2), 2)

    # ================= PARTICLES =================
    for p in particles:
        p.update(highs)
        p.draw(screen, center, hue)

    # ================= SHOCKWAVE =================
    if beat_bass > 0:
        shockwave += 20
        pygame.draw.circle(screen, (255,255,255),
                           center, int(base_radius+shockwave), 3)
        beat_bass -= 1

    # ================= SCREEN FLASH =================
    if beat_bass > 0:
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(40)
        overlay.fill((255,255,255))
        screen.blit(overlay, (0,0))

    pygame.display.flip()

    if recording:
        pygame.image.save(screen, f"frames/frame_{frame_id:05}.png")
        frame_id += 1

# ================= CLEANUP =================
stream.stop()
pygame.quit()
