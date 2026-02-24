import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time

# ---------------- INIT ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

particles = []
shockwaves = []
velocity_trails = []

left_energy = 0
right_energy = 0
last_left = None
last_right = None
last_dist = None
last_supernova = 0
blackhole_active = False
spin_force = 0.3

# ---------------- GESTURES ----------------
def pinch(lm):
    return math.dist((lm[4].x,lm[4].y),(lm[8].x,lm[8].y)) < 0.04

def open_hand(lm):
    return math.dist((lm[0].x,lm[0].y),(lm[8].x,lm[8].y)) > 0.17

# ---------------- PARTICLES (3D FAKE DEPTH) ----------------
def spawn_particles(center,color,n=8):
    for _ in range(n):
        angle=random.uniform(0,2*np.pi)
        speed=random.uniform(1,4)
        depth=random.uniform(0.5,2.0)
        particles.append({
            "pos":np.array(center,float),
            "vel":np.array([math.cos(angle)*speed,math.sin(angle)*speed]),
            "z":depth,
            "life":random.randint(60,120),
            "color":color
        })

def update_particles(frame,gravity=None):
    global particles
    h,w,_=frame.shape
    new=[]
    for p in particles:
        if gravity is not None:
            d=gravity-p["pos"]
            dist=np.linalg.norm(d)+1
            d/=dist
            tangent=np.array([-d[1],d[0]])
            p["vel"]+=d*0.4+tangent*spin_force

        p["pos"]+=p["vel"]
        p["vel"]*=0.98
        p["life"]-=1

        if p["life"]<=0:
            continue

        scale=int(3/p["z"])
        x,y=int(p["pos"][0]),int(p["pos"][1])

        if 0<=x<w and 0<=y<h:
            cv2.circle(frame,(x,y),scale,p["color"],-1)

        new.append(p)

    particles=new

# ---------------- BODY VELOCITY TRAIL ----------------
def update_velocity_trail(frame,mask):
    global velocity_trails
    velocity_trails.append(mask.copy())
    if len(velocity_trails) > 18:
        velocity_trails.pop(0)

    for i,trail in enumerate(velocity_trails):
        alpha = i / len(velocity_trails) * 0.5
        colored = cv2.cvtColor(trail, cv2.COLOR_GRAY2BGR)
        frame = cv2.addWeighted(frame,1,colored,alpha,0)

    return frame

# ---------------- FX ----------------
def glow(frame,center,energy,color):
    radius=int(40+energy*0.8)
    overlay=frame.copy()
    for r in range(radius,0,-8):
        alpha=(r/radius)*0.25
        cv2.circle(overlay,center,r,color,-1)
        frame=cv2.addWeighted(overlay,alpha,frame,1-alpha,0)
    return frame

def bloom(frame):
    blur=cv2.GaussianBlur(frame,(0,0),15)
    return cv2.addWeighted(frame,1,blur,0.5,0)

def shock(center):
    shockwaves.append({"c":center,"r":10})

def update_shocks(frame):
    global shockwaves
    new=[]
    for s in shockwaves:
        s["r"]+=15
        if s["r"]<400:
            cv2.circle(frame,s["c"],s["r"],(255,255,255),2)
            new.append(s)
    shockwaves=new

def ribbon_beam(frame,a,b):
    steps=20
    for i in range(steps):
        t=i/steps
        x=int(a[0]*(1-t)+b[0]*t + math.sin(time.time()*5+t*10)*10)
        y=int(a[1]*(1-t)+b[1]*t)
        cv2.circle(frame,(x,y),4,(255,255,255),-1)

def blackhole(frame,center):
    overlay=frame.copy()
    cv2.circle(overlay,center,90,(0,0,0),-1)
    cv2.circle(overlay,center,130,(80,80,80),2)
    return cv2.addWeighted(overlay,0.85,frame,0.15,0)

def lens_distortion(frame,center):
    h,w=frame.shape[:2]
    y,x=np.indices((h,w),np.float32)
    dx,dy=x-center[0],y-center[1]
    dist=np.sqrt(dx*dx+dy*dy)+1
    factor=1000/(dist+80)
    x+=dx/dist*factor
    y+=dy/dist*factor
    return cv2.remap(frame,x,y,cv2.INTER_LINEAR)

# ---------------- CAMERA ENHANCE ----------------
def enhance_camera(frame):
    frame=cv2.resize(frame,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_CUBIC)
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    frame=cv2.filter2D(frame,-1,kernel)
    return frame

# ---------------- MAIN LOOP ----------------
while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.flip(frame,1)
    frame=enhance_camera(frame)
    h,w,_=frame.shape

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res=hands.process(rgb)

    left=None
    right=None

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm,hd in zip(res.multi_hand_landmarks,res.multi_handedness):
            pts=lm.landmark
            cx,cy=int(pts[0].x*w),int(pts[0].y*h)

            if pinch(pts):
                if hd.classification[0].label=="Left":
                    left=(cx,cy)
                    left_energy=min(left_energy+3,120)
                else:
                    right=(cx,cy)
                    right_energy=min(right_energy+3,120)

            if open_hand(pts):
                if left_energy>40:
                    shock((cx,cy))
                    left_energy=0
                if right_energy>40:
                    shock((cx,cy))
                    right_energy=0

    gravity=None

    if left:
        frame=glow(frame,left,left_energy,(255,180,100))
        spawn_particles(left,(255,200,150))

    if right:
        frame=glow(frame,right,right_energy,(120,200,255))
        spawn_particles(right,(180,230,255))

    if left and right:
        ribbon_beam(frame,left,right)
        d=math.dist(left,right)

        if last_dist and d-last_dist>70 and time.time()-last_supernova>1:
            shock(((left[0]+right[0])//2,(left[1]+right[1])//2))
            last_supernova=time.time()

        last_dist=d

        if d<180:
            center=((left[0]+right[0])//2,(left[1]+right[1])//2)
            gravity=np.array(center)
            frame=blackhole(frame,center)
            frame=lens_distortion(frame,center)

    update_particles(frame,gravity)
    update_shocks(frame)
    frame=bloom(frame)

    # Velocity mask from motion
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(21,21),0)
    if 'prev' not in globals():
        prev=blur
    diff=cv2.absdiff(prev,blur)
    _,mask=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)
    prev=blur
    frame=update_velocity_trail(frame,mask)

    cv2.imshow("REALITY BREAKER ENGINE 3D",frame)

    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
