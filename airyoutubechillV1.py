import cv2
import mediapipe as mp
import numpy as np

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# MEDIAPIPE
# =========================
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.6)

# =========================
# MODES
# =========================
black_bg = False
blur_bg = False
neon = False
cinema = False

print("""
CONTROLS:
B -> Black Background
L -> Blur Background
N -> Neon Outline
C -> Cinema Auto Crop
Q -> Quit
""")

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # =========================
    # LOW LIGHT BOOST
    # =========================
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    frame = cv2.merge((l, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    # =========================
    # SEGMENTATION
    # =========================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmenter.process(rgb)
    mask = result.segmentation_mask
    condition = mask > 0.5

    output = frame.copy()

    if black_bg:
        black = np.zeros((h, w, 3), dtype=np.uint8)
        output = np.where(condition[:, :, None], output, black)

    if blur_bg:
        blur = cv2.GaussianBlur(frame, (55, 55), 0)
        output = np.where(condition[:, :, None], output, blur)

    # =========================
    # NEON OUTLINE
    # =========================
    if neon:
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 100, 200)
        glow = cv2.dilate(edges, None, iterations=2)
        output[glow > 0] = (0, 255, 255)  # neon cyan

    # =========================
    # CINEMA AUTO CROP
    # =========================
    if cinema:
        faces = face_detector.process(rgb)
        if faces.detections:
            d = faces.detections[0]
            box = d.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            pad = 100
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            output = output[y1:y2, x1:x2]
            output = cv2.resize(output, (w, h))

    # =========================
    # UI
    # =========================
    cv2.putText(output, "CHILL MODE ACTIVE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Chill Stream Cam", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        black_bg = not black_bg
    elif key == ord('l'):
        blur_bg = not blur_bg
    elif key == ord('n'):
        neon = not neon
    elif key == ord('c'):
        cinema = not cinema
    elif key == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
