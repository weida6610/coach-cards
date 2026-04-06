"""Normalize face size across all coach photos.

For each *_web.jpg in photos/:
  1. Detect the largest face with OpenCV Haar cascade
  2. Compute crop so face height = TARGET_FACE_RATIO of output height
  3. Center horizontally on face, anchor face top near TOP_MARGIN_RATIO
  4. Output to photos/{name}_norm.jpg with fixed aspect ratio (3:4 portrait)
"""
from pathlib import Path
import cv2

SRC = Path(r"C:\Users\User\coach-cards\photos")
OUT_W, OUT_H = 900, 1200            # 3:4 portrait
TARGET_FACE_RATIO = 0.22            # face height = 22% of output height
TOP_MARGIN_RATIO  = 0.13            # face top sits 13% from top of crop
QUALITY = 92

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # Pick the largest face
    return max(faces, key=lambda f: f[2] * f[3])

for p in sorted(SRC.glob("*_web.jpg")):
    img = cv2.imread(str(p))
    if img is None:
        print(f"SKIP {p.name}: cannot read")
        continue
    h, w = img.shape[:2]
    face = detect_face(img)
    if face is None:
        print(f"WARN {p.name}: no face detected, skipped")
        continue
    fx, fy, fw, fh = face

    # Desired face height in output = TARGET_FACE_RATIO * OUT_H
    desired_face_h = TARGET_FACE_RATIO * OUT_H
    scale = desired_face_h / fh                       # how much to scale source to make face the right size

    # Resize source so face matches target
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)

    # Face position in resized image
    rfx = fx * scale
    rfy = fy * scale
    rfw = fw * scale
    rfh = fh * scale
    face_cx = rfx + rfw / 2
    face_top = rfy

    # Crop window: centered on face_cx, face_top sits at TOP_MARGIN_RATIO * OUT_H
    crop_x = int(round(face_cx - OUT_W / 2))
    crop_y = int(round(face_top - TOP_MARGIN_RATIO * OUT_H))

    # Clamp inside resized bounds; if image too small, pad
    pad_l = max(0, -crop_x)
    pad_t = max(0, -crop_y)
    pad_r = max(0, (crop_x + OUT_W) - new_w)
    pad_b = max(0, (crop_y + OUT_H) - new_h)
    if pad_l or pad_t or pad_r or pad_b:
        resized = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_REPLICATE)
        crop_x += pad_l
        crop_y += pad_t

    crop = resized[crop_y:crop_y + OUT_H, crop_x:crop_x + OUT_W]
    out_path = SRC / p.name.replace("_web.jpg", "_norm.jpg")
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, QUALITY,
                                      cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
    print(f"OK   {p.name} -> {out_path.name}  face={fw}x{fh}  scale={scale:.2f}  out={OUT_W}x{OUT_H}")
