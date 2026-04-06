"""Remove DeeVid AI watermark from top-right of Victor_web-2.jpg via inpainting."""
from pathlib import Path
import cv2
import numpy as np

SRC = Path(r"C:\Users\User\coach-cards\photos\Victor_web-2.jpg")
DST = Path(r"C:\Users\User\coach-cards\photos\Victor_web.jpg")

img = cv2.imread(str(SRC))
h, w = img.shape[:2]   # 2400 x 1792

# Watermark is roughly top 5% height, right 32% width (generous margin)
mask = np.zeros((h, w), dtype=np.uint8)
y0 = int(h * 0.005)
y1 = int(h * 0.055)
x0 = int(w * 0.66)
x1 = int(w * 0.995)
mask[y0:y1, x0:x1] = 255

# TELEA inpainting works well for small uniform areas
out = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
cv2.imwrite(str(DST), out, [cv2.IMWRITE_JPEG_QUALITY, 92, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
print(f"OK -> {DST.name}  mask=({x0},{y0})-({x1},{y1})")
