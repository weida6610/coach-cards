"""HEIC/JPG -> high-quality JPG, long edge 1600px, q=92, EXIF orientation respected."""
from pathlib import Path
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

SRC = Path(r"C:\Users\User\coach-cards\photos")
MAX_EDGE = 1600
QUALITY = 92

for p in sorted(SRC.iterdir()):
    if p.suffix.lower() not in {".heic", ".jpg", ".jpeg"}:
        continue
    if p.stem.endswith("_web"):
        continue
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)        # honor EXIF orientation
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    out = SRC / f"{p.stem}_web.jpg"
    img.save(out, "JPEG", quality=QUALITY, optimize=True, progressive=True)
    print(f"OK  {p.name} -> {out.name}  {img.size[0]}x{img.size[1]}  ({out.stat().st_size//1024} KB)")
