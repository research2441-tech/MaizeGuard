"""Field-like acquisition degradations (Gaussian blur, defocus/motion, low light, JPEG, sensor noise, glare)."""
import io, numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def apply_corruption(im, kind, level, seed=0):
    rng = np.random.default_rng(seed)
    w, h = im.size; s = max(w, h) / 640.0  # scale-aware strength
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(radius=[1.5, 3.0, 5.0][level - 1] * s))
    if kind == "motion":
        k = int([7, 13, 21][level - 1] * s) | 1
        a = np.asarray(im, np.float32); out = np.zeros_like(a)
        for i in range(k): out += np.roll(a, i - k // 2, axis=1)
        return Image.fromarray(np.clip(out / k, 0, 255).astype(np.uint8))
    if kind == "dark":
        f = [0.45, 0.28, 0.15][level - 1]
        a = np.asarray(im, np.float32) / 255.
        a = (a ** 1.2) * f + rng.normal(0, [0.01, 0.02, 0.03][level - 1], a.shape)
        return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        b = io.BytesIO(); im.save(b, "JPEG", quality=[20, 10, 5][level - 1]); b.seek(0)
        return Image.open(b).convert("RGB")
    if kind == "noise":
        a = np.asarray(im, np.float32) / 255.
        a = a + rng.normal(0, [0.06, 0.12, 0.20][level - 1], a.shape)
        return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    if kind == "glare":
        a = np.asarray(im, np.float32) / 255.
        yy, xx = np.mgrid[0:h, 0:w]; cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = [0.25, 0.4, 0.6][level - 1] * max(w, h)
        m = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
        a = a * (1 - 0.8 * m) + 0.95 * m
        return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    raise ValueError(kind)
