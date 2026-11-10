"""Image preprocessing and handcrafted forensic feature extraction."""
from __future__ import annotations

from pathlib import Path
import io

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 256
ELA_QUALITY = 90
LBP_BINS = 256


def load_image(path: str | Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """Read image as RGB, resize with aspect ratio preserved, and pad to square."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = image
    return canvas


def ela_map(rgb: np.ndarray, quality: int = ELA_QUALITY) -> np.ndarray:
    """Return a normalized RGB ELA difference map after JPEG recompression."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    diff = cv2.absdiff(rgb, recompressed)
    peak = float(np.percentile(diff, 99.5))
    return np.clip(diff.astype(np.float32) * (255.0 / max(peak, 1.0)), 0, 255).astype(np.uint8)


def _lbp(gray: np.ndarray) -> np.ndarray:
    """8-neighbor, radius-one LBP with a fixed clockwise bit ordering."""
    p = np.pad(gray, 1, mode="edge")
    center = p[1:-1, 1:-1]
    result = np.zeros(gray.shape, dtype=np.uint8)
    neighbors = ((0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1))
    for bit, (dy, dx) in enumerate(neighbors):
        result |= ((p[1 + dy:1 + dy + gray.shape[0], 1 + dx:1 + dx + gray.shape[1]] >= center).astype(np.uint8) << bit)
    return result


def _block_stats(arr: np.ndarray, blocks: int = 4) -> list[float]:
    values: list[float] = []
    h, w = arr.shape
    for by in range(blocks):
        for bx in range(blocks):
            tile = arr[by*h//blocks:(by+1)*h//blocks, bx*w//blocks:(bx+1)*w//blocks].astype(np.float32)
            values.extend([float(tile.mean()), float(tile.std()), float(np.percentile(tile, 90))])
    return values


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """Build a deterministic vector from ELA, DCT, LBP, and color statistics."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels")
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ela = ela_map(rgb)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)

    # DCT energy over log-magnitude frequency tiles, excluding the DC coefficient.
    dct = cv2.dct(gray.astype(np.float32) / 255.0)
    dct = np.log1p(np.abs(dct))
    dct_features = _block_stats(dct, blocks=4)

    lbp = _lbp(gray)
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, 256), density=False)
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)

    color_features: list[float] = []
    for channel in cv2.split(rgb):
        color_features.extend([float(channel.mean()) / 255, float(channel.std()) / 255,
                               float(np.percentile(channel, 10)) / 255, float(np.percentile(channel, 90)) / 255])
    ela_features = _block_stats(ela_gray, blocks=4)
    # ELA global quantiles complement regional ELA summaries.
    ela_features.extend([float(np.mean(ela_gray))/255, float(np.std(ela_gray))/255,
                         float(np.percentile(ela_gray, 99))/255])
    return np.concatenate([hist, np.asarray(dct_features + ela_features + color_features, dtype=np.float32)])


def suspicious_heatmap(rgb: np.ndarray) -> np.ndarray:
    """Create a heuristic region map from locally elevated ELA; not a pixel mask."""
    ela = ela_map(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)))
    gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), 7)
    baseline = cv2.GaussianBlur(smooth, (0, 0), 21)
    local_excess = np.maximum(smooth.astype(np.float32) - baseline.astype(np.float32), 0)
    score = cv2.normalize(local_excess, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(score, cv2.COLORMAP_JET)
