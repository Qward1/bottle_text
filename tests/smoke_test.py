from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from app.processor import encode_image, process_image, rotate_bound


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "_out"
OUT_DIR.mkdir(exist_ok=True)


def build_synthetic_bottle(rotation_degrees: float) -> tuple[bytes, dict[str, float]]:
    h, w = 1100, 820
    canvas = np.full((h, w, 3), 232, dtype=np.uint8)

    cv2.rectangle(canvas, (240, 120), (600, 980), (210, 214, 219), thickness=-1)
    cv2.rectangle(canvas, (315, 20), (525, 180), (206, 210, 214), thickness=-1)
    cv2.rectangle(canvas, (265, 360), (575, 730), (242, 242, 242), thickness=-1)

    weak_color = (150, 150, 150)
    cv2.putText(canvas, "LOT 7241", (300, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.0, weak_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "EXP 12.11.2027", (285, 565), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (160, 160, 160), 2, cv2.LINE_AA)
    cv2.putText(canvas, "BATCH 096", (310, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (155, 155, 155), 2, cv2.LINE_AA)

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    overlay = canvas.copy()
    cv2.ellipse(overlay, (430, 555), (145, 34), -18, 0, 360, (255, 255, 255), thickness=-1)
    cv2.ellipse(overlay, (455, 535), (120, 20), -18, 0, 360, (250, 250, 250), thickness=-1)
    canvas = cv2.addWeighted(overlay, 0.23, canvas, 0.77, 0)

    shadow = np.zeros_like(canvas)
    cv2.rectangle(shadow, (280, 520), (560, 600), (18, 18, 18), thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (41, 41), 0)
    canvas = cv2.addWeighted(canvas, 1.0, shadow, 0.08, 0)

    rng = np.random.default_rng(11)
    noise = rng.normal(0, 9, size=canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    rotated = rotate_bound(noisy, rotation_degrees)
    return encode_image(rotated, ext=".jpg", quality=92), {"rotation_degrees": float(rotation_degrees)}


def build_challenging_bottle() -> tuple[bytes, dict[str, float]]:
    return build_synthetic_bottle(rotation_degrees=1.7)


def build_sideways_bottle() -> tuple[bytes, dict[str, float]]:
    return build_synthetic_bottle(rotation_degrees=88.0)


def build_distractor_bottle() -> tuple[bytes, dict[str, float]]:
    h, w = 1100, 820
    canvas = np.full((h, w, 3), 230, dtype=np.uint8)

    cv2.rectangle(canvas, (245, 120), (605, 990), (208, 212, 218), thickness=-1)
    cv2.rectangle(canvas, (320, 20), (530, 185), (205, 209, 214), thickness=-1)
    cv2.rectangle(canvas, (275, 345), (585, 760), (243, 243, 243), thickness=-1)

    weak_color = (148, 148, 148)
    cv2.putText(canvas, "30.03.26", (308, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.0, weak_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "31.10.25S", (296, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (154, 154, 154), 2, cv2.LINE_AA)

    # Distractor on the right side that looks like a sharp label fragment.
    cv2.rectangle(canvas, (530, 700), (605, 935), (238, 238, 238), thickness=-1)
    for offset in range(0, 68, 8):
        cv2.line(canvas, (542 + offset, 712), (542 + offset, 925), (120, 120, 120), 2, cv2.LINE_AA)
    cv2.line(canvas, (515, 744), (596, 744), (130, 130, 130), 3, cv2.LINE_AA)
    cv2.line(canvas, (520, 792), (600, 792), (126, 126, 126), 3, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.ellipse(overlay, (438, 560), (148, 34), -16, 0, 360, (255, 255, 255), thickness=-1)
    cv2.ellipse(overlay, (465, 540), (125, 18), -16, 0, 360, (250, 250, 250), thickness=-1)
    canvas = cv2.addWeighted(overlay, 0.21, canvas, 0.79, 0)

    shadow = np.zeros_like(canvas)
    cv2.rectangle(shadow, (285, 520), (563, 605), (20, 20, 20), thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (39, 39), 0)
    canvas = cv2.addWeighted(canvas, 1.0, shadow, 0.08, 0)

    rng = np.random.default_rng(17)
    noise = rng.normal(0, 9, size=canvas.shape).astype(np.int16)
    noisy = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rotated = rotate_bound(noisy, 6.5)

    return encode_image(rotated, ext=".jpg", quality=92), {"rotation_degrees": 6.5}


def save_debug_outputs(oriented_bgr: np.ndarray, metadata: dict) -> None:
    overlay = oriented_bgr.copy()

    bx1 = metadata["bottle_box"]["x1"]
    by1 = metadata["bottle_box"]["y1"]
    bx2 = metadata["bottle_box"]["x2"]
    by2 = metadata["bottle_box"]["y2"]
    x1 = metadata["crop_box"]["x1"]
    y1 = metadata["crop_box"]["y1"]
    x2 = metadata["crop_box"]["x2"]
    y2 = metadata["crop_box"]["y2"]

    if metadata.get("bottle_found"):
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 160, 0), 3)
    if metadata.get("crop_found"):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(OUT_DIR / "debug_roi.jpg"), overlay)

    crop = oriented_bgr[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(str(OUT_DIR / "crop_preview.jpg"), crop)


if __name__ == "__main__":
    real_path = os.getenv("BOTTLE_TEST_IMAGE")
    expected = None

    if real_path and Path(real_path).exists():
        content = Path(real_path).read_bytes()
    else:
        content, expected = build_challenging_bottle()

    result = process_image(content, detector_backend="craft")

    cv2.imwrite(str(OUT_DIR / "improved.jpg"), result.improved_bgr)
    cv2.imwrite(str(OUT_DIR / "bw.png"), result.bw)
    cv2.imwrite(str(OUT_DIR / "high_contrast.jpg"), result.high_contrast)

    meta = result.metadata.__dict__
    save_debug_outputs(result.oriented_bgr, meta)

    print(meta)
    print("Saved outputs to:", OUT_DIR)

    assert result.improved_bgr.size > 0
    assert result.bw.size > 0
    assert result.high_contrast.size > 0

    if expected is not None:
        assert meta["bottle_found"] is True
        assert meta["crop_found"] is True
        assert meta["bottle_confidence"] > 1.4
        assert meta["detection_confidence"] > 2.0
        assert result.crop_bgr.shape[1] >= result.crop_bgr.shape[0], result.crop_bgr.shape
        assert abs(meta["bottle_rotation_degrees"]) >= max(0.8, expected["rotation_degrees"] * 0.4)
