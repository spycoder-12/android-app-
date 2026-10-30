"""Image validation, loading, quality hints, managed storage and model preprocessing.

Pure Python (Pillow + NumPy) - no Kivy imports, so it is fully unit-testable.
"""
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MAX_BYTES = 20 * 1024 * 1024      # 20 MB
MIN_SIDE = 64                     # px; smaller images carry too little detail
STORED_MAX_SIDE = 1600            # px; stored/preview copies are downscaled to this
Image.MAX_IMAGE_PIXELS = 60_000_000  # guard against decompression bombs


class ImageError(Exception):
    """Raised with a message that is safe to show directly to the user."""


def validate_image(path):
    """Check that `path` is a readable, supported, non-corrupted image.
    Returns {'width','height','format','size_bytes'}; raises ImageError otherwise."""
    if not path:
        raise ImageError("No image selected. Please choose or capture a leaf photo first.")
    p = Path(path)
    if not p.is_file():
        raise ImageError("The selected file could not be found. It may have been moved or deleted.")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ImageError("Unsupported file type. Please use a JPG, PNG, WEBP or BMP image.")
    size = p.stat().st_size
    if size == 0:
        raise ImageError("This file is empty. Please pick a different image.")
    if size > MAX_BYTES:
        raise ImageError(f"This image is too large ({size / 1048576:.0f} MB). The limit is "
                         f"{MAX_BYTES // 1048576} MB.")
    try:
        with Image.open(p) as im:
            im.verify()                      # cheap structural check
        with Image.open(p) as im:            # verify() invalidates the object, so reopen
            im.load()                        # forces full decode; catches truncation
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ImageError("This file is not a valid image or appears to be corrupted.") from None
    if fmt not in ALLOWED_FORMATS:
        raise ImageError("Unsupported image format. Please use a JPG, PNG, WEBP or BMP image.")
    if min(w, h) < MIN_SIDE:
        raise ImageError(f"This image is too small ({w}x{h}). Use a photo at least "
                         f"{MIN_SIDE}x{MIN_SIDE} pixels.")
    return {"width": w, "height": h, "format": fmt, "size_bytes": size}


def load_image(path):
    """Validate then return an upright RGB PIL image (EXIF rotation applied,
    transparency flattened onto white)."""
    validate_image(path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            return bg
        return im.convert("RGB")


def assess_quality(img):
    """Cheap heuristic hints (NOT calibrated, NOT a guarantee). Returns a list of warnings."""
    warnings = []
    w, h = img.size
    if min(w, h) < 224:
        warnings.append("Image resolution is low; move closer to the leaf.")
    g = np.asarray(ImageOps.grayscale(img.resize((256, 256))), dtype=np.float32)
    mean = float(g.mean())
    if mean < 40:
        warnings.append("Image looks very dark. Try better lighting.")
    elif mean > 225:
        warnings.append("Image looks overexposed. Avoid direct harsh light.")
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    if float(lap.var()) < 20:
        warnings.append("Image may be blurry or lacks detail. Hold the phone steady and focus on the leaf.")
    return warnings


def save_managed_copy(src_path, dest_dir):
    """Store an upright, downscaled JPEG copy in app-managed storage; return its path.
    Used for preview (Kivy ignores EXIF rotation) and later for history."""
    img = load_image(src_path)
    img.thumbnail((STORED_MAX_SIDE, STORED_MAX_SIDE), Image.LANCZOS)
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, f"{uuid.uuid4().hex}.jpg")
    img.save(out, "JPEG", quality=90)
    return out


def preprocess_for_model(img, size=(224, 224), normalization="minus1_1", resize_mode="squash"):
    """Convert a PIL RGB image to a float32 batch of shape (1, H, W, 3).

    !! These settings MUST match the training pipeline (decided in Phase 4 and recorded in
    model/model_config.json). Defaults assume MobileNetV2-style preprocessing and the
    squash-resize used by keras image_dataset_from_directory.
    normalization: 'minus1_1' -> x/127.5-1 | 'zero_1' -> x/255 | 'none' -> raw 0..255
    resize_mode:   'squash' (ignore aspect ratio) | 'center_crop'
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if resize_mode == "center_crop":
        img = ImageOps.fit(img, size, Image.BILINEAR)
    elif resize_mode == "squash":
        img = img.resize(size, Image.BILINEAR)
    else:
        raise ValueError(f"unknown resize_mode: {resize_mode}")
    arr = np.asarray(img, dtype=np.float32)          # RGB order, (H, W, 3)
    if normalization == "minus1_1":
        arr = arr / 127.5 - 1.0
    elif normalization == "zero_1":
        arr = arr / 255.0
    elif normalization != "none":
        raise ValueError(f"unknown normalization: {normalization}")
    return arr[np.newaxis, ...]
