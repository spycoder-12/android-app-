import numpy as np
import pytest
from PIL import Image

from services import image_service as svc


def make(tmp_path, name="leaf.jpg", size=(300, 200), mode="RGB", color=(40, 160, 60), fmt=None):
    p = tmp_path / name
    Image.new(mode, size, color).save(p, fmt)
    return str(p)


def test_valid_jpeg_and_png(tmp_path):
    assert svc.validate_image(make(tmp_path))["format"] == "JPEG"
    assert svc.validate_image(make(tmp_path, "a.png"))["format"] == "PNG"


def test_empty_input():
    for bad in (None, ""):
        with pytest.raises(svc.ImageError, match="No image selected"):
            svc.validate_image(bad)


def test_missing_file(tmp_path):
    with pytest.raises(svc.ImageError, match="could not be found"):
        svc.validate_image(str(tmp_path / "nope.jpg"))


def test_unsupported_extension(tmp_path):
    p = tmp_path / "notes.txt"; p.write_text("hi")
    with pytest.raises(svc.ImageError, match="Unsupported file type"):
        svc.validate_image(str(p))


def test_empty_file(tmp_path):
    p = tmp_path / "e.jpg"; p.write_bytes(b"")
    with pytest.raises(svc.ImageError, match="empty"):
        svc.validate_image(str(p))


def test_corrupted_file(tmp_path):
    p = tmp_path / "bad.jpg"; p.write_bytes(b"this is not an image" * 50)
    with pytest.raises(svc.ImageError, match="corrupted"):
        svc.validate_image(str(p))


def test_truncated_file(tmp_path):
    good = make(tmp_path, size=(400, 400))
    data = open(good, "rb").read()
    t = tmp_path / "trunc.jpg"; t.write_bytes(data[: len(data) // 2])
    with pytest.raises(svc.ImageError, match="corrupted"):
        svc.validate_image(str(t))


def test_too_small(tmp_path):
    with pytest.raises(svc.ImageError, match="too small"):
        svc.validate_image(make(tmp_path, size=(20, 20)))


def test_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "MAX_BYTES", 100)
    with pytest.raises(svc.ImageError, match="too large"):
        svc.validate_image(make(tmp_path, size=(300, 300)))


def test_wrong_extension_real_format_is_still_checked(tmp_path):
    p = tmp_path / "x.png"; p.write_bytes(b"\x00" * 100)
    with pytest.raises(svc.ImageError):
        svc.validate_image(str(p))


def test_load_converts_to_rgb(tmp_path):
    assert svc.load_image(make(tmp_path, "g.png", mode="L", color=128)).mode == "RGB"
    rgba = svc.load_image(make(tmp_path, "t.png", mode="RGBA", color=(0, 0, 0, 0)))
    assert rgba.mode == "RGB" and rgba.getpixel((5, 5)) == (255, 255, 255)  # transparent -> white


def test_exif_rotation_applied(tmp_path):
    img = Image.new("RGB", (300, 200), (10, 200, 10))
    exif = img.getexif(); exif[0x0112] = 6  # rotate 90 CW when displayed
    p = tmp_path / "rot.jpg"; img.save(p, exif=exif)
    assert svc.load_image(str(p)).size == (200, 300)


def test_preprocess_shape_dtype_range(tmp_path):
    img = svc.load_image(make(tmp_path))
    x = svc.preprocess_for_model(img)
    assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
    assert -1.0 <= x.min() and x.max() <= 1.0


def test_preprocess_custom_size_and_modes(tmp_path):
    img = svc.load_image(make(tmp_path))
    assert svc.preprocess_for_model(img, size=(128, 128)).shape == (1, 128, 128, 3)
    z = svc.preprocess_for_model(img, normalization="zero_1")
    assert 0.0 <= z.min() and z.max() <= 1.0
    r = svc.preprocess_for_model(img, normalization="none")
    assert r.max() > 1.0
    assert svc.preprocess_for_model(img, resize_mode="center_crop").shape == (1, 224, 224, 3)


def test_preprocess_channel_order_is_rgb(tmp_path):
    img = Image.new("RGB", (100, 100), (255, 0, 0))  # pure red
    x = svc.preprocess_for_model(img, normalization="none")
    assert x[0, 0, 0].tolist() == [255.0, 0.0, 0.0]


def test_preprocess_bad_options(tmp_path):
    img = svc.load_image(make(tmp_path))
    with pytest.raises(ValueError):
        svc.preprocess_for_model(img, normalization="bogus")
    with pytest.raises(ValueError):
        svc.preprocess_for_model(img, resize_mode="bogus")


def test_managed_copy(tmp_path):
    out = svc.save_managed_copy(make(tmp_path, size=(3000, 2000)), str(tmp_path / "store"))
    w, h = Image.open(out).size
    assert max(w, h) == svc.STORED_MAX_SIDE and out.endswith(".jpg")


def test_quality_warnings(tmp_path):
    dark = Image.new("RGB", (400, 400), (5, 5, 5))
    assert any("dark" in w for w in svc.assess_quality(dark))
    flat = Image.new("RGB", (400, 400), (120, 120, 120))
    assert any("blurry" in w for w in svc.assess_quality(flat))
    small = Image.new("RGB", (100, 100), (120, 120, 120))
    assert any("resolution" in w for w in svc.assess_quality(small))
