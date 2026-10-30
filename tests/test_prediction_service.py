"""Plumbing tests with FAKE engines/models. They check label mapping, preprocessing shape, thresholds and
error handling - NOT model accuracy (see training/reports/metrics.json for that)."""
import json

import numpy as np
import pytest
from PIL import Image

from services import image_service
from services.numpy_engine import NumpyEngine
from services.prediction_service import ModelError, PredictionService

KEYS = ["a_disease", "b_disease", "healthy"]
LABELS = {"a_disease": "A disease", "b_disease": "B disease", "healthy": "Healthy"}


def write_cfg(d, **over):
    cfg = {"plant": "tomato", "version": "t1", "engine": "numpy", "model_file": "m.npz", "input_size": 32,
           "normalization": "zero_1", "resize_mode": "squash", "class_keys": KEYS, "class_labels": LABELS,
           "thresholds": {"min_confidence": 0.6, "min_margin": 0.15}}
    cfg.update(over)
    (d / "model_config.json").write_text(json.dumps(cfg))


class Fixed:
    """Fake engine returning a preset probability vector and recording its input shape."""
    def __init__(self, probs): self.p, self.shape, self.num_classes = np.array(probs, np.float32), None, len(probs)
    def predict_proba(self, b): self.shape = b.shape; return np.tile(self.p, (len(b), 1))


def leaf(path, size=(200, 160)):
    Image.new("RGB", size, (50, 150, 60)).save(path); return str(path)


def svc(tmp_path, probs, **over):
    write_cfg(tmp_path, **over)
    return PredictionService(str(tmp_path), engine_override=Fixed(probs))


def test_missing_model(tmp_path):
    with pytest.raises(ModelError, match="not installed"):
        PredictionService(str(tmp_path))


def test_invalid_config(tmp_path):
    (tmp_path / "model_config.json").write_text("{not json")
    with pytest.raises(ModelError, match="invalid"):
        PredictionService(str(tmp_path))
    (tmp_path / "model_config.json").write_text(json.dumps({"plant": "tomato"}))
    with pytest.raises(ModelError, match="invalid"):
        PredictionService(str(tmp_path))


def test_missing_weights_file(tmp_path):
    write_cfg(tmp_path)
    with pytest.raises(ModelError, match="missing"):
        PredictionService(str(tmp_path))


def test_class_count_mismatch(tmp_path):
    write_cfg(tmp_path)
    with pytest.raises(ModelError, match="outputs"):
        PredictionService(str(tmp_path), engine_override=Fixed([0.5, 0.5]))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_label_mapping_follows_config_order(tmp_path, idx):
    probs = [0.02, 0.02, 0.02]; probs[idx] = 0.96
    s = svc(tmp_path, probs)
    p = s.predict(leaf(tmp_path / "l.png"))
    assert p.top_key == KEYS[idx] and p.top_label == LABELS[KEYS[idx]] and p.status == "ok"
    assert p.confidence == pytest.approx(0.96) and len(p.top_k) == 3


def test_engine_receives_correct_input_shape(tmp_path):
    s = svc(tmp_path, [0.9, 0.05, 0.05])
    s.predict(leaf(tmp_path / "l.png", (500, 300)))
    assert s.engine.shape == (1, 32, 32, 3)


def test_low_confidence_is_inconclusive(tmp_path):
    p = svc(tmp_path, [0.45, 0.35, 0.20]).predict(leaf(tmp_path / "l.png"))
    assert p.status == "inconclusive" and any("not confident" in w for w in p.warnings)


def test_small_margin_is_inconclusive(tmp_path):
    p = svc(tmp_path, [0.62, 0.55 - 0.17, 0.0 + 0.2]).predict(leaf(tmp_path / "l.png"))  # conf .62, margin .24 -> ok
    assert p.status == "ok"
    p = svc(tmp_path, [0.64, 0.60, 0.0]).predict(leaf(tmp_path / "l.png"))
    assert p.status == "inconclusive"


def test_non_leaf_image_warns(tmp_path):
    Image.new("RGB", (200, 200), (120, 120, 125)).save(tmp_path / "grey.png")
    p = svc(tmp_path, [0.97, 0.02, 0.01]).predict(str(tmp_path / "grey.png"))
    assert p.status == "inconclusive" and any("does not look like a leaf" in w for w in p.warnings)


def test_always_includes_scope_warning(tmp_path):
    p = svc(tmp_path, [0.97, 0.02, 0.01]).predict(leaf(tmp_path / "l.png"))
    assert any("only knows 3" in w for w in p.warnings)


def test_invalid_inputs_raise_user_errors(tmp_path):
    s = svc(tmp_path, [0.9, 0.05, 0.05])
    with pytest.raises(image_service.ImageError):
        s.predict(None)
    bad = tmp_path / "bad.jpg"; bad.write_bytes(b"junk" * 30)
    with pytest.raises(image_service.ImageError):
        s.predict(str(bad))


def test_nan_output_rejected(tmp_path):
    s = svc(tmp_path, [np.nan, 0.5, 0.5])
    with pytest.raises(ModelError, match="invalid"):
        s.predict(leaf(tmp_path / "l.png"))


def _naive_forward(z, x):
    """Independent loop-based reference for the NumPy engine."""
    i = 0
    while f"conv{i}_w" in z:
        w, b, pool = z[f"conv{i}_w"], z[f"conv{i}_b"], bool(z[f"conv{i}_pool"])
        H, W, C = x.shape; F = w.shape[3]
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0))); out = np.zeros((H, W, F), np.float32)
        for r in range(H):
            for c in range(W):
                patch = pad[r:r + 3, c:c + 3, :]
                out[r, c] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2])) + b
        x = np.maximum(out, 0)
        if pool:
            x = x[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2, F).max(axis=(1, 3))
        i += 1
    logits = x.mean(axis=(0, 1)) @ z["dense_w"] + z["dense_b"]
    e = np.exp(logits - logits.max()); return e / e.sum()


def test_numpy_engine_matches_naive_reference(tmp_path):
    rng = np.random.default_rng(0)
    z = {"conv0_w": rng.normal(size=(3, 3, 3, 4)).astype(np.float32), "conv0_b": rng.normal(size=4).astype(np.float32),
         "conv0_pool": np.array(True),
         "conv1_w": rng.normal(size=(3, 3, 4, 5)).astype(np.float32), "conv1_b": rng.normal(size=5).astype(np.float32),
         "conv1_pool": np.array(False),
         "dense_w": rng.normal(size=(5, 3)).astype(np.float32), "dense_b": rng.normal(size=3).astype(np.float32)}
    np.savez(tmp_path / "m.npz", **z)
    x = rng.random((2, 12, 12, 3)).astype(np.float32)
    got = NumpyEngine(str(tmp_path / "m.npz")).predict_proba(x)
    ref = np.stack([_naive_forward(z, xi) for xi in x])
    assert np.abs(got - ref).max() < 1e-4 and np.allclose(got.sum(1), 1.0, atol=1e-5)
