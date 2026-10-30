import json
import os

import numpy as np
from PIL import Image

from services.app_context import AppContext
from services.prediction_service import ModelError
import pytest


class Fixed:
    num_classes = 3
    def predict_proba(self, b): return np.tile(np.array([0.9, 0.05, 0.05], np.float32), (len(b), 1))


def make_ctx(tmp_path):
    md = tmp_path / "model"; md.mkdir()
    (md / "model_config.json").write_text(json.dumps({
        "plant": "tomato", "version": "t", "engine": "numpy", "model_file": "m.npz", "input_size": 16,
        "normalization": "zero_1", "resize_mode": "squash",
        "class_keys": ["early_blight", "late_blight", "healthy"],
        "class_labels": {"early_blight": "Early blight", "late_blight": "Late blight", "healthy": "Healthy"},
        "thresholds": {"min_confidence": 0.6, "min_margin": 0.15}}))
    ctx = AppContext(str(tmp_path / "data"), model_dir=str(md))   # no weights file -> predictor None
    assert ctx.predictor is None and "missing" in ctx.model_error
    from services.prediction_service import PredictionService
    ctx.predictor = PredictionService(str(md), engine_override=Fixed()); ctx.model_error = None
    return ctx


def img(tmp_path):
    p = tmp_path / "l.jpg"; Image.new("RGB", (200, 200), (40, 150, 50)).save(p); return str(p)


def test_analyze_saves_history_when_enabled(tmp_path):
    ctx = make_ctx(tmp_path)
    rec = ctx.analyze(img(tmp_path))
    assert rec["saved"] and rec["predicted_key"] == "early_blight"
    assert ctx.db.count() == 1 and ctx.db.get(rec["id"])["predicted_label"] == "Early blight"


def test_analyze_respects_privacy_setting(tmp_path):
    ctx = make_ctx(tmp_path); ctx.settings.set("save_history", False)
    rec = ctx.analyze(img(tmp_path))
    assert not rec["saved"] and rec["id"] is None and ctx.db.count() == 0


def test_analyze_without_model_raises(tmp_path):
    ctx = make_ctx(tmp_path); ctx.predictor = None; ctx.model_error = "Model not installed."
    with pytest.raises(ModelError, match="not installed"):
        ctx.analyze(img(tmp_path))
