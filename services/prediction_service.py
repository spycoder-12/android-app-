"""Loads the trained model + config and turns an image into a labelled, honestly-qualified result.

Nothing here is hardcoded: labels, input size, normalization and thresholds come from
model/model_config.json, which training/convert_model.py writes next to the model weights."""
import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from services import image_service

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")


class ModelError(Exception):
    """Model missing, incompatible or failing; message is safe to show."""


@dataclass
class Prediction:
    plant: str
    top_key: str
    top_label: str
    confidence: float
    status: str                      # "ok" | "inconclusive"
    top_k: list = field(default_factory=list)   # [{"key","label","prob"}]
    warnings: list = field(default_factory=list)
    model_version: str = ""


class TFLiteEngine:
    name = "tflite"

    def __init__(self, path):
        Interpreter = None
        for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
            try:
                Interpreter = __import__(mod, fromlist=["Interpreter"]).Interpreter
                break
            except Exception:
                continue
        if Interpreter is None:
            raise ModelError("No TFLite runtime is installed.")
        self.it = Interpreter(model_path=path)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def predict_proba(self, batch):
        outs = []
        for x in batch:
            self.it.set_tensor(self.i["index"], x[np.newaxis].astype(self.i["dtype"]))
            self.it.invoke()
            outs.append(self.it.get_tensor(self.o["index"])[0])
        return np.stack(outs)


class PredictionService:
    def __init__(self, model_dir=MODEL_DIR, engine_override=None):
        cfg_path = os.path.join(model_dir, "model_config.json")
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                self.cfg = json.load(fh)
            self.keys = list(self.cfg["class_keys"])
            self.labels = dict(self.cfg["class_labels"])
            self.size = int(self.cfg["input_size"])
            self.normalization = self.cfg["normalization"]
            self.resize_mode = self.cfg["resize_mode"]
            self.plant = self.cfg["plant"]
            self.version = str(self.cfg["version"])
            self.min_conf = float(self.cfg["thresholds"]["min_confidence"])
            self.min_margin = float(self.cfg["thresholds"]["min_margin"])
        except FileNotFoundError:
            raise ModelError("The disease model is not installed (model/model_config.json is missing). "
                             "Train or copy the model first; see the README.") from None
        except (ValueError, KeyError) as exc:
            raise ModelError(f"The model configuration is invalid or incompatible: {exc}") from exc

        self.engine = engine_override or self._load_engine(model_dir)
        n = getattr(self.engine, "num_classes", None)
        if n is not None and n != len(self.keys):
            raise ModelError(f"Model has {n} outputs but the config lists {len(self.keys)} classes.")

    def _load_engine(self, model_dir):
        kind = self.cfg.get("engine", "numpy")
        f = os.path.join(model_dir, self.cfg["model_file"])
        if not os.path.isfile(f):
            raise ModelError(f"Model file is missing: {self.cfg['model_file']}")
        try:
            if kind == "numpy":
                from services.numpy_engine import NumpyEngine
                return NumpyEngine(f)
            if kind == "tflite":
                return TFLiteEngine(f)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"The model file could not be loaded: {exc}") from exc
        raise ModelError(f"Unknown model engine: {kind}")

    @staticmethod
    def _leaf_color_fraction(img):
        """Heuristic: share of pixels with leaf-like hues (yellow/green/brown, not too dark/grey)."""
        hsv = np.asarray(img.resize((96, 96)).convert("HSV"), dtype=np.float32)
        h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
        return float(((h >= 20) & (h <= 170) & (s > 0.15) & (v > 0.15)).mean())

    def predict(self, image_or_path, top_k=3):
        img = (image_or_path if isinstance(image_or_path, Image.Image)
               else image_service.load_image(image_or_path))   # raises ImageError for bad files
        x = image_service.preprocess_for_model(img, (self.size, self.size),
                                               self.normalization, self.resize_mode)
        try:
            probs = self.engine.predict_proba(x)[0]
        except Exception as exc:
            raise ModelError(f"Analysis failed: {exc}") from exc
        if probs.shape[0] != len(self.keys) or not np.all(np.isfinite(probs)):
            raise ModelError("The model returned an invalid result.")
        order = np.argsort(probs)[::-1]
        tk = [{"key": self.keys[i], "label": self.labels[self.keys[i]], "prob": float(probs[i])}
              for i in order[:top_k]]
        conf, margin = tk[0]["prob"], tk[0]["prob"] - tk[1]["prob"]
        warnings, status = [], "ok"
        if conf < self.min_conf or margin < self.min_margin:
            status = "inconclusive"
            warnings.append("The model is not confident about this image. Retake a clearer, closer "
                            "photo of a single leaf, or ask an expert.")
        if self._leaf_color_fraction(img) < 0.10:
            status = "inconclusive"
            warnings.append("This image does not look like a leaf (few leaf-like colors found). "
                            "The result may be meaningless.")
        warnings.append(f"The model only knows {len(self.keys)} {self.plant} leaf conditions. A photo of "
                        "another plant or a different problem can still get a wrong, confident-looking answer.")
        return Prediction(self.plant, tk[0]["key"], tk[0]["label"], conf, status, tk, warnings, self.version)
