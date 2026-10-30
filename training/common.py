"""Shared training helpers: class mapping, leaf-group splitting, model architecture."""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PLANT = "tomato"
CLASS_PREFIX = "Tomato___"
IMG_SIZE = 112
NORMALIZATION = "zero_1"   # x/255 -> [0,1]; MUST equal model/model_config.json
RESIZE_MODE = "squash"
SEED = 42
CACHE = Path(__file__).resolve().parent / "cache"
REPORTS = Path(__file__).resolve().parent / "reports"

# Folder name in the dataset -> stable label key used by the app (data/diseases.json)
FOLDER_TO_KEY = {
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
    "Tomato___Tomato_mosaic_virus": "mosaic_virus",
    "Tomato___healthy": "healthy",
}
CLASS_KEYS = sorted(FOLDER_TO_KEY.values())   # index order == model output order


def group_key(folder, fname, leaf_map):
    """Identify which physical leaf an image shows, so one leaf never straddles splits.
    1) the dataset's own leaf-map; 2) fallback: blocks of 8 consecutive photo numbers
    (approximate, may leave small residual leakage); 3) the file itself."""
    stem = fname.split("___", 1)[-1].rsplit(".", 1)[0]
    hit = leaf_map.get(stem.lower())
    if hit:
        return hit[0]
    m = re.match(r"^(.*?)\s*(\d+)$", stem)
    if m:
        return f"{folder}:::heur:{m.group(1).lower()}:{int(m.group(2)) // 8}"
    return f"{folder}:::file:{fname}"


def assign_splits(groups_by_class, seed=SEED, fractions=(0.70, 0.15, 0.15)):
    """groups_by_class: {class: {group: [files]}} -> {file_id: 'train'|'val'|'test'}."""
    import random
    rng = random.Random(seed)
    out = {}
    for cls, groups in groups_by_class.items():
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        seen = 0
        for k in keys:
            frac = seen / total
            split = "train" if frac < fractions[0] else ("val" if frac < fractions[0] + fractions[1] else "test")
            for f in groups[k]:
                out[f] = split
            seen += len(groups[k])
    return out


BLOCKS = [(16, True), (32, True), (64, True), (96, True), (128, False)]  # (filters, pool_after)


def build_core(num_classes, size=IMG_SIZE):
    """Compact CNN (about 50M MACs). Input is already scaled to [0,1]. No augmentation layers
    here, so the exported graph is exactly what runs on device."""
    from tensorflow import keras
    L = keras.layers
    inp = L.Input((size, size, 3))
    x = inp
    for i, (f, pool) in enumerate(BLOCKS):
        x = L.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = L.BatchNormalization(name=f"bn{i}")(x)
        x = L.ReLU()(x)
        if pool:
            x = L.MaxPooling2D(2)(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(num_classes, activation="softmax", name="logits")(x)
    return keras.Model(inp, out, name="plantcare_small_cnn")


def build_mobilenet(num_classes, size=IMG_SIZE):
    """Transfer-learning option (needs ImageNet weights download; use on PC/Colab).
    Expects inputs in [0,1] and rescales to [-1,1] internally so the app pipeline is unchanged."""
    from tensorflow import keras
    L = keras.layers
    base = keras.applications.MobileNetV2((size, size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inp = L.Input((size, size, 3))
    x = L.Rescaling(2.0, offset=-1.0)(inp)
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(num_classes, activation="softmax", name="logits")(x)
    return keras.Model(inp, out, name="plantcare_mobilenetv2")


def load_cache():
    import numpy as np
    d = np.load(CACHE / "tomato_cache.npz", allow_pickle=True)
    return d["X"], d["y"], d["split"], list(d["files"])
