"""Build a leaf-grouped train/val/test split and cache preprocessed images.

Dataset: PlantVillage (Hughes & Salathe; Mohanty et al. 2016), tomato classes.
Get it (no need to download the whole repo):
  git clone --filter=blob:none --no-checkout --depth 1 https://github.com/spMohanty/PlantVillage-Dataset pv
  cd pv && git sparse-checkout init --no-cone
  git sparse-checkout set README.md CITATION.cff leaf-map.json 'raw/color/Tomato___*' && git checkout
LICENSE: no license file is in that repo. Verify terms on the dataset's Hugging Face card
(huggingface.co/datasets/mohanty/PlantVillage) before redistributing data or models.

Usage: python training/prepare_data.py --data pv/raw/color --leaf-map pv/leaf-map.json
"""
import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from PIL import Image

from common import (CACHE, CLASS_KEYS, FOLDER_TO_KEY, IMG_SIZE, NORMALIZATION, REPORTS,
                    RESIZE_MODE, assign_splits, group_key)
from services.image_service import load_image, preprocess_for_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--leaf-map", required=True)
    a = ap.parse_args()
    leaf_map = json.load(open(a.leaf_map))
    CACHE.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

    records, groups_by_class = [], defaultdict(lambda: defaultdict(list))
    for folder, key in FOLDER_TO_KEY.items():
        d = f"{a.data}/{folder}"
        for f in sorted(__import__("os").listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                fid = f"{folder}/{f}"
                records.append((fid, key, f"{d}/{f}"))
                groups_by_class[key][group_key(folder, f, leaf_map)].append(fid)
    splits = assign_splits(groups_by_class)

    X = np.zeros((len(records), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    y = np.zeros(len(records), np.int16); sp = []; files = []; bad = 0
    for i, (fid, key, path) in enumerate(records):
        try:
            img = load_image(path)   # same loader as the app
            arr = preprocess_for_model(img, (IMG_SIZE, IMG_SIZE), "none", RESIZE_MODE)[0]
            X[i] = arr.astype(np.uint8)
        except Exception as e:
            bad += 1; print("skip", path, e)
        y[i] = CLASS_KEYS.index(key); sp.append(splits[fid]); files.append(fid)
        if i % 3000 == 0: print(i, "/", len(records), flush=True)
    sp = np.array(sp)
    np.savez(CACHE / "tomato_cache.npz", X=X, y=y, split=sp, files=np.array(files))

    # split manifest + leakage check
    with open(REPORTS / "split_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "class", "split"])
        for fid, key, _ in records: w.writerow([fid, key, splits[fid]])
    leak = 0
    for key, groups in groups_by_class.items():
        for g, fs in groups.items():
            if len({splits[f] for f in fs}) > 1: leak += 1
    print("images", len(records), "skipped", bad, "group-split violations", leak)
    for s in ("train", "val", "test"):
        print(s, int((sp == s).sum()), {CLASS_KEYS[c]: int(((y == c) & (sp == s)).sum()) for c in range(len(CLASS_KEYS))})


if __name__ == "__main__":
    main()
