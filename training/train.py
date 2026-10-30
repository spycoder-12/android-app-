"""Train the tomato classifier.  python training/train.py [--backbone small_cnn|mobilenetv2] [--epochs N]
small_cnn trains from scratch (works offline). mobilenetv2 = transfer learning (needs ImageNet weights;
run on a PC/Colab). Writes training/cache/<name>.keras and training/reports/history.json."""
import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import (CACHE, CLASS_KEYS, IMG_SIZE, REPORTS, SEED, build_core, build_mobilenet,
                    load_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="small_cnn", choices=["small_cnn", "mobilenetv2"])
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    X, y, split, _ = load_cache()
    tr, va = split == "train", split == "val"       # test split is NOT touched here
    n = len(CLASS_KEYS)
    counts = np.bincount(y[tr], minlength=n)
    cw = {i: float((counts.sum() / (n * counts[i])) ** 0.5) for i in range(n)}  # softened balancing

    core = (build_core if a.backbone == "small_cnn" else build_mobilenet)(n)
    aug = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.15, fill_mode="reflect"),
        keras.layers.RandomZoom((-0.15, 0.15), fill_mode="reflect"),
        keras.layers.RandomBrightness(0.2, value_range=(0, 1)),
        keras.layers.RandomContrast(0.2),
    ])
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    model = keras.Model(inp, core(aug(inp)))        # augmentation only exists in the training wrapper

    cw_arr = np.array([cw[i] for i in range(n)], np.float32)

    class Batches(keras.utils.PyDataset):
        """Reads straight from the cached uint8 array (no copies; keeps RAM low)."""
        def __init__(self, mask, shuffle, weighted):
            super().__init__(workers=1, use_multiprocessing=False)
            self.ids, self.shuffle, self.weighted = np.where(mask)[0], shuffle, weighted
            self.rng = np.random.default_rng(SEED); self.on_epoch_end()
        def __len__(self): return int(np.ceil(len(self.ids) / a.batch))
        def __getitem__(self, i):
            sel = np.sort(self.ids[i * a.batch:(i + 1) * a.batch])
            xb, yb = X[sel].astype(np.float32) / 255.0, y[sel].astype(np.int32)
            return (xb, yb, cw_arr[yb]) if self.weighted else (xb, yb)
        def on_epoch_end(self):
            if self.shuffle: self.rng.shuffle(self.ids)

    steps = (int(tr.sum()) // a.batch) * a.epochs
    lr = keras.optimizers.schedules.CosineDecay(2e-3, steps)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    best = CACHE / f"best_{a.backbone}.weights.h5"
    cbs = [keras.callbacks.ModelCheckpoint(best, monitor="val_accuracy", save_best_only=True,
                                           save_weights_only=True, verbose=1)]
    t0 = time.time()
    h = model.fit(Batches(tr, True, True), validation_data=Batches(va, False, False), epochs=a.epochs,
                  callbacks=cbs, verbose=2)
    model.load_weights(best)                          # restore best-by-validation
    core.save(CACHE / f"{a.backbone}.keras")
    json.dump({"backbone": a.backbone, "epochs": a.epochs, "batch": a.batch,
               "train_seconds": round(time.time() - t0), "history": h.history,
               "class_keys": CLASS_KEYS, "img_size": IMG_SIZE},
              open(REPORTS / "history.json", "w"), indent=1)
    print("done; best val_acc", max(h.history["val_accuracy"]))


if __name__ == "__main__":
    main()
