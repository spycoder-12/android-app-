"""Evaluate on the held-out TEST split (never used for training or model selection).
Also picks confidence thresholds on the VALIDATION split and runs a simulated-degradation stress test.
python training/evaluate.py [--backbone small_cnn]"""
import argparse
import json

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix

from common import CACHE, CLASS_KEYS, REPORTS, load_cache


def predict(core, X, bs=128):
    out = []
    for i in range(0, len(X), bs):
        out.append(core.predict(X[i:i + bs].astype("float32") / 255.0, verbose=0))
    return np.concatenate(out)


def perturb(X, kind, rng):
    out = np.empty_like(X)
    for i, a in enumerate(X):
        im = Image.fromarray(a)
        if kind == "darker":   im = Image.eval(im, lambda v: int(v * 0.55))
        elif kind == "brighter": im = Image.eval(im, lambda v: min(255, int(v * 1.45)))
        elif kind == "blur":   im = im.filter(ImageFilter.GaussianBlur(1.6))
        elif kind == "crop":
            s = X.shape[1]; c = int(s * 0.7); x0, y0 = rng.integers(0, s - c, 2)
            im = im.crop((x0, y0, x0 + c, y0 + c)).resize((s, s), Image.BILINEAR)
        elif kind == "noise":
            n = rng.normal(0, 18, a.shape); im = Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))
        out[i] = np.asarray(im)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="small_cnn")
    a = ap.parse_args()
    from tensorflow import keras
    core = keras.models.load_model(CACHE / f"{a.backbone}.keras")
    X, y, split, _ = load_cache()
    va, te = split == "val", split == "test"
    pv, pt = predict(core, X[va]), predict(core, X[te])
    yv, yt = y[va], y[te]

    # thresholds chosen on VALIDATION only: smallest confidence whose accepted predictions are >=98% accurate
    # while still accepting at least 80% of validation images
    conf_v, ok_v = pv.max(1), pv.argmax(1) == yv
    chosen = {"min_confidence": 0.6, "note": "98%/80% target not reachable; default used"}
    for t in np.arange(0.30, 0.99, 0.01):
        m = conf_v >= t
        if m.sum() >= 0.8 * len(m) and ok_v[m].mean() >= 0.98:
            chosen = {"min_confidence": round(float(t), 2),
                      "val_accuracy_when_accepted": round(float(ok_v[m].mean()), 4),
                      "val_coverage": round(float(m.mean()), 4)}
            break
    chosen["min_margin"] = 0.15

    pred = pt.argmax(1)
    rep = classification_report(yt, pred, labels=list(range(len(CLASS_KEYS))),
                                target_names=CLASS_KEYS, output_dict=True, zero_division=0)
    cm = confusion_matrix(yt, pred, labels=list(range(len(CLASS_KEYS))))
    acc = float((pred == yt).mean())
    conf_t = pt.max(1); acc_t = pred == yt
    m = conf_t >= chosen["min_confidence"]
    stress = {}
    rng = np.random.default_rng(0)
    for kind in ["darker", "brighter", "blur", "crop", "noise"]:
        Xp = perturb(X[te], kind, rng)
        stress[kind] = round(float((predict(core, Xp).argmax(1) == yt).mean()), 4)

    metrics = {"split": "test", "n_test": int(te.sum()), "accuracy": round(acc, 4),
               "macro_f1": round(rep["macro avg"]["f1-score"], 4),
               "weighted_f1": round(rep["weighted avg"]["f1-score"], 4),
               "val_accuracy": round(float(ok_v.mean()), 4),
               "test_accuracy_when_confident": round(float(acc_t[m].mean()), 4) if m.any() else None,
               "test_coverage_at_threshold": round(float(m.mean()), 4),
               "per_class": {k: {kk: round(v, 4) for kk, v in rep[k].items() if kk != "support"} |
                             {"support": int(rep[k]["support"])} for k in CLASS_KEYS},
               "stress_test_accuracy_simulated": stress,
               "caveats": ["Lab-condition PlantVillage images only; no real-world field photos were evaluated.",
                           "Leaf grouping is complete only for images present in the dataset's leaf-map; the rest use an approximate heuristic."]}
    json.dump(metrics, open(REPORTS / "metrics.json", "w"), indent=1)
    json.dump(chosen, open(REPORTS / "thresholds.json", "w"), indent=1)
    np.savetxt(REPORTS / "confusion_matrix.csv", cm, fmt="%d", delimiter=",", header=",".join(CLASS_KEYS))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(CLASS_KEYS)), CLASS_KEYS, rotation=60, ha="right"); ax.set_yticks(range(len(CLASS_KEYS)), CLASS_KEYS)
        for i in range(len(cm)):
            for j in range(len(cm)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=110)
    except Exception as e:
        print("plot skipped:", e)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=1))
    print(classification_report(yt, pred, target_names=CLASS_KEYS, digits=3, zero_division=0))
    print("thresholds", chosen)


if __name__ == "__main__":
    main()
