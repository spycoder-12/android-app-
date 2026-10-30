"""Export the trained Keras model for the app and VALIDATE the export against Keras.
 - model/tomato_cnn.npz  : BatchNorm-folded weights for the pure-NumPy engine (default on Android)
 - model/tomato.tflite   : optional TFLite export (validated too)
 - model/model_config.json : input size, normalization, class order, thresholds, evaluation summary
python training/convert_model.py [--backbone small_cnn]"""
import argparse
import json
import time

import numpy as np

from common import (BLOCKS, CACHE, CLASS_KEYS, IMG_SIZE, NORMALIZATION, PLANT, REPORTS, RESIZE_MODE, ROOT,
                    load_cache)

MODEL_DIR = ROOT / "model"


def fold(core):
    z = {}
    for i, (_, pool) in enumerate(BLOCKS):
        w = core.get_layer(f"conv{i}").get_weights()[0]
        bn = core.get_layer(f"bn{i}")
        g, b, mu, var = bn.get_weights()
        s = g / np.sqrt(var + bn.epsilon)
        z[f"conv{i}_w"] = (w * s).astype(np.float32)
        z[f"conv{i}_b"] = (b - mu * s).astype(np.float32)
        z[f"conv{i}_pool"] = np.array(pool)
    dw, db = core.get_layer("logits").get_weights()
    z["dense_w"], z["dense_b"] = dw.astype(np.float32), db.astype(np.float32)
    return z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="small_cnn")
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    from tensorflow import keras
    import tensorflow as tf
    from services.numpy_engine import NumpyEngine
    core = keras.models.load_model(CACHE / f"{a.backbone}.keras")
    MODEL_DIR.mkdir(exist_ok=True)

    np.savez_compressed(MODEL_DIR / "tomato_cnn.npz", **fold(core))
    eng = NumpyEngine(MODEL_DIR / "tomato_cnn.npz")

    X, y, split, _ = load_cache()
    idx = np.where(split == "test")[0]
    idx = np.random.default_rng(1).choice(idx, a.n, replace=False)
    xb = X[idx].astype("float32") / 255.0
    ref = core.predict(xb, verbose=0)
    t0 = time.perf_counter(); got = eng.predict_proba(xb); per_img = (time.perf_counter() - t0) / a.n * 1000
    report = {"numpy_vs_keras": {"n": a.n, "max_abs_diff": float(np.abs(ref - got).max()),
                                 "argmax_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
                                 "ms_per_image_on_this_machine": round(per_img, 1)}}
    assert report["numpy_vs_keras"]["max_abs_diff"] < 1e-3, "NumPy export does not match Keras!"

    try:   # TFLite export, validated against Keras
        conv = tf.lite.TFLiteConverter.from_keras_model(core)
        (MODEL_DIR / "tomato.tflite").write_bytes(conv.convert())
        it = tf.lite.Interpreter(model_path=str(MODEL_DIR / "tomato.tflite")); it.allocate_tensors()
        i, o = it.get_input_details()[0], it.get_output_details()[0]
        tfl = []
        for x in xb[:50]:
            it.set_tensor(i["index"], x[None]); it.invoke(); tfl.append(it.get_tensor(o["index"])[0])
        tfl = np.stack(tfl)
        report["tflite_vs_keras"] = {"n": 50, "max_abs_diff": float(np.abs(ref[:50] - tfl).max()),
                                     "argmax_agreement": float((ref[:50].argmax(1) == tfl.argmax(1)).mean())}
    except Exception as e:
        report["tflite_vs_keras"] = f"skipped: {e}"

    info = json.load(open(ROOT / "data" / "diseases.json"))["plants"][PLANT]["diseases"]
    metrics = json.load(open(REPORTS / "metrics.json"))
    thr = json.load(open(REPORTS / "thresholds.json"))
    cfg = {"plant": PLANT, "version": "1.0.0", "backbone": a.backbone, "engine": "numpy",
           "model_file": "tomato_cnn.npz", "input_size": IMG_SIZE, "normalization": NORMALIZATION,
           "resize_mode": RESIZE_MODE, "channel_order": "RGB", "class_keys": CLASS_KEYS,
           "class_labels": {k: info[k]["name"] for k in CLASS_KEYS},
           "thresholds": {"min_confidence": thr["min_confidence"], "min_margin": thr["min_margin"]},
           "dataset": "PlantVillage (tomato, color), leaf-grouped 70/15/15 split",
           "evaluation": {k: metrics[k] for k in ("n_test", "accuracy", "macro_f1", "stress_test_accuracy_simulated", "caveats")}}
    json.dump(cfg, open(MODEL_DIR / "model_config.json", "w"), indent=1)
    json.dump(report, open(REPORTS / "export_validation.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
