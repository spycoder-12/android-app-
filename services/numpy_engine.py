"""Pure-NumPy CNN inference. Needs only numpy, so it works inside a python-for-android APK
without a TensorFlow/TFLite runtime. Weights come from training/convert_model.py (BatchNorm
already folded into the convolutions)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class NumpyEngine:
    name = "numpy"

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.convs = []
        i = 0
        while f"conv{i}_w" in z:
            self.convs.append((z[f"conv{i}_w"].astype(np.float32), z[f"conv{i}_b"].astype(np.float32),
                               bool(z[f"conv{i}_pool"])))
            i += 1
        self.dense_w = z["dense_w"].astype(np.float32)
        self.dense_b = z["dense_b"].astype(np.float32)
        self.num_classes = self.dense_b.shape[0]

    @staticmethod
    def _conv3x3(x, w, b):
        h, wd, c = x.shape
        pad = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        win = sliding_window_view(pad, (3, 3), axis=(0, 1))        # (H, W, C, 3, 3)
        cols = win.transpose(0, 1, 3, 4, 2).reshape(h * wd, 9 * c)  # (H*W, kh*kw*C)
        return (cols @ w.reshape(9 * c, -1) + b).reshape(h, wd, -1)

    def predict_proba(self, batch):
        """batch: float32 (N, H, W, 3), already preprocessed. Returns (N, classes) probabilities."""
        outs = []
        for x in batch:
            for w, b, pool in self.convs:
                x = np.maximum(self._conv3x3(x, w, b), 0.0)
                if pool:
                    h, wd, c = x.shape
                    x = x[: h // 2 * 2, : wd // 2 * 2].reshape(h // 2, 2, wd // 2, 2, c).max(axis=(1, 3))
            logits = x.mean(axis=(0, 1)) @ self.dense_w + self.dense_b
            e = np.exp(logits - logits.max())
            outs.append(e / e.sum())
        return np.stack(outs)
