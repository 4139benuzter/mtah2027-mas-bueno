"""Numpy inference of the bidirectional GRU side model (calib/gru_model.npz); no PyTorch needed.

Same equations as torch.nn.GRU (gate order r, z, n). Input: the 67 per-frame features of
possession/learned.py, standardised with the training mean and std stored in the model file.
Prediction over overlapping windows of WIN frames (stride WIN // 2), averaged, as in training.
"""
import os

import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib", "gru_model.npz")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gru_dir(x, w_ih, w_hh, b_ih, b_hh, reverse):
    """x: (B, T, F) -> (B, T, H)."""
    B, T, _ = x.shape
    H = w_hh.shape[1]
    gi = x @ w_ih.T + b_ih                      # (B, T, 3H)
    h = np.zeros((B, H))
    out = np.empty((B, T, H))
    steps = range(T - 1, -1, -1) if reverse else range(T)
    for t in steps:
        gh = h @ w_hh.T + b_hh
        r = _sig(gi[:, t, :H] + gh[:, :H])
        z = _sig(gi[:, t, H:2 * H] + gh[:, H:2 * H])
        n = np.tanh(gi[:, t, 2 * H:] + r * gh[:, 2 * H:])
        h = (1.0 - z) * n + z * h
        out[:, t] = h
    return out


def _forward(m, x):
    h = np.maximum(x @ m["inp.0.weight"].T + m["inp.0.bias"], 0.0)
    for layer in range(int(m["layers"])):
        k = "gru.weight_ih_l%d" % layer
        fw = _gru_dir(h, m[k], m["gru.weight_hh_l%d" % layer], m["gru.bias_ih_l%d" % layer], m["gru.bias_hh_l%d" % layer], False)
        bw = _gru_dir(h, m[k + "_reverse"], m["gru.weight_hh_l%d_reverse" % layer],
                      m["gru.bias_ih_l%d_reverse" % layer], m["gru.bias_hh_l%d_reverse" % layer], True)
        h = np.concatenate([fw, bw], axis=2)
    logits = h @ m["out.weight"].T + m["out.bias"]
    e = np.exp(logits - logits.max(axis=2, keepdims=True))
    return e / e.sum(axis=2, keepdims=True)


def load(path=MODEL_FILE):
    with np.load(path) as d:
        return {k: d[k].astype(np.float64) for k in d.files}


def predict(m, X):
    """(n, 67) features -> (n, 3) probabilities P(H), P(A), P(D)."""
    Z = np.nan_to_num((X - m["mu"]) / m["sd"], nan=0.0, posinf=0.0, neginf=0.0)
    n, win = len(Z), int(m["win"])
    stride = win // 2
    starts = list(range(0, max(n - win, 0) + 1, stride))
    if starts[-1] + win < n:
        starts.append(n - win)
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    for b in range(0, len(starts), 128):
        st = starts[b:b + 128]
        p = _forward(m, np.stack([Z[a:a + win] for a in st]))
        for a, q in zip(st, p):
            acc[a:a + win] += q
            cnt[a:a + win] += 1
    return acc / np.maximum(cnt, 1)[:, None]
