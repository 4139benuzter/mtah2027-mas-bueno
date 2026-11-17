"""Short possessions from brief contacts and kicks: a learned filter on top of the decoder.

The decoder (model.py) changes side cheaply only at the start of a control episode. That is
right for the definition, but it misses most possessions shorter than 2 s: on the three Metrica
matches 44% of them start with no contact episode at all (a first-time kick or deflection moves
the ball too fast to count as contact) and 32% start with a brief contact that the episode rules
treat as no evidence. Taken as changes of side, such contacts are right only about one time in
three. So a filter must decide which of them are real.

Candidates are every contact episode (episodes.py) and every kick: a sharp change of the ball's
velocity next to a player. Each candidate gets a few features (its kind and length, the change
of velocity, the distances to both sides, the sides of the candidates before and after it, and
what the decoder itself says around it). A logistic regression, trained on labelled OTHER
matches only, gives the probability that the candidate starts a true possession of its side.
Where that probability is at least tau and the decoder says the other side has the ball, the
candidate's side is inserted from its onset to the next candidate of the other side (at most
max_len_s later).

numpy only; the fitted weights are stored as plain numbers (calib/touch_weights.json).
"""

import json
import os

import numpy as np

from .features import moving_mean

FPS = 25
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib",
                       "touch_weights.json")
KINDS = ("controlled", "pass", "touch", "unresolved", "kick")


class TouchParams:
    dv_min = 4.0          # m/s, change of the ball's velocity over 6 frames that counts as a kick
    kick_radius = 2.0     # m, a kick needs a player this close
    nms = 5               # frames, one kick per +- this many frames
    tol = 12              # frames, a candidate "starts" a true possession within +- this
    max_len_s = 4.0       # s, longest possession the filter may insert
    l2 = 1.0              # ridge penalty of the logistic regression (standardised features)
    tau = 0.5             # probability threshold


def ball_velocity(ball, fps=FPS):
    """(n, 2) m/s, centred differences smoothed over +- 1 frame; NaN where unknown."""
    n = len(ball)
    v = np.full((n, 2), np.nan)
    v[1:-1] = (ball[2:] - ball[:-2]) * fps / 2.0
    return np.stack([moving_mean(v[:, 0], 1), moving_mean(v[:, 1], 1)], axis=1)


def delta_v(vel):
    """(n,) |v(t+3) - v(t-3)|, NaN where either side is unknown."""
    n = len(vel)
    dv = np.full(n, np.nan)
    d = vel[6:] - vel[:-6]
    dv[3:-3] = np.hypot(d[:, 0], d[:, 1])
    return dv


def kick_candidates(feat, dv, eps, p=TouchParams):
    """[(s, e, team, 'kick')] at local maxima of dv next to a player, outside same-side episodes."""
    d_home = np.nan_to_num(feat["d_home"], nan=np.inf)
    d_away = np.nan_to_num(feat["d_away"], nan=np.inf)
    inplay = feat["visible"] & ~feat["oob"]
    x = np.nan_to_num(dv, nan=0.0)
    n = len(x)
    spans = {0: [(es, ee) for es, ee, ek, _, _ in eps if ek == 0],
             1: [(es, ee) for es, ee, ek, _, _ in eps if ek == 1]}
    starts = {k: np.array([a for a, _ in spans[k]]) for k in (0, 1)}
    stops = {k: np.array([b for _, b in spans[k]]) for k in (0, 1)}
    out = []
    for t in np.flatnonzero((x > p.dv_min) & inplay):
        lo, hi = max(0, t - p.nms), min(n, t + p.nms + 1)
        if x[t] < x[lo:hi].max() or (x[lo:t] == x[t]).any():
            continue
        w0, w1 = max(0, t - 2), min(n, t + 3)
        dh, da = d_home[w0:w1].min(), d_away[w0:w1].min()
        if min(dh, da) > p.kick_radius:
            continue
        k = 0 if dh <= da else 1
        if len(starts[k]) and np.any((starts[k] - 5 <= t) & (t <= stops[k] + 5)):
            continue
        out.append((int(t), int(t) + 1, k, "kick"))
    return out


def candidates(feat, eps, dv, p=TouchParams):
    """All candidates sorted by onset: (s, e, team, kind)."""
    cands = [(int(s), int(e), int(k), kind) for s, e, k, kind, _ in eps]
    cands += kick_candidates(feat, dv, eps, p)
    return sorted(cands)


def _first_after(arr, t):
    i = np.searchsorted(arr, t, side="right")
    return arr[i] if i < len(arr) else None


def _last_before(arr, t):
    i = np.searchsorted(arr, t, side="left") - 1
    return arr[i] if i >= 0 else None


def featurize(cands, feat, dv, path, fps=FPS):
    """(m, F) feature matrix, one row per candidate."""
    n = len(path)
    d_home = np.clip(np.nan_to_num(feat["d_home"], nan=9.0), 0, 9.0)
    d_away = np.clip(np.nan_to_num(feat["d_away"], nan=9.0), 0, 9.0)
    speed = np.nan_to_num(feat["v_ball_s"], nan=0.0)
    dvz = np.nan_to_num(dv, nan=0.0)
    vis_s = np.nan_to_num(moving_mean(feat["visible"].astype(float), fps), nan=0.0)
    on = np.array([c[0] for c in cands])
    team = np.array([c[2] for c in cands])
    on_by = {k: on[team == k] for k in (0, 1)}
    path_on = {k: np.flatnonzero(path == k) for k in (0, 1)}
    cap = 5 * fps
    win = 3 * fps
    rows = []
    for i, (s, e, k, kind) in enumerate(cands):
        o = 1 - k
        own, opp = (d_home, d_away) if k == 0 else (d_away, d_home)
        w0, w1 = max(0, s - 2), min(n, s + 3)
        has_prev, has_next = i > 0, i + 1 < len(cands)
        nxt_k = _first_after(path_on[k], s)
        lst_k = _last_before(path_on[k], s)
        b0, a0 = max(0, s - 6), min(n, e + 1)
        a1 = min(n, e + 7)
        rows.append([float(kind == name) for name in KINDS] + [
            np.log1p(e - s),
            dvz[max(0, s - 3):min(n, s + 4)].max() / 10.0,
            speed[b0:s].mean() / 10.0 if s > b0 else 0.0,
            speed[a0:a1].mean() / 10.0 if a1 > a0 else 0.0,
            own[w0:w1].min(), opp[w0:w1].min(), opp[w0:w1].min() - own[w0:w1].min(),
            float(has_prev and cands[i - 1][2] == k),
            min(cap, s - cands[i - 1][0]) / fps if has_prev else cap / fps,
            float(has_prev and cands[i - 1][3] in ("controlled", "pass")),
            float(has_next and cands[i + 1][2] == k),
            min(cap, cands[i + 1][0] - s) / fps if has_next else cap / fps,
            float(i + 2 < len(cands) and cands[i + 2][2] == k),
            np.sum((on_by[k] > s) & (on_by[k] <= s + win)),
            np.sum((on_by[o] > s) & (on_by[o] <= s + win)),
            np.sum((on_by[k] < s) & (on_by[k] >= s - win)),
            np.sum((on_by[o] < s) & (on_by[o] >= s - win)),
            min(cap, (nxt_k - s) if nxt_k is not None else cap) / fps,
            min(cap, (s - lst_k) if lst_k is not None else cap) / fps,
            vis_s[s],
        ])
    return np.array(rows, dtype=np.float64).reshape(len(cands), -1)


def targets(cands, gt_spells, p=TouchParams):
    """(m,) 1 where a true possession of the candidate's side starts within +- tol of its onset."""
    starts = {k: np.array([s for s, e, c in gt_spells if c == k]) for k in (0, 1)}
    y = np.zeros(len(cands))
    for i, (s, e, k, kind) in enumerate(cands):
        st = starts[k]
        if len(st) and np.min(np.abs(st - s)) <= p.tol:
            y[i] = 1.0
    return y


def acted(cands, path):
    """(m,) bool: the decoder gives the ball to the other side (not D) at the candidate's onset."""
    return np.array([path[s] in (0, 1) and path[s] != k for s, e, k, kind in cands], dtype=bool)


def fit_logistic(X, y, l2=TouchParams.l2, iters=50):
    """Ridge logistic regression by Newton's method on standardised features. Returns a dict."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (q - y) + reg * w
        hess = (Z * (q * (1 - q))[:, None]).T @ Z + np.diag(reg) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def predict(model, X):
    if model.get("cols") is not None:
        X = X[:, model["cols"]]       # a variant trained on a subset of the features
    mu, sd, w = np.array(model["mu"]), np.array(model["sd"]), np.array(model["w"])
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-(Z @ w)))


def insert(path, cands, prob, tau, fps=FPS, p=TouchParams):
    """Copy of path with short possessions inserted at accepted candidates."""
    out = path.copy()
    max_len = int(round(p.max_len_s * fps))
    on = np.array([c[0] for c in cands])
    team = np.array([c[2] for c in cands])
    for i, (s, e, k, kind) in enumerate(cands):
        if prob[i] < tau or out[s] not in (0, 1) or out[s] == k:
            continue
        later = np.flatnonzero((team == 1 - k) & (on > max(s, e - 1)) & (on <= s + max_len))
        if not len(later):
            continue
        end = int(on[later[0]])
        dead = np.flatnonzero(out[s:end] == 2)
        if len(dead):
            end = s + int(dead[0])
        out[s:end] = k
    return out


def apply(ball, feat, eps, path, fps=FPS, weights=WEIGHTS):
    """The decoder's path with the filter applied, using the stored weights and threshold."""
    with open(weights) as handle:
        model = json.load(handle)
    with np.errstate(all="ignore"):
        dv = delta_v(ball_velocity(ball, fps))
    cands = candidates(feat, eps, dv)
    if not cands:
        return path
    prob = predict(model, featurize(cands, feat, dv, path, fps))
    return insert(path, cands, prob, model["tau"], fps)
