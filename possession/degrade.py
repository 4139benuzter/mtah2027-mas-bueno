"""Ball degradation models and label-free ball statistics, for domain matching.

The competition ball is soft at the moment of contact (median peak change of velocity at a
contact onset about 3 m/s, against 7-12 m/s on the public matches), but its speeds and its
distances to the players look like the public balls as recorded. A model trained on the public
balls therefore meets inputs it never saw. Here the public ball is degraded by a simple tracker
model until a vector of label-free statistics matches the competition ball's; the filter can then
be trained and tested on degraded public balls.

Nothing here reads labels. `target` reads only the competition tracking and prints summary
statistics; `fit` reads only public matches.

    python -m possession.degrade target                      # competition statistics (summary only)
    python -m possession.degrade fit --family box --grid "h=0,2,4;noise=0,0.2"
"""

import itertools
import json
import os
import sys
import warnings

import numpy as np

from . import episodes as EP
from . import features
from . import touches as T

FPS = 25
SEED = 0
STATS = ("pk25", "pk50", "pk75", "lag50", "dv50", "dv90", "v50", "v90", "vraw99", "zero", "acc99",
         "d10", "d15", "d25", "ond", "epm", "eplen")
SHARES = ("zero", "d10", "d15", "d25")
TARGET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib",
                           "domain_target.json")


# --- degradation families -------------------------------------------------------------------
def _gauss(x, sigma):
    """Centred Gaussian smoothing ignoring NaN; NaN stays NaN."""
    if sigma <= 0:
        return x.copy()
    half = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    ok = ~np.isnan(x)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    out[~ok] = np.nan
    return out


def _ema(x, alpha):
    """Causal exponential smoothing; restarts after a gap. NaN stays NaN."""
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            prev = np.nan
            continue
        prev = v if np.isnan(prev) else prev + alpha * (v - prev)
        out[i] = prev
    return out


def _kalman(pos, q, r, fps=FPS):
    """Causal constant-velocity Kalman filter on one axis (q: m/s^2 process, r: m measurement)."""
    dt = 1.0 / fps
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    out = np.full(len(pos), np.nan)
    x, P = None, None
    for i, z in enumerate(pos):
        if np.isnan(z):
            x = None
            continue
        if x is None:
            x, P = np.array([z, 0.0]), np.diag([r ** 2, 25.0])
        else:
            x, P = F @ x, F @ P @ F.T + Q
            s = P[0, 0] + r ** 2
            g = P[:, 0] / s
            x = x + g * (z - x[0])
            P = P - np.outer(g, P[0, :])
        out[i] = x[0]
    return out


def _subsample(x, k):
    """Keep every k-th frame and interpolate linearly between kept frames (gaps stay NaN)."""
    if k <= 1:
        return x.copy()
    idx = np.arange(len(x))
    keep = (idx % k == 0) & ~np.isnan(x)
    if keep.sum() < 2:
        return x.copy()
    out = np.interp(idx, idx[keep], x[keep])
    out[np.isnan(x)] = np.nan
    return out


# --- cnoise / nohold (found by the fit workflow on 2026-11-17; ported unchanged) -----------
# The DFL ball holds its last position exactly during stoppages (runs of 30-45 s); the
# competition ball has no such holds (an automatic tracker reports no ball, or a moving one).
# nohold makes holds of >= m frames missing, then applies cnoise: smoothing (Gaussian s, with a
# lighter Gaussian sf blended in above v0 m/s over vw m/s), temporally correlated noise added
# after the smoothing (white noise filtered by a unit-energy Gaussian of tn frames, std post m),
# output rounded to dq m, then lag.
def _hold_mask(ball, m):
    """(n,) bool: frames that repeat the previous position exactly, in runs of >= m such frames."""
    same = np.zeros(len(ball), dtype=bool)
    same[1:] = np.all(ball[1:] == ball[:-1], axis=1) & ~np.isnan(ball[1:, 0])
    edges = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    out = np.zeros(len(ball), dtype=bool)
    for s0, s1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s1 - s0 >= m:
            out[s0:s1] = True
    return out


def _smooth_axis(x, params):
    if "h" in params:
        return features.moving_mean(x, int(params["h"]))
    return _gauss(x, float(params.get("s", 0.0)))


def _adaptive(b, params):
    heavy = np.stack([_smooth_axis(b[:, j], params) for j in range(2)], axis=1)
    if "sf" not in params:
        return heavy
    light = np.stack([_gauss(b[:, j], float(params["sf"])) for j in range(2)], axis=1)
    sp = features.moving_mean(features.ball_speed(heavy, FPS), 2)
    w = np.clip((np.nan_to_num(sp, nan=0.0) - float(params["v0"])) / float(params.get("vw", 4.0)), 0.0, 1.0)
    return heavy + w[:, None] * (light - heavy)


def _gauss_noise(white, tn):
    if tn <= 0:
        return white
    half = int(np.ceil(4 * tn))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / tn) ** 2)
    k /= np.sqrt((k ** 2).sum())
    return np.convolve(white, k, mode="same")


def _shift(out, lag):
    lag = int(lag)
    if lag > 0:
        return np.vstack([np.full((lag, 2), np.nan), out[:-lag]])
    if lag < 0:
        return np.vstack([out[-lag:], np.full((-lag, 2), np.nan)])
    return out


def _cnoise(ball, params, seed):
    rng = np.random.default_rng(seed)
    b = ball.astype(np.float64).copy()
    missing = np.isnan(b[:, 0])
    w_pre, w_post = rng.normal(0.0, 1.0, b.shape), rng.normal(0.0, 1.0, b.shape)
    b = b + float(params.get("pre", 0.0)) * w_pre
    post, tn = float(params.get("post", 0.0)), float(params.get("tn", 0.0))
    sm = _adaptive(b, params)
    cols = []
    for j in range(2):
        y = sm[:, j]
        if post > 0:
            y = y + post * _gauss_noise(w_post[:, j], tn)
        cols.append(y)
    out = np.stack(cols, axis=1)
    dq = float(params.get("dq", 0.0))
    if dq > 0:
        out = np.round(out / dq) * dq
    out[missing] = np.nan
    out = _shift(out, params.get("lag", 0))
    out[missing] = np.nan
    return out


def _nohold(ball, params, seed):
    b = ball.astype(np.float64).copy()
    hold = _hold_mask(b, int(params.get("m", 10)))
    pad = int(params.get("pad", 0))
    if pad > 0:
        for e in np.flatnonzero(hold[:-1] & ~hold[1:]) + 1:
            hold[e:e + pad] = True
    b[hold] = np.nan
    return _cnoise(b, params, seed)


def apply(ball, family, params, seed=SEED):
    """Degraded copy of ball (n, 2). Order: position noise, family filter, lag (ball later)."""
    if family == "nohold":
        return _nohold(ball, params, seed)
    if family == "cnoise":
        return _cnoise(ball, params, seed)
    rng = np.random.default_rng(seed)
    b = ball.astype(np.float64).copy()
    missing = np.isnan(b[:, 0])
    noise = float(params.get("noise", 0.0))
    if noise > 0:
        b = b + rng.normal(0.0, noise, b.shape)
    cols = []
    for j in range(2):
        x = b[:, j]
        if family == "none":
            y = x
        elif family == "box":
            y = features.moving_mean(x, int(params["h"]))
        elif family == "gauss":
            y = _gauss(x, float(params["s"]))
        elif family == "ema":
            y = _ema(x, float(params["a"]))
        elif family == "kalman":
            y = _kalman(x, float(params["q"]), float(params["r"]))
        elif family == "sub":
            y = features.moving_mean(_subsample(x, int(params["k"])), int(params.get("h", 0)))
        else:
            raise ValueError("unknown family " + family)
        cols.append(y)
    out = np.stack(cols, axis=1)
    out[missing] = np.nan
    lag = int(params.get("lag", 0))
    if lag > 0:
        out = np.vstack([np.full((lag, 2), np.nan), out[:-lag]])
    return out


# --- label-free statistics ------------------------------------------------------------------
def stats(trk, pitch, kickoffs, graded, fps=FPS):
    """dict of the STATS for one match. graded: (n,) bool of frames to use (no labels needed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat = features.compute(trk, pitch, fps, kickoffs=kickoffs)
        dv = T.delta_v(T.ball_velocity(trk["ball"], fps))
        vraw = features.ball_speed(trk["ball"], fps)
    inplay = feat["visible"] & ~feat["oob"] & graded
    n = len(inplay)
    dmin = np.fmin(feat["d_home"], feat["d_away"])
    eps = EP.episodes_from(EP.contact_team(feat, inplay))
    x = np.nan_to_num(dv, nan=0.0)
    vis = feat["visible"]
    pk, lag, ond = [], [], []
    for s, e, k in eps:
        if s < 15 or s + 15 >= n or not vis[s - 8:s + 9].all():
            continue
        seg = x[s - 15:s + 16]
        pk.append(x[s - 8:s + 9].max())
        lag.append(int(np.argmax(seg)) - 15)
        ond.append(dmin[s])
    step = np.hypot(*(trk["ball"][1:] - trk["ball"][:-1]).T)
    ip1 = inplay[1:] & inplay[:-1] & np.isfinite(step)
    vb = feat["v_ball_s"]
    acc = np.abs(np.diff(vraw))
    okacc = inplay[1:] & np.isfinite(acc)
    dmi = dmin[inplay & np.isfinite(dmin)]
    mins = inplay.sum() / fps / 60.0
    okdv = inplay & np.isfinite(dv)
    okv = inplay & np.isfinite(vb)
    okr = inplay & np.isfinite(vraw)
    return {
        "pk25": float(np.percentile(pk, 25)), "pk50": float(np.median(pk)), "pk75": float(np.percentile(pk, 75)),
        "lag50": float(np.median(lag)),
        "dv50": float(np.median(dv[okdv])), "dv90": float(np.percentile(dv[okdv], 90)),
        "v50": float(np.median(vb[okv])), "v90": float(np.percentile(vb[okv], 90)),
        "vraw99": float(np.percentile(vraw[okr], 99)),
        "zero": float((step[ip1] < 1e-6).mean()),
        "acc99": float(np.percentile(acc[okacc], 99)),
        "d10": float((dmi < 1.0).mean()), "d15": float((dmi < 1.5).mean()), "d25": float((dmi < 2.5).mean()),
        "ond": float(np.median(ond)),
        "epm": len(eps) / mins, "eplen": float(np.median([e - s for s, e, k in eps])),
    }


def distance(st, target):
    """Mean squared log-ratio over STATS (shares offset by 0.01; lag50 as difference / 3 frames)."""
    terms = []
    for key in STATS:
        a, b = st[key], target[key]
        if key == "lag50":
            terms.append(((a - b) / 3.0) ** 2)
        elif key in SHARES:
            terms.append(np.log((a + 0.01) / (b + 0.01)) ** 2)
        else:
            terms.append(np.log(max(a, 1e-3) / max(b, 1e-3)) ** 2)
    return float(np.mean(terms))


# --- command line -----------------------------------------------------------------------------
def _competition_stats():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from games import GAME, GAMES, half_video_blocks
    from possession.atd import load_tracking
    from possession.keepers import fix_keepers
    from possession.model import KICKOFF_FRAMES
    from possession.run import graded_mask
    g = GAMES[GAME]
    trk = fix_keepers(load_tracking(), half_video_blocks(GAME), *g["pitch"])
    return stats(trk, g["pitch"], KICKOFF_FRAMES, graded_mask(g["n_video_frames"]), g["fps"])


def _public(source):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "evaluation"))
    from possession import touch_eval as E
    from possession import sample_game
    ms = E.fine_matches()
    ms = [m for m in ms if (m.startswith("metrica") if source == "metrica" else
                            (not m.startswith("metrica")) if source == "dfl" else True)]
    out = []
    for m in ms:
        trk, labels, info = E.load(m)
        out.append((E.name_of(m), trk, info.get("pitch", sample_game.PITCH), info["kickoffs"], labels != "X"))
    return out


def _grid(spec):
    keys, vals = [], []
    for part in spec.split(";"):
        k, v = part.split("=")
        keys.append(k.strip())
        vals.append([float(x) for x in v.split(",")])
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def pooled(per_match):
    """Median of each statistic over matches."""
    return {k: float(np.median([s[k] for s in per_match])) for k in STATS}


def evaluate(matches, family, params):
    per = []
    for name, trk, pitch, kick, graded in matches:
        t = dict(trk)
        t["ball"] = apply(trk["ball"], family, params)
        per.append(stats(t, pitch, kick, graded))
    return pooled(per)


def main(argv):
    if argv[0] == "target":
        st = _competition_stats()
        print(json.dumps({k: round(v, 4) for k, v in st.items()}))
        return
    if argv[0] == "fit":
        family = argv[argv.index("--family") + 1]
        grid = _grid(argv[argv.index("--grid") + 1]) if "--grid" in argv else [{}]
        source = argv[argv.index("--source") + 1] if "--source" in argv else "dfl"
        target = json.load(open(TARGET_FILE))
        matches = _public(source)
        for params in grid:
            st = evaluate(matches, family, params)
            print(json.dumps({"family": family, "params": params, "D": round(distance(st, target), 5),
                              "stats": {k: round(v, 3) for k, v in st.items()}}), flush=True)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
