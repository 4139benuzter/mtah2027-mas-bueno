"""Learned emissions (v35): a gradient-boosted frame model in place of the hand-built emissions.

v31's emissions are hand-built rules. Here a gradient-boosted classifier, trained on labelled
OTHER matches only (seven DFL and three Metrica matches, their balls degraded to look like the
competition ball, possession/degrade.py), gives P(H), P(A), P(D) for every frame. Its inputs are
the tracking features, v31's own (tag-free) emissions and decoded path, and their context over
+-0.5, 2 and 5 s. The same Viterbi decoder as v31 turns the probabilities into possessions. On the
competition match the Smart Tagging terms of v31 are added to the learned log-probabilities
unchanged (they are zero on the public matches, which have no tags).

The trees are trained with LightGBM (training only; see possession/learned_eval.py) and stored as
plain JSON (calib/learned_model.json). Inference here is numpy only, so regenerating the CSV
needs nothing beyond requirements.txt.
"""

import json
import os

import numpy as np

from . import episodes as EP
from . import features as F
from .model import (Params, emissions, viterbi, merge_short_dead, snap_dead_onsets,
                    gated_team_costs)

FPS = 25
CAP = 250                     # frames; "since" and "until" features are capped at 10 s
WINDOWS = (12, 50, 125)       # +-0.5 s, 2 s, 5 s
MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calib",
                          "learned_model.json")


# --- small helpers ------------------------------------------------------------------------------
def _since(mask, cap=CAP):
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.arange(n) - last
    out[last < 0] = cap
    return np.minimum(out, cap).astype(np.float32)


def _until(mask, cap=CAP):
    return _since(mask[::-1], cap)[::-1].copy()


def attack_sign(trk, halves):
    """(n,) +1 where the home side attacks towards +x, -1 where towards -x (per half, label-free:
    each side's mean x over the half lies in the half it defends)."""
    n = len(trk["ball"])
    sign = np.ones(n, dtype=np.float32)
    home = trk["team"] == 0
    for a, b in halves:
        px = trk["players"][a:b + 1, :, 0]
        hx, ax = np.nanmean(px[:, home]), np.nanmean(px[:, ~home])
        sign[a:b + 1] = 1.0 if ax > hx else -1.0
    return sign


# Swap rule of every feature under exchanging the two teams: ("pair", other), "neg" or "same".
def _swap_spec(names):
    spec = {}
    for name in names:
        if name.endswith("_h") and name[:-2] + "_a" in names:
            spec[name] = ("pair", name[:-2] + "_a")
        elif name.endswith("_a") and name[:-2] + "_h" in names:
            spec[name] = ("pair", name[:-2] + "_h")
        elif name.startswith("NEG_"):
            spec[name] = ("neg",)
        else:
            spec[name] = ("same",)
    return spec


def frame_features(trk, feat, log_e0, path0, share_ep, kind_of, eps, pitch, halves):
    """dict name -> (n,) float32. Names ending _h/_a are team pairs; NEG_ names flip sign."""
    n = len(trk["ball"])
    L, W = pitch
    f = {}
    dh = np.clip(np.nan_to_num(feat["d_home"], nan=40.0), 0, 40)
    da = np.clip(np.nan_to_num(feat["d_away"], nan=40.0), 0, 40)
    f["d_h"], f["d_a"] = dh, da
    f["NEG_ddiff"] = da - dh
    f["dmin"] = np.minimum(dh, da)
    f["v_ball"] = np.nan_to_num(feat["v_ball_s"], nan=-1.0)
    f["v_h"] = np.nan_to_num(feat["v_home"], nan=-1.0)
    f["v_a"] = np.nan_to_num(feat["v_away"], nan=-1.0)
    for key in ("visible", "oob", "hold", "line_carry", "ignored", "kicked", "lost_near_line"):
        f[key] = feat[key].astype(np.float32)
    for key in ("frac_run", "rest_frac", "n_tracked", "activity", "area"):
        f[key] = np.nan_to_num(feat[key], nan=-1.0)
    kt = feat["kick_team"]
    f["NEG_kick"] = np.where(kt == 0, 1.0, np.where(kt == 1, -1.0, 0.0))
    f["NEG_shape"] = np.nan_to_num(feat["shape"], nan=0.0)
    sign = attack_sign(trk, halves)
    bx, by = trk["ball"][:, 0], trk["ball"][:, 1]
    f["NEG_x"] = np.where(np.isfinite(bx), sign * (bx / L - 0.5) * 2, 0.0)
    f["y_abs"] = np.where(np.isfinite(by), np.abs(by / W - 0.5) * 2, -1.0)
    le = np.clip(log_e0, -20, 0)
    f["le_h"], f["le_a"], f["le_d"] = le[:, 0], le[:, 1], le[:, 2]
    f["NEG_share"] = share_ep - 0.5
    f["kind"] = kind_of.astype(np.float32)
    f["p0_h"], f["p0_a"], f["p0_d"] = ((path0 == k).astype(np.float32) for k in (0, 1, 2))
    change = np.concatenate(([True], path0[1:] != path0[:-1]))
    f["p0_since"] = _since(change)
    f["p0_until"] = _until(np.concatenate((change[1:], [True])))
    vis = feat["visible"]
    f["vis_since"], f["vis_until"] = _since(vis), _until(vis)
    for kinds, tag in ((("controlled", "pass"), "c"), (("controlled", "pass", "touch", "unresolved"), "e")):
        for team, side in ((0, "h"), (1, "a")):
            on = np.zeros(n, dtype=bool)
            for s, e, k, kind, _ in eps:
                if k == team and kind in kinds:
                    on[s] = True
            f["since{}_{}".format(tag, side)] = _since(on)
            f["until{}_{}".format(tag, side)] = _until(on)
    base = {"NEG_ddiff": f["NEG_ddiff"], "visible": f["visible"], "NEG_share": f["NEG_share"],
            "frac_run": f["frac_run"], "v_ball": np.nan_to_num(feat["v_ball_s"], nan=0.0),
            "NEG_le": le[:, 0] - le[:, 1], "NEG_p0": f["p0_h"] - f["p0_a"], "p0_d": f["p0_d"]}
    for half in WINDOWS:
        for key, x in base.items():
            name = ("NEG_r{}_{}".format(half, key[4:]) if key.startswith("NEG_")
                    else "r{}_{}".format(half, key))
            f[name] = np.nan_to_num(F.moving_mean(x.astype(np.float64), half), nan=0.0)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def swap_teams(X, names):
    """Copy of X (rows, F) with the two teams exchanged."""
    spec = _swap_spec(names)
    col = {name: j for j, name in enumerate(names)}
    out = X.copy()
    for name, rule in spec.items():
        j = col[name]
        if rule[0] == "pair":
            out[:, j] = X[:, col[rule[1]]]
        elif rule[0] == "neg":
            out[:, j] = -X[:, j]
    return out


def v31_inputs(trk, pitch, kickoffs, graded, fps=FPS, p=Params):
    """Tag-free v31 pieces on one match: feat, log_e0, eps, share, kind, path0."""
    import possession.model as M
    n = len(trk["ball"])
    M.KICKOFF_FRAMES = tuple(kickoffs)
    feat = F.compute(trk, pitch, fps, kickoffs=kickoffs)
    log_e0, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, fps, p)
    inplay = feat["visible"] & ~feat["oob"]
    share, _, _, kind_of = EP.home_share(feat, inplay, fps)
    team_costs = gated_team_costs(diag["eps"], n, fps, p)
    path0 = merge_short_dead(viterbi(log_e0, p.switch_team, p.switch_dead, team_costs), fps, p)
    path0 = snap_dead_onsets(path0, feat, fps, p)
    return feat, log_e0, diag["eps"], share, kind_of, path0


# --- numpy inference of LightGBM trees (exported JSON) -------------------------------------------
def _flatten(tree):
    """Arrays for one tree: feature, threshold, default_left, missing (0 None, 1 Zero, 2 NaN),
    left, right (negative = ~leaf index), leaf values."""
    feats, thr, dl, miss, left, right, leaves = [], [], [], [], [], [], []

    def walk(node):
        if "split_feature" not in node:
            leaves.append(node.get("leaf_value", 0.0))
            return ~(len(leaves) - 1)
        i = len(feats)
        feats.append(node["split_feature"])
        thr.append(node["threshold"])
        dl.append(bool(node["default_left"]))
        miss.append({"None": 0, "Zero": 1, "NaN": 2}[node.get("missing_type", "None")])
        left.append(0)
        right.append(0)
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    root = walk(tree["tree_structure"])
    return {"root": root, "f": np.array(feats, dtype=np.int64), "t": np.array(thr, dtype=np.float64),
            "dl": np.array(dl, dtype=bool), "m": np.array(miss, dtype=np.int8),
            "l": np.array(left, dtype=np.int64), "r": np.array(right, dtype=np.int64),
            "v": np.array(leaves, dtype=np.float64)}


def _tree_predict(tr, X):
    n = len(X)
    if tr["root"] < 0:
        return np.full(n, tr["v"][~tr["root"]])
    node = np.full(n, tr["root"], dtype=np.int64)
    out = np.zeros(n)
    active = np.ones(n, dtype=bool)
    while active.any():
        idx = np.flatnonzero(active)
        nd = node[idx]
        x = X[idx, tr["f"][nd]].astype(np.float64)
        isnan = np.isnan(x)
        m = tr["m"][nd]
        go_left = np.where(isnan, 0.0, x) <= tr["t"][nd]
        zero = (m == 1) & (isnan | (np.abs(x) <= 1e-35))
        go_left = np.where((m == 2) & isnan, tr["dl"][nd], go_left)
        go_left = np.where(zero, tr["dl"][nd], go_left)
        nxt = np.where(go_left, tr["l"][nd], tr["r"][nd])
        leaf = nxt < 0
        out[idx[leaf]] = tr["v"][~nxt[leaf]]
        active[idx[leaf]] = False
        node[idx[~leaf]] = nxt[~leaf]
    return out


def predict_json(model, X):
    """(n, 3) class probabilities from an exported model dict."""
    k = model["num_class"]
    raw = np.zeros((len(X), k))
    for i, tree in enumerate(model["trees"]):
        if "_flat" not in tree:
            tree["_flat"] = _flatten(tree)
        raw[:, i % k] += _tree_predict(tree["_flat"], X)
    raw -= raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)


def decode(prob, extra_log_e, feat, eps, cfg, fps=FPS):
    """Viterbi path from learned probabilities. cfg: T, switch_team, switch_dead, gate."""
    n = len(prob)
    p = type("PL", (Params,), {"switch_team": cfg["switch_team"], "switch_dead": cfg["switch_dead"],
                               "gate_switches": bool(cfg["gate"])})
    log_e = cfg["T"] * np.log(np.clip(prob, 1e-9, 1.0)) + extra_log_e
    team_costs = gated_team_costs(eps, n, fps, p) if p.gate_switches else None
    path = merge_short_dead(viterbi(log_e, p.switch_team, p.switch_dead, team_costs), fps, p)
    return snap_dead_onsets(path, feat, fps, p)


def load_model(path=MODEL_FILE):
    with open(path) as handle:
        return json.load(handle)
