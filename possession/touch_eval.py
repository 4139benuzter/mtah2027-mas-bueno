"""Held-out test of the short-possession filter (possession/touches.py) on labelled OTHER matches.

    python -m possession.touch_eval build  [--jobs 8]   # decode + candidates, cached in cache/touch/
    python -m possession.touch_eval diag                # classifier precision/recall, leave one match out
    python -m possession.touch_eval select              # the rule of calib/touch_rule.md
    python -m possession.touch_eval final               # weights on all ten, SkillCorner report

Matches: Metrica sample games 1-3 and seven DFL matches (three ball conditions each: as recorded,
59% visibility with seeds 0 and 1), plus 19 SkillCorner matches (as recorded) for the report.
Nothing here reads the competition match.
"""

import glob
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import possession.model as M  # noqa: E402
from possession import features, sample_game, touches as T  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              snap_dead_onsets, gated_team_costs)
from possession.validate import drop_ball  # noqa: E402

FPS = 25
EXT = os.environ.get("MTAH_EXTERNAL", os.path.join(os.path.dirname(ROOT), "external"))
# Variants: ball conditions (name, visibility, seed) plus a smoothing half-width in frames applied
# to the ball after any removal, and the kick threshold. "raw" is the first test (v32);
# "smooth" matches the competition ball, whose velocity changes at contacts are soft (median peak
# 3.0 m/s against 9-12 m/s on Metrica and DFL; label-free, from the tracking input).
VARIANTS = {
    "raw": {"conds": (("rec", None, 0), ("v59s0", 0.59, 0), ("v59s1", 0.59, 1)), "smooth": 0, "dv_min": 4.0},
    "smooth": {"conds": (("s10rec", None, 0), ("s10v59s0", 0.59, 0), ("s10v59s1", 0.59, 1)),
               "smooth": 10, "dv_min": 4.0},
    "smooth_k2": {"conds": (("s10rec", None, 0), ("s10v59s0", 0.59, 0), ("s10v59s1", 0.59, 1)),
                  "smooth": 10, "dv_min": 2.0},
    # v32b (calib/touch_rule_nodv.md): the raw cache, without the velocity-change feature (column 6)
    "nodv": {"conds": (("rec", None, 0), ("v59s0", 0.59, 0), ("v59s1", 0.59, 1)), "smooth": 0, "dv_min": 4.0,
             "cache": "touch", "drop": (6,)},
}
VARIANT = os.environ.get("TOUCH_VARIANT", "raw")
_V = VARIANTS[VARIANT]
CACHE = os.path.join(ROOT, "cache", _V.get("cache", "touch" if VARIANT == "raw" else "touch_" + VARIANT))
CONDS = _V["conds"]
T.TouchParams.dv_min = _V["dv_min"]
N_FEATURES = 25
COLS = [j for j in range(N_FEATURES) if j not in _V["drop"]] if "drop" in _V else None
SELECT = os.path.join(CACHE, "select.json" if COLS is None else "select_{}.json".format(VARIANT))
EXCLUDED_SC = {"SC2017461"}
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 2.0)    # 2.0 = filter off (v31 unchanged)
WEIGHTS = os.path.join(ROOT, "calib", "touch_weights.json" if COLS is None else
                       "touch_weights_{}.json".format(VARIANT))


def fine_matches():
    dfl = sorted(glob.glob(os.path.join(EXT, "dfl-npz", "*.npz")))
    return ["metrica1", "metrica2", "metrica3"] + dfl


def sc_matches():
    paths = sorted(glob.glob(os.path.join(EXT, "skillcorner-npz", "*.npz")))
    return [p for p in paths if os.path.basename(p)[:-4] not in EXCLUDED_SC]


def name_of(match):
    return match if match.startswith("metrica") else os.path.basename(match)[:-4]


def load(match):
    if match.startswith("metrica"):
        g = int(match[-1])
        folder = os.path.join(EXT, "metrica-sample-game3" if g == 3 else "metrica-sample-game12")
        return sample_game.load(folder, g)
    return sample_game.load(match)


def smooth_ball(ball, half):
    """Centred moving mean of the ball over +- half frames, kept missing where it was missing."""
    out = np.stack([features.moving_mean(ball[:, 0], half), features.moving_mean(ball[:, 1], half)], axis=1)
    out[np.isnan(ball[:, 0])] = np.nan
    return out


def decode_v31(trk, labels, info):
    """(path, feat, eps): exactly possession/validate.py's decode, keeping the intermediates."""
    n = len(labels)
    graded = labels != "X"
    M.KICKOFF_FRAMES = info["kickoffs"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS, kickoffs=info["kickoffs"])
        log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, Params)
    team_costs = gated_team_costs(diag["eps"], n, FPS, Params)
    path = merge_short_dead(viterbi(log_e, Params.switch_team, Params.switch_dead, team_costs), FPS, Params)
    return snap_dead_onsets(path, feat, FPS, Params), feat, diag["eps"]


def build_one(job):
    match, cond, vis, seed = job
    out = os.path.join(CACHE, "{}__{}.npz".format(name_of(match), cond))
    if os.path.isfile(out):
        return out
    trk, labels, info = load(match)
    if vis:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", vis, seed)
    half = VARIANTS[VARIANT]["smooth"] if not match.endswith("npz") or "SC" not in os.path.basename(match) else 0
    if half:
        trk["ball"] = smooth_ball(trk["ball"], half)
    gt = pm.label_codes(labels)
    path, feat, eps = decode_v31(trk, labels, info)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dv = T.delta_v(T.ball_velocity(trk["ball"]))
    cands = T.candidates(feat, eps, dv)
    X = T.featurize(cands, feat, dv, path)
    y = T.targets(cands, pm.possession_spells(gt, pm.valid_runs(gt)))
    c = np.array([[s, e, k, T.KINDS.index(kind)] for s, e, k, kind in cands], dtype=np.int64).reshape(-1, 4)
    np.savez_compressed(out, gt=gt, path=path, cands=c, X=X, y=y, act=T.acted(cands, path))
    return out


def jobs_for(matches, conds):
    return [(m, c, v, s) for m in matches for c, v, s in conds]


def cached(match, cond):
    d = np.load(os.path.join(CACHE, "{}__{}.npz".format(name_of(match), cond)))
    cands = [(int(s), int(e), int(k), T.KINDS[int(i)]) for s, e, k, i in d["cands"]]
    return {"gt": d["gt"], "path": d["path"], "cands": cands, "X": d["X"], "y": d["y"], "act": d["act"]}


def score(gt, path):
    pred = np.full(len(gt), pm.X, dtype=np.int8)
    graded = gt != pm.X
    pred[graded] = path[graded]
    return pm.evaluate(gt, pred)["spell_f1_mean"]


def train(data, keys):
    X = np.vstack([data[k]["X"][data[k]["act"]] for k in keys])
    y = np.concatenate([data[k]["y"][data[k]["act"]] for k in keys])
    if COLS is None:
        return T.fit_logistic(X, y)
    model = T.fit_logistic(X[:, COLS], y)
    model["cols"] = COLS
    return model


def s_of(data, match, model, tau):
    """Mean spellF1 over the conditions of one match with the filter at tau (2.0 = off)."""
    vals = []
    for cond in [k[1] for k in data if k[0] == match]:
        d = data[(match, cond)]
        path = d["path"]
        if tau <= 1.0 and len(d["cands"]):
            path = T.insert(path, d["cands"], T.predict(model, d["X"]), tau)
        vals.append(score(d["gt"], path))
    return float(np.mean(vals))


def heldout_scores(data, matches, model_for):
    """s[m][tau] with the model for m trained without m (model_for(m) -> model)."""
    return {m: {tau: s_of(data, m, model_for(m), tau) for tau in TAUS} for m in matches}


def pick_tau(s, matches):
    J = {tau: np.mean([s[m][tau] for m in matches]) for tau in TAUS}
    return max(TAUS, key=lambda t: (round(J[t], 6), t > 1)), J   # an exact tie keeps the filter off


def load_all(matches, conds):
    return {(name_of(m), c): cached(m, c) for m in matches for c in conds}


def cmd_build(args):
    os.makedirs(CACHE, exist_ok=True)
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 4
    work = jobs_for(fine_matches(), CONDS)
    if "--fine-only" not in args:
        work += jobs_for(sc_matches(), CONDS[:1])
    with ProcessPoolExecutor(n_jobs) as pool:
        for out in pool.map(build_one, work):
            print("cached", os.path.basename(out), flush=True)


def cmd_diag(_):
    names = [name_of(m) for m in fine_matches()]
    data = load_all(fine_matches(), [c for c, _, _ in CONDS])
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        tp = fp = pos = 0
        for m in names:
            model = train(data, [k for k in data if k[0] != m])
            for cond in [c for c, _, _ in CONDS]:
                d = data[(m, cond)]
                q = T.predict(model, d["X"][d["act"]])
                yy = d["y"][d["act"]]
                tp += int(((q >= tau) & (yy == 1)).sum())
                fp += int(((q >= tau) & (yy == 0)).sum())
                pos += int((yy == 1).sum())
        print("tau {:.1f}: accepted {:5d}  precision {:.2f}  recall of true starts among acted {:.2f}".format(
            tau, tp + fp, tp / max(1, tp + fp), tp / max(1, pos)))


def cmd_select(_):
    names = [name_of(m) for m in fine_matches()]
    assert len(names) == 10, names
    data = load_all(fine_matches(), [c for c, _, _ in CONDS])
    gains, picks = {}, {}
    for m in names:
        inner = [x for x in names if x != m]
        # tau chosen on the other nine, each of them predicted by a model trained without it and m
        s_in = heldout_scores(data, inner, lambda x: train(data, [k for k in data if k[0] not in (m, x)]))
        tau, _ = pick_tau(s_in, inner)
        model = train(data, [k for k in data if k[0] != m])
        gains[m] = s_of(data, m, model, tau) - s_of(data, m, model, 2.0)
        picks[m] = tau
        print("  held out {:12s} tau {}  gain {:+.4f}".format(m, tau if tau <= 1 else "off", gains[m]), flush=True)
    g = np.array([gains[m] for m in names])
    met = np.array([gains[m] for m in names if m.startswith("metrica")])
    dfl = np.array([gains[m] for m in names if not m.startswith("metrica")])
    ok1, ok2, ok3 = g.mean() >= 0.010, (g > 0).sum() >= 8, met.mean() > 0 and dfl.mean() > 0
    print("fine labels, leave one match out: mean gain {:+.4f} ({}), wins {}/10 ({}), Metrica {:+.4f} "
          "DFL {:+.4f} ({})".format(g.mean(), ok1, (g > 0).sum(), ok2, met.mean(), dfl.mean(), ok3))
    json.dump({"gains": gains, "taus": picks}, open(SELECT, "w"), indent=1)
    print("FINE_RULE", "PASS" if ok1 and ok2 and ok3 else "FAIL")


def cmd_final(_):
    names = [name_of(m) for m in fine_matches()]
    data = load_all(fine_matches(), [c for c, _, _ in CONDS])
    s_all = heldout_scores(data, names, lambda x: train(data, [k for k in data if k[0] != x]))
    tau, J = pick_tau(s_all, names)
    model = train(data, list(data))
    model["tau"] = tau
    print("tau on all ten (leave-one-out predictions):", {t: round(v, 4) for t, v in J.items()}, "->", tau)
    sc = load_all(sc_matches(), [CONDS[0][0]])
    sc_names = [name_of(m) for m in sc_matches()]
    d_sc = np.array([s_of(sc, m, model, tau) - s_of(sc, m, model, 2.0) for m in sc_names])
    sel = json.load(open(SELECT))
    g = sel["gains"]
    met = np.mean([g[m] for m in names if m.startswith("metrica")])
    dfl = np.mean([g[m] for m in names if not m.startswith("metrica")])
    three = np.mean([met, dfl, d_sc.mean()])
    rng = np.random.default_rng(0)
    boots = [rng.choice(d_sc, len(d_sc)).mean() for _ in range(10000)]
    print("SkillCorner ({}): mean gain {:+.4f} (95% {:+.4f} to {:+.4f}), wins {}/{}".format(
        len(d_sc), d_sc.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5),
        (d_sc > 0).sum(), len(d_sc)))
    print("three groups weighted equally: {:+.4f} (Metrica {:+.4f}, DFL {:+.4f}, SkillCorner {:+.4f})".format(
        three, met, dfl, d_sc.mean()))
    print("THREE_GROUP", "PASS" if three >= 0.005 else "FAIL")
    if tau <= 1.0:
        json.dump(model, open(WEIGHTS, "w"), indent=1)
        print("weights written to", WEIGHTS)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": cmd_build, "diag": cmd_diag, "select": cmd_select, "final": cmd_final}[cmd](sys.argv[2:])
