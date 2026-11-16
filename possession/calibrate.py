"""Choose decoder settings on labelled OTHER matches (Metrica's sample games), never on BAR-ATH.

The competition rules forbid tuning on the public 5-minute ground truth (RULES.md section 2) and
allow labelled data about other games. This script runs the unchanged decoder of possession/ on
one of Metrica's public sample matches (see possession/sample_game.py) for every setting in a
grid, scores each run with the official evaluation/possession_metrics.py against labels derived
from the human events of that match, and writes one JSON line per setting. The selection rule
(select()) prefers settings whose grid NEIGHBOURHOOD scores well on every match, and among the
near-best the one closest to v27; crossval() estimates its gain on a match it has not seen.

The sample matches have no Smart Tagging, so the tagging terms are off here; only the
tracking-driven settings in GRID are calibrated, and everything else stays at its default.

    python -m possession.calibrate --sample <folder> --game 3 --out calib/g3.jsonl
    python -m possession.calibrate --crossval  calib/g1.jsonl calib/g2.jsonl calib/g3.jsonl
    python -m possession.calibrate --summarize calib/g1.jsonl calib/g2.jsonl calib/g3.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import features, sample_game  # noqa: E402
from possession.episodes import EpisodeParams  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              snap_dead_onsets)

# The grid. Values bracket the current defaults (marked *) by football-plausible amounts.
GRID = {
    "switch_team": [2.0, 3.0, 4.0, 6.0, 9.0, 15.0],  # * 15 nats
    "switch_dead": [15.0, 20.0, 25.0],             # * 25 nats
    "radius": [0.75, 1.0, 1.25, 1.5],              # * 1.5 m contact radius (episodes + per frame)
    "hold_s": [0.5, 0.75, 1.0, 2.0],               # * 2.0 s of contact = control
    "min_dead_s": [2.0, 3.0, 4.0],                 # * 2.0 s shortest stoppage kept
    "medium": [0.5, 0.65, 0.8],                    # * 0.5 home share of an unresolved contact
}
DEFAULT = {"switch_team": 15.0, "switch_dead": 25.0, "radius": 1.5, "hold_s": 2.0,
           "min_dead_s": 2.0, "medium": 0.5}   # the v27 values, the reference point
NEAR_BEST = 0.005   # combine(): settings this close to the best count as equally good
V28 = {"switch_team": 4.0, "switch_dead": 25.0, "radius": 0.75, "hold_s": 0.75,
       "min_dead_s": 2.0, "medium": 0.8}  # chosen on game 3 alone (superseded)
V29 = {"switch_team": 6.0, "switch_dead": 25.0, "radius": 1.5, "hold_s": 0.75,
       "min_dead_s": 2.0, "medium": 0.5}  # chosen on games 1, 2 and 3; the values in the code

_STATE = {}
_EP_DEFAULTS = {k: v for k, v in vars(EpisodeParams).items() if not k.startswith("_")}


def drop_ball(ball, graded, target, seed=0, fps=25):
    """Copy of ball with extra random gaps (0.4-5 s) until the ball is visible on `target` of the
    graded frames. Sample Game 3 shows the ball on ~68% of graded frames, BAR-ATH on ~59%; a
    setting that only wins when the ball is easy to see should not be chosen. Seeded, so the
    calibration is reproducible."""
    ball = ball.copy()
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(graded)
    while (~np.isnan(ball[graded, 0])).mean() > target:
        start = int(rng.choice(idx))
        length = int(rng.integers(int(0.4 * fps), 5 * fps + 1))
        ball[start:start + length] = np.nan
    return ball


def _init(folder, ball_visible=None, seed=0, game=3):
    """Load the calibration match once per worker process."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        trk, labels, info = sample_game.load(folder, game)
        if ball_visible:
            trk["ball"] = drop_ball(trk["ball"], labels != "X", ball_visible, seed)
        feat = features.compute(trk, sample_game.PITCH, sample_game.FPS,
                                kickoffs=info["kickoffs"])
    gt = pm.label_codes(labels)
    halves = []
    for _, first, last in info["halves"]:
        g = np.full(len(gt), pm.X, dtype=gt.dtype)
        g[first:last + 1] = gt[first:last + 1]
        halves.append(g)
    _STATE.update(feat=feat, gt=gt, gt_halves=halves, graded=labels != "X", n=len(labels),
                  kickoffs=info["kickoffs"])


def decode(cfg):
    """Run the unchanged decoder with one setting; return the label codes (X off the graded run)."""
    for k, v in _EP_DEFAULTS.items():
        setattr(EpisodeParams, k, v)
    EpisodeParams.radius = cfg["radius"]
    EpisodeParams.hold_s = cfg["hold_s"]
    EpisodeParams.medium = cfg.get("medium", _EP_DEFAULTS["medium"])
    P = type("P", (Params,), {"switch_team": cfg["switch_team"],
                              "switch_dead": cfg["switch_dead"],
                              "sigma_control": cfg["radius"],
                              "min_dead_s": cfg["min_dead_s"]})
    s = _STATE
    fps = sample_game.FPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_e, _ = emissions(s["feat"], np.zeros(s["n"]), [], [], [], s["graded"], s["n"], fps,
                             P, kickoffs=s["kickoffs"])
    path = merge_short_dead(viterbi(log_e, P.switch_team, P.switch_dead), fps, P)
    path = snap_dead_onsets(path, s["feat"], fps, P)
    pred = np.full(s["n"], pm.X, dtype=np.int8)
    pred[s["graded"]] = path[s["graded"]]
    return pred


def score(cfg):
    t0 = time.time()
    pred = decode(cfg)
    s = _STATE
    full = pm.evaluate(s["gt"], pred)
    out = {"cfg": cfg, "spell_f1": full["spell_f1_mean"], "macro_f1": full["macro_f1"],
           "spell_f1_per_iou": {str(k): v["f1"] for k, v in full["spell_f1"].items()},
           "halves_spell_f1": [pm.evaluate(g, pred)["spell_f1_mean"] for g in s["gt_halves"]],
           "seconds": round(time.time() - t0, 1)}
    for key in ("n_gt_spells", "n_pred_spells", "gt_possessions", "pred_possessions"):
        if key in full:
            out[key] = full[key]
    return out


def configs():
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*(GRID[k] for k in keys))]


def neighbourhood_scores(rows):
    """Mean spell F1 over each setting and its +/-1-step grid neighbours (one axis at a time)."""
    keys = list(GRID)
    index = {tuple(r["cfg"][k] for k in keys): r["spell_f1"] for r in rows}
    out = []
    for r in rows:
        point = [r["cfg"][k] for k in keys]
        vals = [r["spell_f1"]]
        for i, k in enumerate(keys):
            pos = GRID[k].index(point[i])
            for step in (-1, 1):
                if 0 <= pos + step < len(GRID[k]):
                    q = list(point)
                    q[i] = GRID[k][pos + step]
                    if tuple(q) in index:
                        vals.append(index[tuple(q)])
        out.append((float(np.mean(vals)), float(np.min(vals)), r))
    return sorted(out, key=lambda t: -t[0])


def _steps(c):
    """Grid steps between a setting (tuple in GRID order) and the v27 values."""
    return sum(abs(GRID[k].index(c[i]) - GRID[k].index(DEFAULT[k])) for i, k in enumerate(GRID))


def select(row_sets):
    """The selection rule on several result sets (one per match or variant): rank settings by the
    mean over the sets of their neighbourhood score, then, among those within NEAR_BEST of the
    best, take the one fewest grid steps from v27. Returns (chosen, per_set lookup)."""
    keys = list(GRID)
    per_set = [{tuple(r["cfg"][k] for k in keys): (m, w, r) for m, w, r in neighbourhood_scores(rows)}
               for rows in row_sets]
    common = set.intersection(*(set(d) for d in per_set))
    score = {c: float(np.mean([d[c][0] for d in per_set])) for c in common}
    best = max(score.values())
    near = [c for c in common if score[c] >= best - NEAR_BEST]
    chosen = min(near, key=lambda c: (_steps(c), -score[c]))
    return chosen, per_set


def crossval(paths):
    """Leave one match out: choose on the other result files, score the choice on the left-out
    one, and compare with v27 there. This estimates what the selection rule gains on a match it
    has never seen -- which is the situation of the competition match."""
    keys = list(GRID)
    rows = [[json.loads(line) for line in open(p)] for p in paths]
    default = tuple(DEFAULT[k] for k in keys)
    gains = []
    for i, path in enumerate(paths):
        chosen, _ = select([r for j, r in enumerate(rows) if j != i])
        own = {tuple(r["cfg"][k] for k in keys): r["spell_f1"] for r in rows[i]}
        gains.append(own[chosen] - own[default])
        print("held out {}: v27 {:.4f}  chosen on the others {:.4f}  ({:+.4f})  {}".format(
            os.path.basename(path), own[default], own[chosen], gains[-1], dict(zip(keys, chosen))))
    print("mean held-out gain {:+.4f}".format(float(np.mean(gains))))
    return gains


def combine(paths):
    """Rank settings by the mean of their neighbourhood scores across several result files
    (e.g. the plain calibration match and the same match with the ball removed to BAR-ATH's
    visibility). A setting must hold up in every variant to rank high."""
    keys = list(GRID)
    per_file = []
    for path in paths:
        rows = [json.loads(line) for line in open(path)]
        per_file.append({tuple(r["cfg"][k] for k in keys): (m, w, r)
                         for m, w, r in neighbourhood_scores(rows)})
    common = set.intersection(*(set(d) for d in per_file))
    ranked = sorted(common, key=lambda c: -np.mean([d[c][0] for d in per_file]))
    print("combined ranking over {} files ({} common settings)".format(len(paths), len(common)))
    for c in ranked[:10]:
        print("  mean nbhd {:.4f}  per file {}  own {}  {}".format(
            np.mean([d[c][0] for d in per_file]), [round(d[c][0], 4) for d in per_file],
            [round(d[c][2]["spell_f1"], 4) for d in per_file], dict(zip(keys, c))))
    default = tuple(DEFAULT[k] for k in keys)
    if default in common:
        print("  DEFAULT (v27): mean nbhd {:.4f}  own {}".format(
            np.mean([d[default][0] for d in per_file]),
            [round(d[default][2]["spell_f1"], 4) for d in per_file]))
    # Step 3: among the settings within NEAR_BEST of the best, the fewest grid steps from v27.
    chosen, _ = select([[d[c][2] for c in d] for d in per_file])
    print("  closest to v27 among the near-best ({} grid steps): own {}".format(
        _steps(chosen), [round(d[chosen][2]["spell_f1"], 4) for d in per_file]))
    return dict(zip(keys, chosen))


def summarize(path):
    rows = [json.loads(line) for line in open(path)]
    print("{} settings scored".format(len(rows)))
    default = [r for r in rows if r["cfg"] == DEFAULT]
    if default:
        d = default[0]
        print("DEFAULT (v27) spell {:.4f}  macro {:.4f}  halves {}".format(
            d["spell_f1"], d["macro_f1"], [round(h, 4) for h in d["halves_spell_f1"]]))
    print("\ntop 8 by raw spell F1")
    for r in sorted(rows, key=lambda r: -r["spell_f1"])[:8]:
        print("  {:.4f}  macro {:.4f}  halves {}  {}".format(
            r["spell_f1"], r["macro_f1"], [round(h, 4) for h in r["halves_spell_f1"]], r["cfg"]))
    ranked = neighbourhood_scores(rows)
    print("\ntop 8 by neighbourhood mean (the selection rule)")
    for mean, worst, r in ranked[:8]:
        print("  nbhd {:.4f} (worst {:.4f})  own {:.4f}  halves {}  {}".format(
            mean, worst, r["spell_f1"], [round(h, 4) for h in r["halves_spell_f1"]], r["cfg"]))
    return ranked[0][2]["cfg"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", default=os.path.join(os.path.dirname(ROOT), "external",
                                                     "metrica-sample-game3"))
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(ROOT, "calib", "results.jsonl"))
    ap.add_argument("--game", type=int, default=3, choices=(1, 2, 3),
                    help="Metrica sample match: 3 chooses the settings, 1 and 2 are held out")
    ap.add_argument("--only-default", action="store_true", help="score the v27 setting only")
    ap.add_argument("--cfg", action="append", default=[], metavar="JSON",
                    help="score only these settings (repeatable); 'v27', 'v28', 'v29' are shortcuts")
    ap.add_argument("--ball-visible", type=float, default=None,
                    help="drop extra ball positions until this share of graded frames shows it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", metavar="RESULTS", nargs="+")
    ap.add_argument("--crossval", metavar="RESULTS", nargs="+",
                    help="leave-one-out over result files, one per match")
    args = ap.parse_args()
    if args.crossval:
        crossval(args.crossval)
        return
    if args.summarize:
        if len(args.summarize) == 1:
            print("selected:", summarize(args.summarize[0]))
        else:
            print("selected:", combine(args.summarize))
        return

    named = {"v27": DEFAULT, "v28": V28, "v29": V29}
    if args.cfg:
        todo = [named[c] if c in named else json.loads(c) for c in args.cfg]
    else:
        todo = [DEFAULT] if args.only_default else configs()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    init_args = (args.sample, args.ball_visible, args.seed, args.game)
    if args.jobs <= 1 or len(todo) == 1:
        _init(*init_args)
        results = [score(c) for c in todo]
    else:
        import multiprocessing as mp
        with mp.Pool(args.jobs, initializer=_init, initargs=init_args) as pool:
            results = pool.map(score, todo, chunksize=1)
    with open(args.out, "w") as handle:
        for r in results:
            handle.write(json.dumps(r) + "\n")
    print("{} settings in {:.0f} s -> {}".format(len(results), time.time() - t0, args.out))
    if len(results) > 2:
        print("selected:", summarize(args.out))
    else:
        for r in results:
            print("{}  spell {:.4f}  macro {:.4f}  halves {}  pred {} / true {}".format(
                r["cfg"], r["spell_f1"], r["macro_f1"], [round(h, 4) for h in r["halves_spell_f1"]],
                r.get("n_pred_spells"), r.get("n_gt_spells")))


if __name__ == "__main__":
    main()
