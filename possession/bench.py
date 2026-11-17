"""v32 candidate bench: v31 plus the candidate settings below, scored on labelled OTHER matches.

Never reads the competition match or its public window. The selection rule is fixed in
calib/v32_rule.md before any result of this script is read, and possession/select_v32.py
implements it. One task = one match in one ball condition: features once, one Viterbi per
emission setting, then every clean-up setting on that Viterbi path, decoded exactly as
possession/run.py decodes (Smart Tagging and the keeper fix are off, as in validate.py).
Works on Windows: one worker process per task, no fork.

    python -m possession.bench --metrica ../external/metrica-sample --sc ../external/skillcorner-npz
                               --jobs 6 --out calib/bench_v32/bench.jsonl
"""

import argparse
import glob
import itertools
import json
import multiprocessing
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
import possession.model as M  # noqa: E402
from possession import features, sample_game  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              merge_same_side_dead, snap_dead_onsets, clean_short_spells,
                              gated_team_costs)
from possession.validate import FPS, drop_ball  # noqa: E402

# First value of every list = v31. Emission settings need their own Viterbi; clean-up settings
# act on the decoded path.
EMISSION = {
    "run_dead_hi_lost": [None, 0.50, 0.40],   # P(D) when nobody runs and the ball is lost
}
CLEANUP = {
    "min_dead_s": [2.0, 3.0],                 # any stoppage shorter than this is folded
    "min_dead_same_s": [0.0, 4.0, 6.0],       # ... and one with the same side on both ends
    "min_blip_s": [0.0, 1.0, 2.0],            # H A H: the short A goes to H
    "min_tail_s": [0.0, 1.0, 2.0],            # H A D: the short A goes to H
}
V31 = {k: v[0] for k, v in itertools.chain(EMISSION.items(), CLEANUP.items())}
CONDITIONS = {"metrica": [(None, 0), (0.59, 0), (0.59, 1)], "sc": [(None, 0)]}


def configs():
    return [dict(zip(list(EMISSION) + list(CLEANUP), values))
            for values in itertools.product(*EMISSION.values(), *CLEANUP.values())]


def scores(gt, pred):
    """spellF1 and macroF1 exactly as possession_metrics.evaluate computes them."""
    runs = pm.valid_runs(gt)
    spells = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return spells["mean_f1"], pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))), \
        spells["n_pred"], spells["n_gt"]


def run_task(task):
    name, sample, game, vis, seed = task
    t0 = time.time()
    trk, labels, info = sample_game.load(sample, game)
    if vis:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", vis, seed)
    n = len(labels)
    graded = labels != "X"
    gt = pm.label_codes(labels)
    M.KICKOFF_FRAMES = info["kickoffs"]          # emissions() reads the module constant
    cond = "rec" if not vis else "v{:.0f}s{}".format(100 * vis, seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS,
                                kickoffs=info["kickoffs"])
        for e_values in itertools.product(*EMISSION.values()):
            ecfg = dict(zip(EMISSION, e_values))
            p = type("P", (Params,), ecfg)
            log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, p)
            team_costs = gated_team_costs(diag["eps"], n, FPS, p) if p.gate_switches else None
            raw = viterbi(log_e, p.switch_team, p.switch_dead, team_costs)
            for c_values in itertools.product(*CLEANUP.values()):
                cfg = dict(ecfg, **dict(zip(CLEANUP, c_values)))
                q = type("Q", (Params,), cfg)
                path = merge_short_dead(raw, FPS, q)
                path = merge_same_side_dead(path, FPS, q)
                path = snap_dead_onsets(path, feat, FPS, q)
                path = clean_short_spells(path, FPS, q)
                pred = np.full(n, pm.X, dtype=np.int8)
                pred[graded] = path[graded]
                spell, macro, n_pred, n_gt = scores(gt, pred)
                rows.append({"match": name, "cond": cond, "cfg": cfg, "spell_f1": spell,
                             "macro_f1": macro, "n_pred_spells": n_pred, "n_gt_spells": n_gt})
    return name, cond, time.time() - t0, rows


def tasks(metrica, sc):
    out = []
    if metrica:
        for g in (1, 2, 3):
            for vis, seed in CONDITIONS["metrica"]:
                out.append(("metrica_game{}".format(g), metrica, g, vis, seed))
    if sc:
        for path in sorted(glob.glob(os.path.join(sc, "SC*.npz"))):
            for vis, seed in CONDITIONS["sc"]:
                out.append((os.path.basename(path)[:-4], path, 3, vis, seed))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrica", default=None, help="folder with Metrica sample games 1-3")
    ap.add_argument("--sc", default=None, help="folder with SkillCorner SC<id>.npz files")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    todo = tasks(args.metrica, args.sc)
    print("{} tasks x {} settings".format(len(todo), len(configs())), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with open(args.out, "w") as handle, multiprocessing.Pool(args.jobs) as pool:
        for name, cond, secs, rows in pool.imap_unordered(run_task, todo):
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            ref = [r for r in rows if r["cfg"] == V31][0]
            print("{:16s} {:6s} {:4.0f} s  v31 spellF1 {:.4f}  ({:.0f} s total)".format(
                name, cond, secs, ref["spell_f1"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
