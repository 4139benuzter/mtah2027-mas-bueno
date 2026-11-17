"""v34 candidate bench: v31 plus three ideas read in other teams' public repositories, scored on
labelled OTHER matches (Metrica 1-3 in three ball conditions, the 19 SkillCorner test matches as
recorded). Never reads the competition match or its public window. Rule: calib/v34_rule.md, fixed
before any result of this script was read; possession/select_v34.py applies it.

    python -m possession.bench_v34 --metrica ../external/metrica-sample --sc ../external/skillcorner-npz
                                   --jobs 5 --out calib/bench_v34/bench.jsonl
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
from possession.bench import scores, CONDITIONS  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              merge_same_side_dead, snap_dead_onsets, snap_dead_to_exit,
                              clean_short_spells, place_handovers_in_gaps, backdate_turnovers,
                              gated_team_costs)
from possession.validate import FPS, drop_ball  # noqa: E402

# First value of every list = v31.
EMISSION = {"oob_reentry_s": [0.0, 1.0]}
CLEANUP = {"exit_snap": [False, True], "turnover_backdate_s": [0.0, 0.5, 1.0]}
V31 = {k: v[0] for k, v in itertools.chain(EMISSION.items(), CLEANUP.items())}


def run_task(task):
    name, sample, game, vis, seed = task
    t0 = time.time()
    trk, labels, info = sample_game.load(sample, game)
    if vis:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", vis, seed)
    n = len(labels)
    graded = labels != "X"
    gt = pm.label_codes(labels)
    M.KICKOFF_FRAMES = info["kickoffs"]
    cond = "rec" if not vis else "v{:.0f}s{}".format(100 * vis, seed)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS, kickoffs=info["kickoffs"])
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
                path = snap_dead_to_exit(path, feat, FPS, q)
                path = clean_short_spells(path, FPS, q)
                path = place_handovers_in_gaps(path, feat, FPS, q)
                path = backdate_turnovers(path, feat, FPS, q)
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
            out.append((os.path.basename(path)[:-4], path, 3, None, 0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrica", default=None)
    ap.add_argument("--sc", default=None)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    todo = tasks(args.metrica, args.sc)
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
