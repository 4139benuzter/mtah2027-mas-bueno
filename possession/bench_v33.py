"""v33 candidate bench: v31 plus straight-line filling of short ball gaps and hand-overs placed
inside ball gaps, scored on labelled OTHER matches with the ball AS RECORDED only.

Never reads the competition match or its public window. The rule is calib/v33_rule.md, fixed
before any result of this script was read; possession/select_v33.py applies it. The artificial
59% conditions of possession/validate.py are not used here: their random 0.4-5 s gaps are much
shorter than the competition match's real ones (median 11 s), so they would favour gap filling.

    python -m possession.bench_v33 --metrica ../external/metrica-sample --sc ../external/skillcorner-npz
                                   --jobs 5 --out calib/bench_v33/bench.jsonl
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
from possession.bench import scores  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              merge_same_side_dead, snap_dead_onsets, clean_short_spells,
                              place_handovers_in_gaps, gated_team_costs)
from possession.validate import FPS  # noqa: E402

# First value of every list = v31.
FEATURE = {"ball_gap_fill_s": [0.0, 0.5, 1.0, 2.0]}     # needs its own features and Viterbi
CLEANUP = {"gap_handover_s": [0.0, 3.0, 8.0]}           # acts on the decoded path
V31 = {k: v[0] for k, v in itertools.chain(FEATURE.items(), CLEANUP.items())}


def run_task(task):
    name, sample, game = task
    t0 = time.time()
    trk, labels, info = sample_game.load(sample, game)
    n = len(labels)
    graded = labels != "X"
    gt = pm.label_codes(labels)
    M.KICKOFF_FRAMES = info["kickoffs"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f_values in itertools.product(*FEATURE.values()):
            fcfg = dict(zip(FEATURE, f_values))
            p = type("P", (Params,), fcfg)
            fill = features.interpolate_gaps(trk["ball"], FPS, p.ball_gap_fill_s) \
                if p.ball_gap_fill_s > 0 else None
            feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS, ball_fill=fill,
                                    kickoffs=info["kickoffs"])
            log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, p)
            team_costs = gated_team_costs(diag["eps"], n, FPS, p) if p.gate_switches else None
            raw = viterbi(log_e, p.switch_team, p.switch_dead, team_costs)
            for c_values in itertools.product(*CLEANUP.values()):
                cfg = dict(fcfg, **dict(zip(CLEANUP, c_values)))
                q = type("Q", (Params,), cfg)
                path = merge_short_dead(raw, FPS, q)
                path = merge_same_side_dead(path, FPS, q)
                path = snap_dead_onsets(path, feat, FPS, q)
                path = clean_short_spells(path, FPS, q)
                path = place_handovers_in_gaps(path, feat, FPS, q)
                pred = np.full(n, pm.X, dtype=np.int8)
                pred[graded] = path[graded]
                spell, macro, n_pred, n_gt = scores(gt, pred)
                rows.append({"match": name, "cond": "rec", "cfg": cfg, "spell_f1": spell,
                             "macro_f1": macro, "n_pred_spells": n_pred, "n_gt_spells": n_gt})
    return name, time.time() - t0, rows


def tasks(metrica, sc):
    out = [("metrica_game{}".format(g), metrica, g) for g in (1, 2, 3)] if metrica else []
    if sc:
        out += [(os.path.basename(p)[:-4], p, 3) for p in sorted(glob.glob(os.path.join(sc, "SC*.npz")))]
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
        for name, secs, rows in pool.imap_unordered(run_task, todo):
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            ref = [r for r in rows if r["cfg"] == V31][0]
            print("{:16s} {:4.0f} s  v31 spellF1 {:.4f}  ({:.0f} s total)".format(
                name, secs, ref["spell_f1"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
