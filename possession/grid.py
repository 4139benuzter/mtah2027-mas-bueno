"""Grid over existing v30 settings, scored on labelled OTHER matches only (validation set).

Never run on the competition match: its labels are private, and its public 5-minute window is not
used to choose anything. The selection rule is fixed in calib/grid_rule.md BEFORE any result is
read. One run = one match in one ball condition; features are computed once, then every setting
in GRID is decoded exactly as possession/validate.py (and so possession/run.py) decodes.

    python -m possession.grid --sample <folder or .npz> [--game 1] [--ball-visible 0.59 --seed 0]
                              --jobs 32 --out calib/grid/<name>.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import possession.model as M  # noqa: E402
from possession import episodes as E, features, sample_game  # noqa: E402
from possession.model import (Params, emissions, viterbi, merge_short_dead,  # noqa: E402
                              snap_dead_onsets, gated_team_costs)
from possession.validate import FPS, drop_ball  # noqa: E402

# First value of every list = the v30 default. Settings only; no new logic.
GRID = {
    "switch_team": [15.0, 10.0, 6.0],
    "gate_window_s": [0.2, 0.5, 1.0],
    "hold_s": [2.0, 1.25, 0.75],
    "carry_s": [1.0, 0.6],
    "radius": [1.5, 1.25, 2.0],
    "min_contact": [3, 2],          # also min_pass_contact (v30 keeps them equal)
    "switch_dead": [25.0, 18.0],
}
DEFAULT = {k: v[0] for k, v in GRID.items()}
MODEL_KEYS = ("switch_team", "gate_window_s", "switch_dead")

_S = {}


def configs():
    return [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]


def apply(cfg):
    """Set the episode settings (module class defaults) and return a Params subclass."""
    for key in ("hold_s", "carry_s", "radius"):
        setattr(E.EpisodeParams, key, cfg[key])
    E.EpisodeParams.min_contact = E.EpisodeParams.min_pass_contact = int(cfg["min_contact"])
    P = type("P", (Params,), {key: cfg[key] for key in MODEL_KEYS})
    return P


def decode(cfg):
    feat, labels, info = _S["feat"], _S["labels"], _S["info"]
    n = len(labels)
    graded = labels != "X"
    p = apply(cfg)
    M.KICKOFF_FRAMES = info["kickoffs"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, p)
    team_costs = gated_team_costs(diag["eps"], n, FPS, p) if p.gate_switches else None
    path = merge_short_dead(viterbi(log_e, p.switch_team, p.switch_dead, team_costs), FPS, p)
    path = snap_dead_onsets(path, feat, FPS, p)
    pred = np.full(n, pm.X, dtype=np.int8)
    pred[graded] = path[graded]
    r = pm.evaluate(_S["gt"], pred)
    return {"cfg": cfg, "spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
            "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True)
    ap.add_argument("--game", type=int, default=3)
    ap.add_argument("--ball-visible", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-default", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    trk, labels, info = sample_game.load(args.sample, args.game)
    if args.ball_visible:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", args.ball_visible, args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS,
                                kickoffs=info["kickoffs"])
    _S.update(feat=feat, labels=labels, info=info, gt=pm.label_codes(labels))
    name = os.path.basename(str(args.sample))[:-4] if str(args.sample).endswith(".npz") \
        else "metrica_game{}".format(args.game)
    cond = "rec" if not args.ball_visible else "v{:.0f}s{}".format(100 * args.ball_visible,
                                                                   args.seed)
    todo = [DEFAULT] if args.only_default else configs()
    print("{} {}: features {:.0f} s, {} settings".format(name, cond, time.time() - t0, len(todo)),
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pool = Pool(args.jobs) if args.jobs > 1 else None   # Linux fork: workers share _S
    rows = pool.imap_unordered(decode, todo, chunksize=1) if pool else map(decode, todo)
    with open(args.out, "w") as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps(dict(match=name, cond=cond, **row)) + "\n")
            if i % 100 == 0:
                print("  {} done, {:.0f} s".format(i + 1, time.time() - t0), flush=True)
    if pool:
        pool.close()
        pool.join()
    print("finished in {:.0f} s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
