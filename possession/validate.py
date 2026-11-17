"""Out-of-sample check of the CURRENT decoder on labelled OTHER matches. Chooses nothing.

The competition match's labels are private and the public 5-minute window must not be used to
choose between versions, so the only legal way to measure the decoder is on other matches with
labels. This script runs possession/ exactly as possession/run.py does -- same Params, same
emissions, gated switch costs, post-passes -- on:

  * Metrica's public sample matches 1-3 (possession/sample_game.py; labels from human events),
  * the DFL/Sportec open matches converted by possession/dfl.py (labels from the corrected
    tracking's own BallPossession and BallStatus, the kind of fields the competition ground truth
    comes from),

and scores the result with the official evaluation/possession_metrics.py. Two inputs do not
exist for these matches and are switched off: Smart Tagging (no tags) and the keeper fix of
possession/keepers.py (their tracking has fixed player identities, so there are no keeper-slot
swaps to fix). --ball-visible removes the ball in random 0.4-5 s gaps until it is tracked on that
share of graded frames (the competition match: 59%), seeded and therefore reproducible.

    python -m possession.validate --sample ../external/metrica-sample-game12 --game 1
    python -m possession.validate --sample ../external/dfl-npz/J03WOH.npz --ball-visible 0.59
"""

import argparse
import json
import os
import sys
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
                              snap_dead_onsets, gated_team_costs, merge_same_side_dead,
                              clean_short_spells, place_handovers_in_gaps,
                              snap_dead_to_exit, backdate_turnovers)

FPS = 25


def drop_ball(ball, graded, target, seed=0, fps=FPS):
    """Copy of ball with extra random gaps (0.4-5 s) until it is visible on `target` of graded."""
    ball = ball.copy()
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(graded)
    while (~np.isnan(ball[graded, 0])).mean() > target:
        start = int(rng.choice(idx))
        ball[start:start + int(rng.integers(int(0.4 * fps), 5 * fps + 1))] = np.nan
    return ball


def decode(trk, labels, info, p=Params):
    """Label codes from the current decoder, X off the graded frames."""
    n = len(labels)
    graded = labels != "X"
    pitch = info.get("pitch", sample_game.PITCH)
    M.KICKOFF_FRAMES = info["kickoffs"]          # emissions() reads the module constant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = features.interpolate_gaps(trk["ball"], FPS, p.ball_gap_fill_s) \
            if p.ball_gap_fill_s > 0 else None
        feat = features.compute(trk, pitch, FPS, ball_fill=fill, kickoffs=info["kickoffs"])
        log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, p)
    team_costs = gated_team_costs(diag["eps"], n, FPS, p) if p.gate_switches else None
    path = merge_short_dead(viterbi(log_e, p.switch_team, p.switch_dead, team_costs), FPS, p)
    path = merge_same_side_dead(path, FPS, p)
    path = snap_dead_onsets(path, feat, FPS, p)
    path = snap_dead_to_exit(path, feat, FPS, p)
    path = clean_short_spells(path, FPS, p)
    path = place_handovers_in_gaps(path, feat, FPS, p)
    path = backdate_turnovers(path, feat, FPS, p)
    pred = np.full(n, pm.X, dtype=np.int8)
    pred[graded] = path[graded]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", required=True, help="Metrica sample folder, or a DFL .npz")
    ap.add_argument("--game", type=int, default=3, choices=(1, 2, 3))
    ap.add_argument("--ball-visible", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="append one JSON line here")
    args = ap.parse_args()

    trk, labels, info = sample_game.load(args.sample, args.game)
    if args.ball_visible:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", args.ball_visible, args.seed)
    gt = pm.label_codes(labels)
    r = pm.evaluate(gt, decode(trk, labels, info))
    name = os.path.basename(str(args.sample)) if str(args.sample).endswith(".npz") \
        else "metrica_game{}".format(args.game)
    row = {"match": name, "ball_visible": args.ball_visible, "seed": args.seed,
           "spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"],
           "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"]}
    print("{match:28s} ball {vis:>8s}  spellF1 {spell_f1:.4f}  macroF1 {macro_f1:.4f}  "
          "possessions {n_pred_spells}/{n_gt_spells}".format(
              vis=str(args.ball_visible or "recorded"), **row), flush=True)
    if args.out:
        with open(args.out, "a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
