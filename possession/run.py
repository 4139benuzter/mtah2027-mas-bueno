"""End to end: tracking + Smart Tagging -> evaluation/results/<name>.csv

    python -m possession.run --name v2_hmm
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from possession.atd import load_tracking  # noqa: E402
from possession import features, tagging  # noqa: E402
from possession.model import Params, emissions, switch_costs, viterbi, merge_short_dead  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
KICKOFF_2ND = 113742     # published in DATA.md / FAQ.md


def graded_mask(n):
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    return graded


def build(params=Params, verbose=True):
    g = GAMES[GAME]
    n, fps, pitch = g["n_video_frames"], g["fps"], g["pitch"]
    trk = load_tracking()
    feat = features.compute(trk, pitch, fps)
    graded = graded_mask(n)
    inst = tagging.load_instances()
    prior = tagging.team_prior(inst, n)
    log_e, diag = emissions(feat, prior, tagging.set_pieces(inst), tagging.goals(inst),
                            graded, n, fps, params)
    cost_team, cost_dead = switch_costs(log_e, feat, graded, fps, params)
    if verbose:
        print("shape logit k {:.3f}  pseudo-label agreement {:.1%} on {} frames".format(
            diag["shape_k"], diag["shape_acc"], diag["shape_n"]))
        print("episodes: {}".format(diag["episodes"]))
        print("switch costs: team {:.1f} nats  dead {:.1f} nats   mean P(D) graded {:.2f}  hold share {:.1%}".format(cost_team, cost_dead, diag["pd_mean_graded"], diag["hold_share"]))
    path = merge_short_dead(viterbi(log_e, cost_team, cost_dead), fps, params)

    labels = np.full(n, "X", dtype="<U1")
    letters = np.array(["H", "A", "D"])
    labels[graded] = letters[path[graded]]
    # Second half is graded from 110808 but kicks off at 113742: dead ball until then.
    _, first2, _ = half_video_blocks(GAME)[1]
    labels[first2:KICKOFF_2ND] = "D"
    return labels, feat


def summarize(labels):
    graded = labels != "X"
    g = labels[graded]
    counts = {k: int((g == k).sum()) for k in "HAD"}
    alive = counts["H"] + counts["A"]
    runs = 0
    prev = None
    for ch in g:
        if ch in "HA" and ch != prev:
            runs += 1
        prev = ch
    mean_dur = alive / runs / 25.0 if runs else 0.0
    print("graded {}  H {:.1%}  A {:.1%}  D {:.1%}  possessions {}  mean {:.1f} s  "
          "home share {:.1%}".format(len(g), counts["H"] / len(g), counts["A"] / len(g),
                                     counts["D"] / len(g), runs, mean_dur,
                                     counts["H"] / alive if alive else 0))


def write(labels, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".csv")
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2_hmm")
    args = ap.parse_args()
    labels, _ = build()
    summarize(labels)
    print("wrote", os.path.relpath(write(labels, args.name), ROOT))
