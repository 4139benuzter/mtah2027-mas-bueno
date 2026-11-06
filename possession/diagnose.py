"""Label-free health checks for a submission, plus the public-window segment table.

    python -m possession.diagnose evaluation/results/v8_hmm.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from possession import tagging  # noqa: E402
from possession.run import graded_mask  # noqa: E402

PURE_ALIVE = ("build up", "progression", "attacking transition", "defensive transition", "long ball")


def main(path):
    g = GAMES[GAME]
    n, fps = g["n_video_frames"], g["fps"]
    gr = graded_mask(n)
    pred = pd.read_csv(path)["label"].values
    inst = tagging.load_instances()
    alive = np.zeros(n, bool)
    for i in inst:
        if i["code"] in PURE_ALIVE:
            alive[max(0, i["start"]):min(n, i["end"])] = True
    dead = np.zeros(n, bool)
    for s, e, t, k in tagging.set_pieces(inst):
        dead[max(0, s):min(n, s + 3 * fps)] = True
    alive &= gr & ~dead
    dead &= gr
    w = pred[gr]
    breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(w)]))
    runs = [(w[bounds[i]], (bounds[i + 1] - bounds[i]) / fps) for i in range(len(bounds) - 1)]
    tl = np.array([l for k, l in runs if k in "HA"])
    dl = np.array([l for k, l in runs if k == "D"])
    print("{}: D share {:.1%}  possessions {}  mean {:.1f} s (p25 {:.1f} p50 {:.1f} p75 {:.1f})  "
          "D runs {} (p50 {:.1f} s)".format(os.path.basename(path), (w == "D").mean(), len(tl),
                                            tl.mean(), *np.percentile(tl, [25, 50, 75]), len(dl),
                                            np.median(dl)))
    print("  P(D | pure alive tags) {:.1%}   P(D | set-piece pre-restart) {:.1%}   "
          "possessions < 2 s: {:.0%}".format((pred[alive] == "D").mean(), (pred[dead] == "D").mean(),
                                            (tl < 2).mean()))
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
    lo, hi = 11185, 18685
    wg = gt[lo:hi]
    breaks = np.flatnonzero(wg[1:] != wg[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(wg)]))
    print("  public window, GT segment -> predicted runs:")
    for i in range(len(bounds) - 1):
        a, b = lo + bounds[i], lo + bounds[i + 1]
        pr = pred[a:b]
        out, cur, cnt = [], pr[0], 0
        for ch in pr:
            if ch == cur:
                cnt += 1
            else:
                out.append("{}{:.1f}".format(cur, cnt / fps))
                cur, cnt = ch, 1
        out.append("{}{:.1f}".format(cur, cnt / fps))
        print("   {:>2} {} {:>5.1f}s | {}".format(i, wg[bounds[i]], (b - a) / fps, " ".join(out)))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        main(path)
