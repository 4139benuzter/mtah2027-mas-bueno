"""Label-free health checks for a submission (Smart Tagging consistency, run statistics).

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
    exp = np.full(n, "", dtype="<U1")
    for i in inst:
        if i["team"] is None:
            continue
        if i["code"] in tagging.OFFENSIVE:
            t = i["team"]
        elif i["code"] in tagging.DEFENSIVE:
            t = "A" if i["team"] == "H" else "H"
        else:
            continue
        exp[max(0, i["start"]):min(n, i["end"])] = t
    m = gr & (exp != "") & np.isin(pred, ["H", "A"])
    print("  team agreement with Smart Tagging phase sides: {:.1%}".format((pred[m] == exp[m]).mean()))
    # No ground truth of any kind is read here: the checks above use the submission and Metrica's
    # automatic Smart Tagging only. (An earlier version also printed the public-window segment
    # table; it was removed, and render_clip.py no longer draws the public labels by default.)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        main(path)
