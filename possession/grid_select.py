"""Apply the fixed selection rule of calib/grid_rule.md to the grid results. Chooses nothing else.

    python -m possession.grid_select [--grid window] calib/grid/*.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np

from possession.grid import default

DEFAULT = default("v30")

NEAR_BEST = 0.003
MIN_MEAN_GAIN = 0.010
MIN_WINS = 8
CONDITIONS = 3


def key(cfg):
    return tuple(sorted(cfg.items()))


def load(paths):
    """s[cfg_key][match] = mean spellF1 over the ball conditions; also the cfg dicts."""
    raw = defaultdict(lambda: defaultdict(list))
    cfgs = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            k = key(r["cfg"])
            cfgs[k] = r["cfg"]
            raw[k][r["match"]].append(r["spell_f1"])
    matches = sorted({m for per in raw.values() for m in per})
    s = {}
    for k, per in raw.items():
        if all(len(per[m]) == CONDITIONS for m in matches):
            s[k] = {m: float(np.mean(per[m])) for m in matches}
    return s, cfgs, matches


def changes(cfg):
    return sum(cfg[k] != DEFAULT[k] for k in DEFAULT)


def pick(s, cfgs, subset):
    J = {k: np.mean([v[m] for m in subset]) for k, v in s.items()}
    best = max(J.values())
    near = [k for k in J if J[k] >= best - NEAR_BEST]
    return min(near, key=lambda k: (changes(cfgs[k]), -J[k])), J


def main(paths):
    global DEFAULT
    if paths and paths[0] == "--grid":
        DEFAULT, paths = default(paths[1]), paths[2:]
    s, cfgs, matches = load(paths)
    assert len(matches) == 10, "expected the ten validation matches, got {}".format(matches)
    d = key(DEFAULT)
    assert d in s, "the v30 default is missing from the results"
    print("{} settings complete on {} matches x {} conditions".format(len(s), len(matches), CONDITIONS))
    gains = {}
    for m in matches:
        k, _ = pick(s, cfgs, [x for x in matches if x != m])
        gains[m] = s[k][m] - s[d][m]
        print("  held out {:14s} pick {}  gain {:+.4f}".format(
            m, {a: b for a, b in cfgs[k].items() if b != DEFAULT[a]}, gains[m]))
    g = np.array([gains[m] for m in matches])
    met = np.array([gains[m] for m in matches if m.startswith("metrica")])
    dfl = np.array([gains[m] for m in matches if not m.startswith("metrica")])
    ok1, ok2 = g.mean() >= MIN_MEAN_GAIN, (g > 0).sum() >= MIN_WINS
    ok3 = met.mean() > 0 and dfl.mean() > 0
    print("leave-one-match-out: mean gain {:+.4f} (rule >= {:+.3f}: {}), wins {}/{} (rule >= {}: {}), "
          "Metrica {:+.4f} DFL {:+.4f} (both > 0: {})".format(
              g.mean(), MIN_MEAN_GAIN, ok1, (g > 0).sum(), len(g), MIN_WINS, ok2,
              met.mean(), dfl.mean(), ok3))
    final, J = pick(s, cfgs, matches)
    print("pick on all ten: {}  J {:.4f} (v30 {:.4f}, best {:.4f})".format(
        {a: b for a, b in cfgs[final].items() if b != DEFAULT[a]}, J[final], J[d], max(J.values())))
    for m in matches:
        print("  {:14s} v30 {:.4f}  pick {:.4f}  {:+.4f}".format(m, s[d][m], s[final][m],
                                                                s[final][m] - s[d][m]))
    verdict = ok1 and ok2 and ok3
    print("VERDICT:", "ADOPT (subject to the label-free checks)" if verdict else "KEEP v30")
    print("FINAL_CFG", json.dumps(cfgs[final] if verdict else DEFAULT))


if __name__ == "__main__":
    main(sys.argv[1:])
