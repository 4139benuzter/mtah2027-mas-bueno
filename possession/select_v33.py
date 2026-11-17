"""The v33 decision of calib/v33_rule.md, applied to the output of possession/bench_v33.py.

Same procedure as possession/select_v32.py, with the ball as recorded only: groups Metrica (3)
and SkillCorner (19, SC2017461 excluded), equal weight; pick within 0.003 of the best mean with
the fewest changes from v31; leave one match out; adopt only if the held-out gain is >= +0.010,
positive within each group, and positive on at least 15 of the 22 matches.

    python -m possession.select_v33 calib/bench_v33/bench.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np

from possession.bench_v33 import V31
from possession.sc_check import EXCLUDED

NEAR_BEST, MIN_GAIN, MIN_POSITIVE = 0.003, 0.010, 15
GROUPS = {"Metrica": 3, "SkillCorner": 19}


def group(match):
    return "Metrica" if match.startswith("metrica") else "SkillCorner"


def key(cfg):
    return tuple(sorted(cfg.items()))


def balanced(values, matches):
    by = defaultdict(list)
    for m in matches:
        by[group(m)].append(values[m])
    return float(np.mean([np.mean(v) for v in by.values()])), \
        {g: float(np.mean(v)) for g, v in by.items()}


def changes(cfg):
    return sum(cfg[k] != V31[k] for k in V31)


def pick(s, cfgs, subset):
    J = {k: balanced(v, subset)[0] for k, v in s.items()}
    best = max(J.values())
    return min([k for k in J if J[k] >= best - NEAR_BEST], key=lambda k: (changes(cfgs[k]), -J[k]))


def diff(cfg):
    return {a: b for a, b in cfg.items() if b != V31[a]} or "v31"


def main(paths):
    s, cfgs = defaultdict(dict), {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["match"] in EXCLUDED or r["cond"] != "rec":
                continue
            cfgs[key(r["cfg"])] = r["cfg"]
            s[key(r["cfg"])][r["match"]] = r["spell_f1"]
    matches = sorted({m for per in s.values() for m in per})
    counts = defaultdict(int)
    for m in matches:
        counts[group(m)] += 1
    assert dict(counts) == GROUPS, dict(counts)
    s = {k: v for k, v in s.items() if len(v) == len(matches)}
    ref = key(V31)
    gains, picks = {}, defaultdict(int)
    for m in matches:
        k = pick(s, cfgs, [x for x in matches if x != m])
        gains[m] = s[k][m] - s[ref][m]
        picks[k] += 1
    for k, n in sorted(picks.items(), key=lambda kv: -kv[1]):
        print("  held-out pick in {} folds: {}".format(n, diff(cfgs[k])))
    total, per_group = balanced(gains, matches)
    positive = sum(g > 0 for g in gains.values())
    print("held-out gain vs v31 (groups weighted equally): {:+.4f}; per group {}; positive on {} of {}".format(
        total, {g: round(v, 4) for g, v in per_group.items()}, positive, len(matches)))
    for k in sorted(s, key=lambda k: -balanced(s[k], matches)[0]):
        j, g = balanced(s[k], matches)
        print("  {:+.4f}  {}  {}".format(j - balanced(s[ref], matches)[0], diff(cfgs[k]),
                                         {a: round(b - balanced(s[ref], matches)[1][a], 4) for a, b in g.items()}))
    final = pick(s, cfgs, matches)
    ok = total >= MIN_GAIN and min(per_group.values()) > 0 and positive >= MIN_POSITIVE
    print("VERDICT:", "ADOPT the pick, subject to the label-free checks" if ok else "KEEP v31")
    print("FINAL_CFG", json.dumps(cfgs[final] if ok else V31))


if __name__ == "__main__":
    main(sys.argv[1:])
