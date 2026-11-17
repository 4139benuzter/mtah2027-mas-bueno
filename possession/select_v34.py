"""The v34 decision of calib/v34_rule.md (same procedure as v32), applied to possession/bench_v34.py output.

Groups: Metrica (sample games 1-3, spellF1 averaged over three ball conditions) and SkillCorner
(19 test matches, ball as recorded). J(c, S) = mean over the groups of the group mean of
spellF1 over the matches of S, so each group weighs the same. Pick on S: every setting within
0.003 of the best J, then the fewest changes from v31, ties to the higher J. Held-out gain: for
each match, pick on the other 21 and take s(pick, m) - s(v31, m). Adopt only if the held-out gain
(groups weighted equally) is >= +0.010, positive within each group, and positive on at least 15
of the 22 matches. Otherwise v31 stays.

    python -m possession.select_v34 calib/bench_v34/bench.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np

from possession.bench_v34 import V31
from possession.sc_check import EXCLUDED

NEAR_BEST = 0.003
MIN_GAIN = 0.010
MIN_POSITIVE = 15
GROUPS = {"Metrica": 3, "SkillCorner": 19}
CONDS = {"Metrica": 3, "SkillCorner": 1}


def key(cfg):
    return tuple(sorted(cfg.items()))


def group(match):
    return "Metrica" if match.startswith("metrica") else "SkillCorner"


def load(paths):
    raw = defaultdict(lambda: defaultdict(list))
    cfgs = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["match"] in EXCLUDED:
                continue
            k = key(r["cfg"])
            cfgs[k] = r["cfg"]
            raw[k][r["match"]].append(r["spell_f1"])
    matches = sorted({m for per in raw.values() for m in per})
    s = {k: {m: float(np.mean(per[m])) for m in matches}
         for k, per in raw.items() if all(len(per[m]) == CONDS[group(m)] for m in matches)}
    return s, cfgs, matches


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
    near = [k for k in J if J[k] >= best - NEAR_BEST]
    return min(near, key=lambda k: (changes(cfgs[k]), -J[k])), J


def diff(cfg):
    return {a: b for a, b in cfg.items() if b != V31[a]} or "v31"


def main(paths):
    s, cfgs, matches = load(paths)
    counts = defaultdict(int)
    for m in matches:
        counts[group(m)] += 1
    assert dict(counts) == GROUPS, dict(counts)
    ref = key(V31)
    print("{} settings complete on {}".format(len(s), dict(counts)))
    gains, picks = {}, defaultdict(int)
    for m in matches:
        k, _ = pick(s, cfgs, [x for x in matches if x != m])
        gains[m] = s[k][m] - s[ref][m]
        picks[k] += 1
    for k, n in sorted(picks.items(), key=lambda kv: -kv[1]):
        print("  held-out pick in {} folds: {}".format(n, diff(cfgs[k])))
    total, per_group = balanced(gains, matches)
    positive = sum(g > 0 for g in gains.values())
    print("held-out gain vs v31 (groups weighted equally): {:+.4f}; per group {}; "
          "positive on {} of {}".format(total, {g: round(v, 4) for g, v in per_group.items()},
                                        positive, len(matches)))
    final, J = pick(s, cfgs, matches)
    jf, gf = balanced(s[final], matches)
    jr, gr = balanced(s[ref], matches)
    print("pick on all {}: {}  J {:.4f} (v31 {:.4f}); groups pick {} v31 {}".format(
        len(matches), diff(cfgs[final]), jf, jr, {g: round(v, 4) for g, v in gf.items()},
        {g: round(v, 4) for g, v in gr.items()}))
    ok = total >= MIN_GAIN and min(per_group.values()) > 0 and positive >= MIN_POSITIVE
    print("VERDICT:", "ADOPT the pick, subject to the label-free checks" if ok else "KEEP v31")
    print("FINAL_CFG", json.dumps(cfgs[final] if ok else V31))


if __name__ == "__main__":
    main(sys.argv[1:])
