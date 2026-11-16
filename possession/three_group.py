"""Decision 2 of calib/skillcorner_rule.md: the selection of calib/grid_rule.md with three groups.

Groups: Metrica (3 matches, 3 ball conditions), DFL (7 matches, 3 conditions) and SkillCorner
(19 test matches, ball as recorded). s(c, m) = spellF1 of setting c on match m averaged over that
match's conditions. J(c, S) = mean over the groups of the group mean of s over the matches of S,
so each group weighs the same. Pick on S: within 0.003 of the best J, fewest changes from v30,
ties to the higher J. Held-out gain: for each match m, pick on the other 28 and take
s(pick, m) - s(v31, m); summarise by the same equal group weighting. Change main only if that
gain is >= +0.005 and no group's mean held-out gain is negative.

    python -m possession.three_group calib/grid/*.jsonl calib/grid_sc/*.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np

from possession.grid import DEFAULT, V31
from possession.sc_check import EXCLUDED

NEAR_BEST = 0.003
MIN_GAIN = 0.005


def key(cfg):
    return tuple(sorted(cfg.items()))


def group(match):
    return "Metrica" if match.startswith("metrica") else "SkillCorner" if match.startswith("SC") \
        else "DFL"


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
    need = {m: (1 if group(m) == "SkillCorner" else 3) for m in matches}
    s = {k: {m: float(np.mean(per[m])) for m in matches}
         for k, per in raw.items() if all(len(per[m]) == need[m] for m in matches)}
    return s, cfgs, matches


def balanced(values, matches):
    by = defaultdict(list)
    for m in matches:
        by[group(m)].append(values[m])
    return float(np.mean([np.mean(v) for v in by.values()])), {g: float(np.mean(v)) for g, v in by.items()}


def changes(cfg):
    return sum(cfg[k] != DEFAULT[k] for k in DEFAULT)


def pick(s, cfgs, subset):
    J = {k: balanced(v, subset)[0] for k, v in s.items()}
    best = max(J.values())
    near = [k for k in J if J[k] >= best - NEAR_BEST]
    return min(near, key=lambda k: (changes(cfgs[k]), -J[k])), J


def main(paths):
    s, cfgs, matches = load(paths)
    counts = defaultdict(int)
    for m in matches:
        counts[group(m)] += 1
    assert dict(counts) == {"Metrica": 3, "DFL": 7, "SkillCorner": 19}, dict(counts)
    ref = key(V31)
    print("{} settings complete on {}".format(len(s), dict(counts)))
    gains = {}
    picks = defaultdict(int)
    for m in matches:
        k, _ = pick(s, cfgs, [x for x in matches if x != m])
        gains[m] = s[k][m] - s[ref][m]
        picks[k] += 1
    for k, n in sorted(picks.items(), key=lambda kv: -kv[1]):
        print("  held-out pick in {} folds: {}".format(n, {a: b for a, b in cfgs[k].items() if b != DEFAULT[a]} or "v30"))
    total, per_group = balanced(gains, matches)
    print("held-out gain vs v31 (groups weighted equally): {:+.4f}; per group {}".format(
        total, {g: round(v, 4) for g, v in per_group.items()}))
    final, J = pick(s, cfgs, matches)
    jf, gf = balanced(s[final], matches)
    jr, gr = balanced(s[ref], matches)
    print("pick on all 29: {}  J {:.4f} (v31 {:.4f}); groups pick {} v31 {}".format(
        {a: b for a, b in cfgs[final].items() if b != DEFAULT[a]} or "v30", jf, jr,
        {g: round(v, 4) for g, v in gf.items()}, {g: round(v, 4) for g, v in gr.items()}))
    ok = total >= MIN_GAIN and min(per_group.values()) >= 0
    print("VERDICT:", "CHANGE main to the pick" if ok else "KEEP v31 and report the conflict")
    print("FINAL_CFG", json.dumps(cfgs[final] if ok else V31))


if __name__ == "__main__":
    main(sys.argv[1:])
