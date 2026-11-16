"""The v31 check on SkillCorner open data, exactly as fixed in calib/skillcorner_rule.md.

    python -m possession.sc_check calib/grid_sc/*.jsonl
"""

import json
import os
import sys

import numpy as np

from possession.grid import DEFAULT

V30 = DEFAULT
V31 = dict(DEFAULT, carry_s=0.6, min_contact=2)
EXCLUDED = {"SC2017461"}       # found the label defect; reported, not tested
RESAMPLES = 10000


def main(paths):
    rows = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["cfg"] in (V30, V31):
                rows.setdefault(r["match"], {})["v31" if r["cfg"] == V31 else "v30"] = r
    print("{:12s} {:>7s} {:>7s} {:>8s}  possessions v30/v31/labels".format("match", "v30", "v31", "d"))
    d = []
    for m in sorted(rows):
        a, b = rows[m]["v30"], rows[m]["v31"]
        diff = b["spell_f1"] - a["spell_f1"]
        mark = "  (excluded)" if m in EXCLUDED else ""
        print("{:12s} {:7.4f} {:7.4f} {:+8.4f}  {}/{}/{}{}".format(
            m, a["spell_f1"], b["spell_f1"], diff, a["n_pred_spells"], b["n_pred_spells"],
            a["n_gt_spells"], mark))
        if m not in EXCLUDED:
            d.append(diff)
    d = np.array(d)
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("test matches {}: mean d {:+.4f}, 95% interval [{:+.4f}, {:+.4f}], v31 better on {}/{}"
          .format(len(d), d.mean(), lo, hi, int((d > 0).sum()), len(d)))
    if d.mean() >= 0:
        print("DECISION 1: v31 confirmed on independent data. Keep it.")
    elif hi < 0:
        print("DECISION 2: v31 not confirmed. Run the three-group selection of the rule.")
    else:
        print("DECISION 3: inconclusive. Keep v31.")


if __name__ == "__main__":
    main(sys.argv[1:])
