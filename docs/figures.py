"""Figures of the technical document, from committed validation results only (other matches).

    python docs/figures.py          # writes docs/fig_granularity.pdf

Reads calib/grid/*.jsonl and calib/grid_sc/*.jsonl (spellF1 of every grid setting on the labelled
other matches). Nothing here reads the competition match or its public window.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
V30 = {"switch_team": 15.0, "gate_window_s": 0.2, "hold_s": 2.0, "carry_s": 1.0, "radius": 1.5,
       "min_contact": 3, "switch_dead": 25.0}
V31 = dict(V30, carry_s=0.6, min_contact=2)
EXCLUDED = {"SC2017461"}           # calib/skillcorner_rule.md: reported, not tested

# Validated categorical slots 1-3 (blue, orange, aqua) plus a marker shape per group.
GROUPS = [("Metrica (3)", "metrica", "#2a78d6", "o"),
          ("DFL (7)", "J03", "#eb6834", "s"),
          ("SkillCorner (19)", "SC", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load():
    rows = {}
    paths = glob.glob(os.path.join(ROOT, "calib", "grid", "*__rec.jsonl")) + \
        glob.glob(os.path.join(ROOT, "calib", "grid_sc", "*__rec.jsonl"))
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r["match"] in EXCLUDED:
                continue
            for name, cfg in (("v30", V30), ("v31", V31)):
                if r["cfg"] == cfg:
                    rows.setdefault(r["match"], {})[name] = r
    return rows


def main():
    rows = load()
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.3, 2.75))
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for label, prefix, colour, marker in GROUPS:
        xs, ys = [], []
        for match, r in sorted(rows.items()):
            if match.startswith(prefix):
                xs.append(r["v31"]["n_pred_spells"] / r["v31"]["n_gt_spells"])
                ys.append(r["v31"]["spell_f1"] - r["v30"]["spell_f1"])
        ax.scatter(xs, ys, s=30, marker=marker, color=colour, edgecolor="white", linewidth=0.8,
                   label=label, zorder=3)
    ax.set_xlabel("possessions predicted by v31 / possessions in the labels (ball as recorded)")
    ax.set_ylabel("spellF1, v31 minus v30")
    ax.text(0.98, 0.0055, "labels finer than our output ←", ha="right", va="center",
            color=MUTED)
    ax.text(1.02, 0.0055, "→ labels coarser than our output", ha="left", va="center",
            color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderaxespad=0.4)
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig_granularity.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
