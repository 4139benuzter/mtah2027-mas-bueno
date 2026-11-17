# Adoption rule for decoder settings re-chosen on the competition-matched ball (v34)

Written and committed on 2026-11-17, before any result of this grid was computed.

**Why.** `calib/domain_choice.json` degrades a public ball so that 17 label-free statistics of the
competition ball are matched (D = 0.0027 against 1.114 for the raw DFL ball). It also reproduces a
pattern it was not fitted to: the competition ball is missing on 40.7% of graded frames, 34.7% in
79 runs of 10 s or more; the degraded DFL balls are missing on 30.5-43.5%, in 42-62 such runs (the
raw DFL ball: 0%). On the degraded balls, v31 falls from 0.557 to 0.462 (ten matches, as recorded).
The loss comes from the smoothing and the noise (-0.085), not from the missing ball (-0.006).
v31's settings were chosen on clean balls (`calib/grid_rule.md`).

v33 (`calib/domain_rule.md`, the short-possession filter on the degraded balls) failed its rule;
this rule is about the decoder's own settings, not about a filter.

**Data.** The ten labelled other matches of `calib/grid_rule.md`, in three ball conditions (as
recorded, 59% visibility with seeds 0 and 1), each degraded by `calib/domain_choice.json` after
any removal of the ball. The competition match and the public window are not used.

**Grid.** `possession/grid_dom.py`, `GRID_DOM`; the first value of each list is v31's:
radius [1.5, 2.0, 2.5]; sigma_control [1.5, 2.5]; min_contact (= min_pass_contact) [2, 3, 5];
merge_gap [5, 12]; carry_s [0.6, 1.0]; gate_window_s [0.2, 0.5]; switch_team [15, 10, 6];
rel_slack [2.0, 4.0]. 864 settings. Nothing else changes.

**Score, pick and honest estimate.** As `calib/grid_rule.md`, with v31 as the reference: s(c, m)
= spellF1 of setting c on match m, the mean over the three degraded conditions. The pick on a set
S takes every setting within 0.003 of the best mean on S and, among them, the one with the fewest
changes from v31 (ties to the higher mean). gain(m) = s(pick on the other nine, m) - s(v31, m).
The final setting is the pick on all ten.

**Adopt only if all hold:**
1. mean held-out gain over the ten matches >= +0.010;
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately;
4. robustness to the degradation model: on the ten matches as recorded, the final setting's mean
   gain over v31 is > 0 averaged over four other degradations that also fit the target (from the
   fit workflow: sub k=8 h=8 noise 0.4 lag 4, D 0.0174; gauss s=6.27 noise 1.0 lag 2, D 0.0187;
   box h=8 noise 0.32 lag 1, D 0.0336; kalman q=1.9 r=1 noise 0.3 lag 3, D 0.0497), and >= -0.005
   under each of them;
5. on the raw public balls (ten matches, three conditions) the final setting's mean gain over v31
   is >= -0.010;
6. the mean of three group gains (Metrica held-out, DFL held-out, SkillCorner as recorded with the
   final setting, 19 matches of `calib/skillcorner_rule.md`) is >= +0.005; SkillCorner alone may
   be negative;
7. the label-free checks of `calib/grid_rule.md` pass on the competition match, with the bound on
   possessions shorter than 2 s at 37.4% (the highest true share among the ten matches).

**If all hold:** the final setting replaces v31's values (v34); the technical document describes
it; the team says yes before any merge into main; the CSV is regenerated and uploaded again.
**Otherwise** v31 stays.

The public window is not scored for this decision.
