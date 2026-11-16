# Adoption rule for the short-possession filter (fixed before any spellF1 result of the filter)

Written and committed on 2026-11-16, before any spellF1 was computed with the filter. Read before
writing it: the share of short possessions v31 misses (Metrica 1-3), the kinds of contact at
their start, and the filter's held-out precision and recall on candidates (all on other matches;
`python -m possession.touch_eval diag`). The features and the constants of
`possession/touches.py` (kick threshold 4 m/s, radius 2 m, tolerance 12 frames, longest insert
4 s, ridge 1.0) were set before this rule and are not varied below.

**Why.** v31 finds about 1 in 10 of the true possessions shorter than 2 s on the Metrica
matches. A possession's frames are few, so per-frame scores barely notice, but each one is a
whole possession for spellF1. `possession/touches.py` proposes them from brief contacts and
kicks and keeps those a logistic regression, trained on other matches, believes.

**Data.** The ten labelled other matches of `calib/grid_rule.md` (Metrica 1-3, DFL x7), each in
three ball conditions (as recorded; 59% visibility, seeds 0 and 1). s(m) = spellF1 on match m
averaged over the three conditions, official metric code. v31 is the reference.

**Held-out gain.** For each match m: train the regression on the candidates of the other nine
(all conditions); choose the threshold tau from {0.5, 0.6, 0.7, 0.8, 0.9, off} by the mean s over
the other nine, each of them scored with a model trained without it and without m (an exact tie
keeps the filter off); gain(m) = s(filter, m) - s(v31, m).
`python -m possession.touch_eval select` implements this.

**Adopt only if all hold:**
1. mean gain over the ten matches >= +0.010;
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately;
4. with SkillCorner (the 19 test matches of `calib/skillcorner_rule.md`, scored with the final
   model, never trained on), the mean of the three group means (Metrica held-out, DFL held-out,
   SkillCorner) is >= +0.005. SkillCorner alone may be negative: its labels hold 256 possessions
   per match against 360-511 on the fine labels, so they lack by construction most of the short
   possessions this filter adds; the DFL labels are of the same kind as the competition's;
5. the label-free checks of `calib/grid_rule.md` pass on the competition match, except that the
   bound on possessions shorter than 2 s is the highest share of such possessions in the true
   labels of the ten matches: 37.4% (J03WMX; the range is 25.2-37.4%), because the filter exists
   to add them.

**Final model.** Trained on all ten matches; tau chosen on all ten with leave-one-match-out
predictions (`python -m possession.touch_eval final`); weights in `calib/touch_weights.json`.

**If all hold:** the filter goes into `possession/run.py` after the decoder, the technical
document describes it, and the team says yes before any merge into main. **Otherwise** v31 stays.

The public window is not scored for this decision.
