# Adoption rule for the short-possession filter without the velocity-change feature (v32b)

Written and committed on 2026-11-17, before any spellF1 of this variant was computed.

**Why.** v32 (`calib/touch_rule.md`, `calib/touch_result.md`) passed its rule on the ten public
matches, but on the competition match it accepts 9% of the candidates where the decoder gives
the ball to the other side, against 36-58% on each of the ten matches (ball as recorded). A
label-free comparison of the candidates' features (summary statistics of the competition
tracking and of our own decoder output; no labels, no public window) shows one cause:

- the peak change of the ball's velocity at the candidate's onset has a mean of 3.7 m/s on the
  competition match, against 12-26 m/s on the ten matches. With the v32 weights it lowers the
  mean logit by 1.30. All the other 24 features together lower it by 0.04;
- the rest of the competition ball's geometry is inside the range of the ten matches as
  recorded: share of in-play frames with the ball within 1.5 m of a player 42.4% (DFL 39-45%),
  ball speed p90 14.4 m/s (13.5-14.2), contact episodes per in-play minute 20.7 (15.6-24.2),
  candidates where the decoder disagrees 2.7 per in-play minute (about 2.7-2.9).

So the competition ball is soft only at the moment of contact. A centred mean over +-10 frames
(used for `calib/touch_result.md`) also lowers the contact share and the speeds, which the
competition ball does not show, so it is not used here.

**Variant.** The filter of `calib/touch_rule.md` with one feature removed: the maximum change of
velocity over [s-3, s+3] (column 6 of `possession/touches.featurize`). 24 features remain.
Candidates (kicks included, still with the 4 m/s threshold), constants, the tau grid and the
procedure do not change. Ball conditions: as recorded, and 59% visibility with seeds 0 and 1
(the cache of v32, `cache/touch/`).

**Adopt only if all hold:**
1. mean held-out gain over the ten matches >= +0.010 (procedure of `calib/touch_rule.md`);
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately;
4. the mean of the three group means (Metrica held-out, DFL held-out, SkillCorner with the final
   model) is >= +0.005; SkillCorner alone may be negative, as in `calib/touch_rule.md`;
5. the label-free checks of `calib/grid_rule.md` pass on the competition match, with the bound
   on possessions shorter than 2 s at 37.4%, as in `calib/touch_rule.md`;
6. transfer, label-free: with the final model and its tau, the share of the competition
   match's acted candidates (the decoder gives the ball to the other side at the onset) that
   pass tau is at least the lowest such share among the ten matches (ball as recorded, same
   final model) minus 10 percentage points.

**If all hold:** v32b replaces v32 on branch `touch-detector`; the technical document describes
it; the team says yes before any merge into main, and the CSV is regenerated and uploaded again.
**Otherwise** v31 stays.

The public window is not scored for this decision.
