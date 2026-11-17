# Adoption rule for the domain-matched short-possession filter (v33)

Written and committed on 2026-11-17, before the degradation was fitted and before any spellF1 on
a degraded ball was computed.

**Why.** v32 and v32b (`calib/touch_result.md`, `calib/touch_result_nodv.md`) learn from public
balls that differ from the competition ball: on the competition ball the change of velocity at a
contact onset is soft (median 2.9 m/s against 7.7 on the DFL matches), the frame-to-frame jitter
is larger (median change of velocity 1.38 m/s against 0.42), and the ball is almost never exactly
still between frames (0.08% of in-play frames against 31.5%). Both filters accept far fewer of the
competition match's candidates than of any public match's. Here the public ball is first degraded
by a tracker model whose settings are fitted to label-free statistics of the competition ball,
and the filter is then trained and tested on the degraded public balls.

**Target.** `calib/domain_target.json`: 17 label-free statistics of the competition tracking
(`python -m possession.degrade target`; summary numbers only, no labels, no public window).

**Choice of the degradation (label-free, fixed before any spellF1).** Every configuration
evaluated by `python -m possession.degrade fit` (families none, box, gauss, ema, kalman, sub, each
with optional position noise and lag; and any further family written as a function of the same
kind) on the seven DFL matches as recorded, seed 0, scored by `degrade.distance` of the pooled
(median over the seven) statistics against the target. The configuration with the lowest D is
chosen; within 2% of the lowest D, the one with fewer settings. It is re-run once in the main
session to confirm its D, and written to `calib/domain_choice.json` before any spellF1 on a
degraded ball is computed. It must have D <= half of the raw ball's D (1.114, so <= 0.557);
otherwise this rule stops here and v31 stays.

**Test.** The procedure of `calib/touch_rule.md` (v32, all 25 features, the same candidates,
constants and tau grid), with every public ball degraded by the chosen configuration after any
removal of the ball: the ten matches in three conditions (as recorded, 59% visibility seeds 0 and
1). The decoder (v31) runs on the degraded ball, as it would on the competition ball. SkillCorner
is not degraded (it is already automatic tracking). gain(m) = s(filter, m) - s(v31, m), both on
the degraded ball.

**Adopt only if all hold:**
1. mean held-out gain over the ten matches >= +0.010;
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately;
4. the mean of the three group means (Metrica, DFL, SkillCorner with the final model) >= +0.005;
   SkillCorner alone may be negative, as in `calib/touch_rule.md`;
5. the label-free checks of `calib/grid_rule.md` pass on the competition match, with the bound on
   possessions shorter than 2 s at 37.4%, as in `calib/touch_rule.md`;
6. transfer, label-free: with the final model and its tau, the share of the competition match's
   acted candidates that pass tau is at least the lowest such share among the ten degraded
   matches (as recorded, same final model) minus 10 percentage points.

**If all hold:** v33 replaces v32 on branch `touch-detector`; the technical document describes
it; the team says yes before any merge into main; the CSV is regenerated and uploaded again.
**Otherwise** v31 stays. No further variant is tried after this one fails.

The public window is not scored for this decision.
