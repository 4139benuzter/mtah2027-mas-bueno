# Result of the filter without the velocity-change feature (v32b) under calib/touch_rule_nodv.md

Run on 2026-11-17 after the rule was committed (ad32a07). Nothing here reads the competition
match's labels or the public window; the competition match appears only through label-free
statistics of its tracking and of our own output.

**The rule fails on condition 6. v31 stays.**

| condition | result | pass |
|---|---|---|
| 1. mean held-out gain >= +0.010 | +0.0392 | yes |
| 2. gain > 0 on >= 8 of 10 | 10 of 10 (+0.007 to +0.078) | yes |
| 3. Metrica > 0 and DFL > 0 | +0.0388 and +0.0394 | yes |
| 4. three groups equally weighted >= +0.005 | +0.0196 (SkillCorner -0.0195, 95% -0.0230 to -0.0158, 0/19) | yes |
| 5. label-free checks on the competition match | D 47.5%, P(D \| alive tags) 8.7%, P(D \| set piece) 94.6%, team agreement 81.4%, < 2 s 20% | yes |
| 6. competition share of acted candidates accepted >= lowest public share - 10 points | 24.8% against 28.1% (public 38.1-57.0%) | **no** |

tau = 0.5 (leave-one-out over all ten: 0.5297 against 0.4905 with the filter off). Weights in
`calib/touch_weights_nodv.json`. On the competition match v32b would change 824 frames and give
320 possessions against v31's 284.

**Reading.** Without the velocity-change feature the competition match's acceptance rises from
9% (v32) to 25%, but it stays below every public match. The remaining shift is mostly in the
distance from the ball to the candidate's own player at the onset (1.0 m on the competition
match against 0.53-0.73 m on the ten matches). SkillCorner, the only labelled data with
automatic tracking, loses on all 19 matches, more than with v32 (-0.004); its labels are
coarse, so part of that loss is by construction, but the sign is the same everywhere.

**Decision.** The rule fails; v31 stays in main. The variant stays on branch `touch-detector`.
