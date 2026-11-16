# Result of the short-possession filter (v32) under calib/touch_rule.md

Run on 2026-11-16 after the rule was committed (892656f). Nothing here reads the competition
match's labels or the public window; the competition match appears only through label-free
statistics of its tracking and of our own output.

**The rule.** All five conditions hold.

| condition | result |
|---|---|
| 1. mean held-out gain >= +0.010 (ten matches, three ball conditions) | +0.0435 |
| 2. gain > 0 on >= 8 of 10 | 10 of 10 (+0.010 to +0.080) |
| 3. Metrica > 0 and DFL > 0 | +0.0384 and +0.0457 |
| 4. three groups equally weighted >= +0.005 | +0.0266 (SkillCorner -0.0043, 95% -0.0065 to -0.0020, 5/19) |
| 5. label-free checks on the competition match | D 47.5%, P(D \| alive tags) 8.7%, P(D \| set piece) 94.6%, team agreement 82.0%, < 2 s 16% |

**But the gain does not transfer to the competition ball.** On the competition match the filter
accepts 17 of 167 candidates where the decoder disagrees (10%), against 45% on the ten matches,
and adds 4 possessions (288 against v31's 284). The reason is in the tracking: at the start of a
contact the competition ball's velocity changes by a median of 3.0 m/s, against 9-12 m/s on
Metrica and DFL (and 4.0 on SkillCorner, also automatic tracking). Its ball is soft at contacts.

Smoothing the ten matches' ball (centred mean over +-10 frames, which brings their median to
about 3 m/s) reproduces this: the filter, trained as above, then accepts 6% of such candidates,
and its leave-one-match-out gain falls to +0.0010 (4 of 10 matches; Metrica +0.0057, DFL -0.0010).
Retrained on the smoothed ball, its candidate precision falls from 0.79 to about 0.50, with a
true candidate near only 37% of true possession starts (59% on the raw ball).

v31 itself scores 0.441 on the smoothed ball (mean of the three conditions; 0.491 on the raw
ball), so the competition score may sit nearer the lower end of the earlier estimates.

**Decision.** The rule passes, but the evidence says the gain belongs to clean ball tracking and
is about zero on a ball like the competition's (+0.001 on the smoothed matches, -0.004 on
SkillCorner). The rule leaves the final yes to the team. Recommendation: keep v31 in main; keep
this filter on branch `touch-detector` for cleaner tracking (for example the Final Pitch data).
