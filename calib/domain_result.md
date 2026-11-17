# Result of the domain-matched short-possession filter (v33) under calib/domain_rule.md

Run on 2026-11-17 after the rule (c1e150c) and the degradation choice (cd28477) were committed.
Nothing here reads the competition match's labels or the public window.

**Degradation.** The fit workflow evaluated 633 configurations of eight families on the seven DFL
matches (public data only). Lowest D: `nohold` (holds of >= 500 frames become missing, Gaussian
smoothing s = 7 frames with a lighter s = 4 above 16 m/s, correlated noise of 0.24 m over 7
frames added after the smoothing, output rounded to 6 mm), D = 0.0027, against 1.114 for the raw
DFL ball; re-run in the main session: 0.0027. Next best families: rts 0.0037, sub 0.0174, gauss
0.0187, box 0.0336, kalman 0.0497, ema 0.0648. On Metrica (not fitted) D = 0.042.

Out-of-sample check (not fitted): the competition ball is missing on 40.7% of graded frames,
34.7% in 79 runs of 10 s or more; the degraded DFL balls on 30.5-43.5%, in 42-62 such runs.

**The rule fails on condition 1. v31 stays.**

| condition | result | pass |
|---|---|---|
| 0. D <= 0.557 | 0.0027 | yes |
| 1. mean held-out gain >= +0.010 | -0.0004 | **no** |
| 2. gain > 0 on >= 8 of 10 | 3 of 10 | **no** |
| 3. Metrica > 0 and DFL > 0 | -0.0010 and -0.0002 | **no** |
| 4. three groups >= +0.005 | -0.0011 (SkillCorner -0.0022, 3/19) | **no** |

On a ball like the competition's, the short-possession filter has nothing left to find: the
contacts that start short possessions are blurred away.

**v31 on the degraded balls.** Mean over the three conditions 0.421 (0.491 on the raw balls); as
recorded 0.462 (0.557). The loss is the smoothing and the noise (-0.085 as recorded), not the
missing ball (-0.006). This motivates `calib/domain_decoder_rule.md` (v34), which re-chooses the
decoder's own settings on the degraded balls.
