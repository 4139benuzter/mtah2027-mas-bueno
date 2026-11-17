# Result of the learned emissions (v35) under calib/learned_rule.md and its Amendment 1

Run on 2026-11-17. `select` and `final` ran on the HPC (job 37825, code a509fff, finished 15:17
UTC+4); the outputs were read only after Amendment 1 (725af87, 15:36). The stress test ran as job
37851 (code 725af87). Conditions 5, 7a and 7b were computed on the laptop, label-free (our own
output and Smart Tagging only). The public window was not scored.

**The rule fails on condition 7a. v31 stays.**

| condition | result | pass |
|---|---|---|
| 1. mean held-out gain >= +0.020 | +0.1255 | yes |
| 2. gain > 0 on >= 8 of 10 | 10 of 10 (+0.065 to +0.166) | yes |
| 3. Metrica > 0 and DFL > 0 | +0.1483 and +0.1157 | yes |
| 4. three groups >= +0.005 | +0.0908 | yes |
| 4b. SkillCorner mean >= -0.005 | +0.0084 | yes |
| 5. label-free checks | D 47.6%, P(D \| alive tags) 11.3%, P(D \| set piece) 91.7%, team agreement 78.5%, < 2 s 25% | yes |
| 6. numpy inference = LightGBM | max abs difference 0.0 | yes |
| 7a. possessions v35/v31 on the competition within the public range +-10% | 1.345 (382 / 284); public 0.985-1.132, bound 1.245 | **no** |
| 7b. changed-frame share within the public range +-5 points | 0.150; public 0.087-0.160 | yes |
| 8. stress test, live gaps to 9.2% | +0.1201 (Metrica +0.1493, DFL +0.1076) | yes |

Decoding setting chosen on all ten: T 1.0, switch_team 4, switch_dead 25, no gate.

**Diagnostics after the result (they decide nothing).**
- Without v31's Smart Tagging terms, v35 gives 384 possessions (ratio 1.352): the tag terms are not
  the cause of 7a.
- On the 19 SkillCorner matches (model trained on all ten), v35/v31 possessions is 1.012-1.181
  (median 1.079); v35 301, v31 277, true 243 on average; gain +0.0084.

**Reading.** On the public matches the learned model is far better than v31 (+0.125, every match,
robust to live gaps, positive on SkillCorner). On the competition match it splits possessions far
more than anywhere it was tested (+35% against at most +18%), so the held-out evidence does not
cover its behaviour there. This is the condition added to catch a v32-like failure to transfer.
