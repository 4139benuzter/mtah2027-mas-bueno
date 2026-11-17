# Result of v35c under calib/learned_c_rule.md (one shot)

Run on 2026-11-17 after the rule (7619364, 15:46 UTC+4). Conditions 1-4b and 8 ran on the HPC (job
37857, code 7619364, public data only); conditions 5, 7a and 7b on the laptop, label-free. The
public window was not scored.

**The rule fails on condition 7a. v31 stays; no further attempt is made before the deadline.**

| condition | result | pass |
|---|---|---|
| 1. mean held-out gain >= +0.020 | +0.1189 | yes |
| 2. gain > 0 on >= 8 of 10 | 10 of 10 (+0.072 to +0.145) | yes |
| 3. Metrica > 0 and DFL > 0 | +0.1423 and +0.1088 | yes |
| 4. three groups >= +0.005 | +0.0864 | yes |
| 4b. SkillCorner mean >= -0.005 | +0.0081 | yes |
| 5. label-free checks | D 47.6%, P(D \| alive tags) 11.3%, P(D \| set piece) 91.7%, team agreement 78.4%, < 2 s 19% | yes |
| 6. numpy inference = LightGBM | 0.0 | yes |
| 7a. possessions v35c/v31 within the public range +-10% | 1.218 (346 / 284); public 0.969-1.062, bound 1.168 | **no** |
| 7b. changed-frame share within the public range +-5 points | 0.149; public 0.081-0.155 | yes |
| 8. stress test, live gaps to 9.2% | +0.1147 (Metrica +0.1412, DFL +0.1033) | yes |

**Reading.** With v31's own switch costs the learned model splits less (346 possessions against
v35's 382), but on the competition match it still adds 22% more possessions than v31, against at
most 6% on the ten public matches with the same setting. The model behaves on the competition
match outside the range where it was tested; the strong held-out gains (+0.119) are evidence about
balls like the degraded public ones, not about this match. Six attempts (v32-v35c) on 16-17 November
all ended with v31 kept.
