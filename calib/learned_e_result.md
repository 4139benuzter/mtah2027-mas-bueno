# Result of v35e under calib/learned_e_rule.md

Run on 18 November 2026, HPC job 38254 (01:15-01:20 UTC+4; code 7619364 plus the job script
`~/v34run/bag_v35e.py`, public data only). The public window was not scored.

**The rule fails on conditions 1 and 3. v35c stays.**

| condition | result | pass |
|---|---|---|
| 1. mean held-out gain over v35c >= +0.004 | +0.0022 | **no** |
| 2. gain > 0 on >= 7 of 10 | 7 of 10 (-0.0057 to +0.0115) | yes |
| 3. Metrica > 0 and DFL > 0 | -0.0027 and +0.0043 | **no** |
| 4. SkillCorner mean >= -0.003 | +0.0019 (10 of 19) | yes |
| 5. competition checks | not run (1 and 3 failed) | - |
| 6. numpy inference = LightGBM | 0.0 for both new models | yes |

Held-out spellF1, v35c -> v35e: metrica1 0.6575 -> 0.6530, metrica2 0.6279 -> 0.6300, metrica3
0.6383 -> 0.6326, J03WMX 0.5022 -> 0.5072, J03WN1 0.4893 -> 0.4914, J03WOH 0.6036 -> 0.6092, J03WOY
0.5306 -> 0.5421, J03WPY 0.5728 -> 0.5724, J03WQQ 0.6185 -> 0.6213, J03WR9 0.5668 -> 0.5704.
macroF1 changes by at most 0.002 on any match. Averaging three training draws changes the score by
about the noise of one draw; the single v35c model is not a lucky draw. The switch
`Params.learned_bag` stays off, and the extra models are not committed.
