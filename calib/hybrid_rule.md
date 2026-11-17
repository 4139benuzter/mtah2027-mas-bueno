# Rule for the hybrid (v31 stoppages, v35c sides), written before any hybrid number

18 November 2026, 02:14 (UTC+4), commit a004779 on branch v35e. The idea came after v35c's one score on
the public window (weak dead-ball timing); this is disclosed. The window is not scored for this decision.

Hybrid label per frame: D where v31 says D; elsewhere v35c's label where v35c says H or A; otherwise v31's
label. v35c = the v35c model and decoding (GRID[8]).

Adopt only if all hold: (1) mean held-out spellF1 gain over v31 on the ten public matches (three
degradation seeds; v35c's held-out models) >= +0.020; (2) gain > 0 on >= 8 of 10; (3) Metrica and DFL
means > 0; (4) on the competition match, possessions <= 1.168 x v31's and the label-free checks of
calib/grid_rule.md pass; (5) everything pushed and verified by 02:40. Otherwise v31 stays.

## Result (HPC job 38287, 02:16; competition checks on the laptop, label-free)

All conditions hold. Held-out spellF1 v31 -> hybrid: metrica1 0.5136 -> 0.5669, metrica2 0.4873 -> 0.5111,
metrica3 0.4958 -> 0.5395, J03WMX 0.4305 -> 0.4449, J03WN1 0.3737 -> 0.3971, J03WOH 0.4586 -> 0.5187,
J03WOY 0.4299 -> 0.4585, J03WPY 0.4570 -> 0.4890, J03WQQ 0.4900 -> 0.5299, J03WR9 0.4825 -> 0.4982.
Mean gain +0.0335, 10 of 10, Metrica +0.0403, DFL +0.0306; macroF1 up on all ten (about +0.01).
Competition: 285 possessions (v31 284), D 47.5%, P(D | alive tags) 8.7%, P(D | set piece) 94.6%,
< 2 s 12%, side agreement 80.4%. Submitted file SHA-256 0743351ebdf0c96cb67d87029f8b148cfff06a33072921c691f034bc401fdbd5.
