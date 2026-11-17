# Adoption rule for v35c: the v35 model with v31's own decoding setting

Written and committed on 2026-11-17, after the v35 result (calib/learned_result.md, af89126) and
before any number of v35c was computed or read.

**Why, and what it costs.** v35 passed every condition of calib/learned_rule.md except 7a: on the
competition match it gave 35% more possessions than v31 (382 against 284), while on every
public match and on SkillCorner the ratio was at most 1.18. Its decoding setting, chosen on the
public matches, was switch_team 4 with no gate. This is the sixth attempt on the same ten matches
(after v32, v32b, v33, v34, v35), and it was chosen after seeing that failure. It is one shot: if
it fails, v31 stays and no further attempt is made before the deadline.

**v35c.** The same trained trees as v35 (calib/learned_model.json, the model trained on all ten
matches; the held-out models of v35 for the held-out estimate). Decoding is fixed in advance at
v31's own setting, `GRID[8]` of possession/learned_eval.py: T 1.0, switch_team 15, switch_dead 25,
gate on. Nothing is selected: no other setting is scored for this decision.

**Conditions** (all must hold; the same thresholds as calib/learned_rule.md with its Amendment 1):
1. mean held-out gain over the ten matches >= +0.020 (match m predicted by the model trained on the
   other nine, the three degradation seeds averaged; `FIXED_INDEX=8 python -m possession.learned_eval fixed`);
2. gain(m) > 0 on at least 8 of the 10;
3. mean gain > 0 within the Metrica matches and within the DFL matches;
4. mean of the three group means (Metrica, DFL, SkillCorner with the model trained on all ten) >= +0.005;
4b. SkillCorner mean gain >= -0.005;
5. the label-free checks of calib/grid_rule.md on the competition match (bound on possessions shorter
   than 2 s at 37.4%);
6. numpy inference = LightGBM (the same trees as v35: max abs difference 0.0);
7a. possessions v35c/v31 on the competition match within the range of that ratio over the ten
   held-out public matches (degraded, seed 0, this setting), widened by 10% at each end;
7b. changed-frame share on the competition within the public range of that share, widened by 5 points;
8. stress test with live gaps to 9.2% (`FIXED_INDEX=8 python -m possession.learned_eval stress`):
   mean held-out gain >= +0.010, > 0 within Metrica and within DFL.

**If all hold:** the team says yes before any merge into main; `calib/learned_model.json` is
committed with `decode` set to `GRID[8]`; the technical document describes v35c with all six attempts
(v32-v35c), their rules, results and amendments; the CSV is regenerated from main, validated and
uploaded again in the same form response. **Otherwise** v31 stays.

The public window is not scored for this decision.
