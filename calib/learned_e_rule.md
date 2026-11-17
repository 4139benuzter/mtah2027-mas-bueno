# Adoption rule for v35e: v35c averaged over three training draws

Written and committed on 18 November 2026 at 01:14 (UTC+4), before any v35e number existed. (The
first commit said "about 00:35"; that was a misread clock. The commit time, 01:14, is correct.)

**Why.** v35c's model used only 3 of every 10 graded frames (one frame offset per degradation
seed). Averaging models trained on the other offsets reduces the model's variance and uses more
of the same data. Nothing is chosen: no setting, feature or threshold changes.

**v35e.** The mean of the class probabilities of three LightGBM models: k = 0 is v35c's model;
k = 1 and k = 2 are trained exactly like it (same features, parameters, 120 rounds, team swap)
except that the training rows of degradation seed s are the frames with index % 10 == (s + 3k) % 10,
and LightGBM's seed is k. Decoding: v35c's, GRID[8] (T 1.0, switch_team 15, switch_dead 25, gate on).
Held-out: for each of the ten matches, the three models are trained without it, as for v35c.

**Adopt v35e instead of v35c only if all hold by 02:00 (UTC+4):**
1. mean held-out spellF1 gain of v35e over v35c on the ten matches (three seeds) >= +0.004;
2. gain > 0 on at least 7 of the 10;
3. mean gain > 0 within Metrica and within DFL;
4. SkillCorner (19 matches, never trained on) mean gain over v35c >= -0.003;
5. on the competition match: possessions v35e/v31 <= 1.218 (v35c's ratio) and the label-free checks
   of calib/grid_rule.md pass (bound on possessions shorter than 2 s 37.4%);
6. numpy inference of each exported model equals LightGBM (max abs difference < 1e-6).

Otherwise v35c stays. The public window is not scored.
