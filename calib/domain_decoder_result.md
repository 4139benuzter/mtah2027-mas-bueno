# Result of the decoder re-tune on the competition-matched ball (v34) under calib/domain_decoder_rule.md

Run on 2026-11-17 after the rule (af37a53) and its Amendment 1 (a16adec) were committed. The grid
(864 settings x 10 public matches x 3 degradation seeds) ran on the HPC (jobs 37711 and 37713,
public data only; a check row, v31 on J03WMX seed 0, gives 0.4288 there and on the laptop). The
outputs of the first, stopped run with the old conditions were not read
(`cache/grid_dom/dom_discarded_unread/`). Nothing here reads the competition match's labels or the
public window.

**The rule fails on conditions 1 and 3. v31 stays.**

| condition | result | pass |
|---|---|---|
| 1. mean held-out gain >= +0.010 | +0.0096 | **no** |
| 2. gain > 0 on >= 8 of 10 | 8 of 10 | yes |
| 3. Metrica > 0 and DFL > 0 | Metrica -0.0062, DFL +0.0164 | **no** |

Held-out gains: J03WMX +0.0186, J03WN1 +0.0398, J03WOH +0.0193, J03WOY +0.0200, J03WPY +0.0040,
J03WQQ +0.0032, J03WR9 +0.0100, metrica1 +0.0143, metrica2 -0.0170, metrica3 -0.0159. The pick on
nine matches was radius 2.5 m and min_contact 5 frames for nine of the ten held-out matches
(radius 2.0 for J03WMX). The pick on all ten: radius 2.5, min_contact 5; mean 0.4732 against
v31's 0.4619 on the degraded balls (in-sample).

Conditions 4-7 were not run: the rule had already failed.

**Reading.** On a ball like the competition's, a wider contact radius and a longer minimum contact
help on all seven DFL matches and hurt on two of the three Metrica matches. The review
(wf_0eb98913-5c9) also found that such settings add predicted possessions and lose on coarser
labels (SkillCorner). Which kind of labels the competition uses is unknown; the gain is not clear
enough to change the submission.
