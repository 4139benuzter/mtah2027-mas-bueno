# Selection rule for the v32 candidates (fixed before any result was read)

Written and committed on 2026-11-17, before `possession/bench.py` ran on any match.
`possession/select_v32.py` implements exactly this rule. The competition match's labels are never
used, and the public 5-minute window is not scored before the decision below is made.

**Why.** The settings grids of `calib/grid_rule.md` and `calib/grid_window_rule.md` moved existing
settings only, and their best held-out gains were small. These candidates add new logic aimed at
how spellF1 counts a possession: a possession cut in two, or one with a spurious sliver of the
other side inside it, scores as a miss and a false alarm. Each candidate is off in v31.

**Candidates** (`possession/bench.py`; the first value of each is v31):

- `run_dead_hi_lost` None / 0.50 / 0.40: the dead-ball probability when nobody is running *and the
  ball is not tracked*, so slow play with a lost ball is not read as a stoppage so easily;
- `min_dead_s` 2.0 / 3.0: stoppages shorter than this are folded (existing setting, never gridded);
- `min_dead_same_s` 0 / 4 / 6 s: a short stoppage with the same side on both ends is folded;
- `min_blip_s` 0 / 1 / 2 s: a short possession inside the other side's play (H A H) is folded;
- `min_tail_s` 0 / 1 / 2 s: a short possession between the other side's play and a stoppage
  (H A D) is folded.

162 combinations.

**Data.** Labelled other matches available on this machine tonight: Metrica sample matches 1-3
(three ball conditions each: as recorded, and reduced to 59% visibility with seeds 0 and 1) and
the 19 SkillCorner test matches of `calib/skillcorner_rule.md` (ball as recorded; SC2017461 stays
excluded). The seven DFL matches cannot be converted on this machine (the conversion needs 48 GB
of memory) and are not used; this is stated in the technical document if a change is adopted.

**Score.** s(c, m) = spellF1 (official code) of setting c on match m, averaged over its ball
conditions. J(c, S) = mean over the two groups (Metrica, SkillCorner) of the group mean of s over
the matches of S, so each group weighs the same.

**Pick on a set S.** Every setting within 0.003 of the best J on S; among those, the one that
changes the fewest settings from v31; ties go to the higher J.

**Honest estimate (leave one match out).** For each of the 22 matches m, pick on the other 21 and
record gain(m) = s(pick, m) - s(v31, m).

**Adopt a change only if all three hold:**
1. the held-out gain, groups weighted equally, is >= +0.010 (twice the bar of the three-group
   rule, because the DFL group is missing);
2. the mean held-out gain is > 0 within Metrica and within SkillCorner, separately;
3. gain(m) > 0 on at least 15 of the 22 matches.

Otherwise v31 stays as it is.

**If adopted:** the final setting is the pick on all 22 matches. It must then pass the label-free
checks of `calib/grid_rule.md` on the competition match (`possession/diagnose.py`, no labels of
any kind): graded frames dead within 3 points of v31's 47.5%; P(D | pure alive tags) <= 12%;
P(D | set-piece pre-restart) >= 90%; team agreement with Smart Tagging phase sides >= 78%;
possessions shorter than 2 s <= 25%. If any check fails, v31 stays.

**Deadline.** If the bench has not finished in time for a submission before the deadline, v31 is
submitted unchanged.
