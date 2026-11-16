# Selection rule for the v30 settings grid (fixed before any grid result was read)

Written and committed on 2026-11-16, before the grid jobs ran. `possession/grid_select.py`
implements exactly this rule. The competition match's labels are never used, and the public
5-minute window is not scored before the decision below is committed.

**Data.** The ten labelled other matches of section 3 of the technical document (Metrica sample
matches 1-3, DFL J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9), each in three ball
conditions: as recorded, and reduced to 59% visibility with seeds 0 and 1.

**Grid.** `possession/grid.py`, `GRID`: 648 combinations of seven existing settings. The first
value of each list is the v30 default. No new logic.

**Score.** s(c, m) = spellF1 of setting c on match m, averaged over the three ball conditions.
J(c, S) = mean of s(c, m) over the matches m in set S.

**Pick on a set S.** Take every setting within 0.003 of the best J on S. Among those, take the
one that changes the fewest settings from v30; ties go to the higher J.

**Honest estimate (leave one match out).** For each of the ten matches m, pick on the other nine
and record gain(m) = s(pick, m) - s(v30, m).

**Adopt a change only if all three hold:**
1. mean gain over the ten matches >= +0.010;
2. gain(m) > 0 on at least 8 of the 10 matches;
3. mean gain > 0 within the Metrica matches and within the DFL matches, separately.

Otherwise v30 stays as it is.

**If adopted:** the final setting is the pick on all ten matches. It must then pass label-free
checks on the competition match (`possession/diagnose.py`, no labels of any kind):

- graded frames dead within 3 points of v30's 47.5%;
- P(D | pure alive tags) <= 12%;
- P(D | set-piece pre-restart) >= 90%;
- team agreement with Smart Tagging phase sides >= 78%;
- possessions shorter than 2 s <= 25%.

If any check fails, v30 stays as it is. The result is then offered to the team (Aaron) before any
merge into main.
