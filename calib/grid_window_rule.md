# Re-examination of the constants that touched the public window (rule fixed before results)

Written and committed on 2026-11-16, before the jobs of the "window" grid ran.

**Why.** Development versions v24-v27 set some constants with reference to the public 5-minute
window (technical document, section 3): the neutral weight of touches and unresolved contacts,
the stoppage-onset snap, and very probably the strength of the episode evidence (0.88, set next to
a 90% agreement measured on the window). The rules make the window a pipeline check, not a
tuning set. So each of these constants is re-chosen here on other matches only.

**Grid.** `possession/grid.py`, `GRID_WINDOW`, from v31: strength of controlled and pass evidence
0.88 / 0.80 / 0.95; touch 0.50 / 0.60; unresolved 0.50 / 0.60; per-frame contact weight inside a
touch 0.1 / 0.3 and inside an unresolved contact 0.3 / 0.1; snap length 3.0 / 0.0 (off) / 1.5 s;
snap minimum drop 0.15 / 0.25; snap tolerance 0.03 / 0.06. 576 combinations. The first value of
each list is the current value.

**Data and selection.** Exactly the procedure of `calib/grid_rule.md`, on the same ten matches
(Metrica 1-3, DFL x7) and three ball conditions, with v31 as the reference: pick within 0.003 of
the best mean with the fewest changes from v31; leave-one-match-out gain against v31; adopt only
if the mean held-out gain is >= +0.010, positive on at least 8 of 10 matches, and positive on
average in both groups; then only if the competition match passes the label-free checks of
`calib/grid_rule.md`. SkillCorner is not used for this choice.

**Outcome either way.** If a change is adopted, the new values are chosen on other matches. If
not, the current values stay, and the document states that they were set with reference to the
window in v24-v27 and then re-examined on ten other matches, where no alternative passed the rule.

The public window is not scored for this decision.
