# Selection rule for the v34 candidates (fixed before any result was read)

Written and committed on 2026-11-17, before `possession/bench_v34.py` ran on any match.
`possession/select_v34.py` implements this rule; it is the procedure of `calib/v32_rule.md` unchanged.

**Where the ideas come from.** On the last evening we read the code and documentation (never
their data or results files) of the public repositories of other teams in this competition and
kept three ideas that our decoder lacked and that their authors had measured on other matches:

- `exit_snap` off / on: a stoppage starts where the ball is seen leaving the pitch (moved forward
  if the first out-of-pitch sighting comes within 5 s after the D start, back if the ball left
  within the 3 s before it); never creates a stoppage or moves its end;
- `oob_reentry_s` 0 / 1 s: an out-of-pitch sighting that is back in play and moving (above 3 m/s)
  within 1 s is an aerial ball drawn outside the lines, not evidence of a stoppage;
- `turnover_backdate_s` 0 / 0.5 / 1 s: a change of side moves back over the frames just before it
  on which the new side is on the ball (Gaussian 1.5 m control >= 0.3 and at least the old side's).

12 combinations, all off in v31. No code of another team is copied; each idea is re-implemented.

**Data and decision.** Exactly as `calib/v32_rule.md`: Metrica 1-3 (three ball conditions) and the
19 SkillCorner test matches (as recorded), groups weighted equally; pick within 0.003 of the best
with the fewest changes from v31; leave one match out; adopt only if the held-out gain is
>= +0.010, positive within each group and positive on at least 15 of 22 matches; then only if the
competition match passes the label-free checks of `calib/grid_rule.md`. Otherwise v31 stays.

This is the third round of new logic on these matches tonight; all three are reported.
