# Independent check of v31 on SkillCorner open data (rule fixed before the test matches ran)

Written and committed on 2026-11-16, before 19 of the 20 SkillCorner matches were converted or
scored. Match 2017461 was converted first; it showed a defect in the first label version
(long event-free stretches labelled as alive play) and led to the X rule in
`possession/skillcorner.py`. On it, before the fix, v31 scored 0.405 and v30 0.436. It is
reported but excluded from the test below.

**Data.** The other 19 matches of https://github.com/SkillCorner/opendata (MIT licence),
converted by `possession/skillcorner.py`, ball as recorded (detected on about 55% of in-play
frames; no artificial removal). Smart Tagging and the keeper fix are off, as for the other
validation matches.

**Test.** d(m) = spellF1(v31, m) - spellF1(v30, m), official metric code, for each of the 19
matches; the mean of d and a 95% paired bootstrap interval (10,000 resamples of matches, seed 0).

**Decision.**
1. Mean d >= 0: v31 is confirmed on independent data. Keep it.
2. Mean d < 0 and the whole interval below 0: v31 is not confirmed. Then re-run the selection of
   `calib/grid_rule.md` with three groups (Metrica, DFL, SkillCorner-19), the group means given
   equal weight, and the leave-one-match-out gain measured against v31. Change main only if that
   held-out gain is at least +0.005 and is not negative in any group; otherwise keep v31 and
   report the conflict to the team.
3. Otherwise (mean d < 0 with an interval that includes 0): inconclusive. Keep v31.

The public window is not scored for this check.
