# Selection rule for the v33 candidates (fixed before any result was read)

Written and committed on 2026-11-17, before `possession/bench_v33.py` ran on any match.
`possession/select_v33.py` implements exactly this rule. The competition match's labels are never
used, and the public 5-minute window is not scored before the decision is made.

**Why.** Two weaknesses of v31 that other matches can measure: (1) the ball is missing on 41% of
the competition match's graded frames, and (2) when the ball reappears already with the other
side, the decoder can only switch at the reappearance, although the hand-over happened inside the
gap. Two further weaknesses are deliberately not addressed: the missed short possessions (catching
more of them helps fine labels and hurts coarse ones, and the competition labels' granularity is
unknown) and untracked goalkeepers (the other matches track their keepers, so no change there
could be measured).

**Candidates** (`possession/bench_v33.py`; the first value of each is v31):

- `ball_gap_fill_s` 0 / 0.5 / 1 / 2 s: ball gaps up to this long, tracked on both sides, are
  filled by a straight line between the last and the next tracked position;
- `gap_handover_s` 0 / 3 / 8 s: a change of side that happens at most 0.5 s after a ball gap up to
  this long, lying entirely inside the losing side's possession, is moved to the middle of the gap
  (the point that minimises the expected error when nothing says where inside the gap it happened).

12 combinations.

**Data.** Metrica sample matches 1-3 and the 19 SkillCorner test matches of
`calib/skillcorner_rule.md` (SC2017461 excluded), **ball as recorded only**. The artificial 59%
conditions are not used: their random 0.4-5 s gaps are far shorter than the competition match's
real gaps (median 11 s) and would favour gap filling. The seven DFL matches are not available on
this machine (their ball is never missing anyway).

**Decision.** Exactly the procedure of `calib/v32_rule.md`: s(c, m) = spellF1; groups weighted
equally; pick within 0.003 of the best with the fewest changes from v31; leave one match out;
adopt only if the held-out gain is >= +0.010, positive within Metrica and within SkillCorner, and
positive on at least 15 of the 22 matches; then only if the competition match passes the
label-free checks of `calib/grid_rule.md`. Otherwise v31 stays.

**Multiple rounds.** This is the second round of new logic tested on these matches tonight
(after `calib/v32_rule.md`). Both rounds and their outcomes are reported in the technical document.
