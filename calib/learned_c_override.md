# Override: v35c is submitted although it failed condition 7a of calib/learned_c_rule.md

Written on 17 November 2026 at about 23:50 (UTC+4), after every v35c result
(calib/learned_c_result.md, a9915f4). The rule and the result files are not changed.

**What the rule said.** v35c failed condition 7a (possessions v35c/v31 on the competition match
1.218, against a public range of 0.969-1.062 widened to 1.168). The rule then said: "If it fails,
v31 stays and no further attempt is made before the deadline."

**What we decided, and who.** The team (the user, on 17 November at about 23:45 UTC+4) chose to
submit v35c anyway, on the condition that three independent audits (technical, decision,
compliance) found no blocker. The reason is the team's objective: a rank in the top 20. v31's
expected spellF1 on the hidden match is about 0.40, below the 0.47-0.55 that other teams report on
their own tests with ball loss (self-reported, not comparable). v35c passed every other condition
of its rule (held-out gain +0.119 on 10 of 10 degraded public matches; +0.115 with live ball gaps
at the competition's 9.2%; SkillCorner +0.008; label-free checks; numpy inference equal to
LightGBM; changed-frame share 0.149 inside the public range 0.081-0.155).

**What this costs.** This is the seventh decision on the same ten matches on 16-17 November
(v32, v32b, v33, v34, v35, v35c, and this override), and it was made after all v35c results.
The held-out gains are evidence about degraded public balls only; there is no labelled evidence
on the competition ball. The decision audit put the chance that v35c beats v31 on the hidden match
at about 0.65 (range 0.55-0.75), with an expected gain of about +0.03 and a band of about -0.06 to
+0.12. A label-free check points against v35c: agreement with the Smart Tagging phase sides falls
from 82.2% (v31) to 78.4%.

**What is not done.** No other decoding setting of the same model is picked (that would be an
eighth decision after the results). The public window is not scored.

**Found by the technical audit, after the decision.** `features.compute` divides the runners by all
player slots (`np.nanmean(speed > 2.0, axis=1)`), not by the tracked players. The competition match
has 48 slots; the training matches have 26-40 (Metrica 26/28/35, DFL 39-40), so the share as the
model sees it is 0.42 of the per-tracked share here, against 0.53-0.85 in training. Diagnostic only
(it decides nothing): with the four running-share inputs of the classifier multiplied by 0.55/0.42
(the DFL scale), v35c gives 339 possessions here instead of 346; by 0.79/0.42, 325. The defect
explains a small part of condition 7a. It is not fixed: a fix needs retraining and a new test, an
eighth decision, with no time before the deadline.

(Names: these v32-v35c are the attempts on branch `touch-detector`. The "v32" and "v33" rounds in
`calib/v32_rule.md` and `calib/v33_rule.md` on main are a separate series with the same names.)

**Public window, after the freeze (18 November about 01:25 UTC+4).** At the team's request the
submitted file (ad4c1503) was scored once on the public 5-minute window, for information only:
macroF1 0.7027, spellF1 0.4121, 16 possessions (truth 17); v31 had 0.9021 / 0.5375 / 15. The
dead-ball class is weak (F1 0.49): one 20 s stoppage called alive, then live play called dead.
The rules make the window a correctness check, not a tuning set, so the result did not change
the submission. No defect was found: the errors sit at three stoppages, not at a block edge.

**Reverted (18 November about 01:35 UTC+4).** After the window score above, the team decided to
return to v31, the version that the pre-committed rule had kept and the file submitted since
16 November (SHA-256 f34036f0...). This decision came after the window score and is disclosed as
such in the technical document (Section 3). Nothing was tuned on the window; no setting of v31
changed. `learned_emissions` is False on main again; the v35c model file stays in the repository
as a record.
