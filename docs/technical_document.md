# Frame-level ball possession from automatic tracking

**More Than A Hack 2027, Phase 1. Team: Más Bueno (Aaron Binoj Amacadu, Karthik, James).**

## 1. Approach

We treat possession as a hidden state with three values, home (`H`), away (`A`) and dead ball
(`D`), that changes rarely, and decode the most probable state sequence over the 199,319 frames
with a Viterbi pass. Every piece of evidence is expressed as a per-frame log-probability of the
three states; a cost is charged for every change of state. The chain therefore only changes its
answer when the evidence for the new state, accumulated over consecutive frames, outweighs the
cost. This is what turns noisy per-frame cues into whole possessions, which is what the ranking
metric scores.

The evidence comes from the provided automatic tracking (ATD) and the Smart Tagging, nothing
else. The design question throughout was not "which player is nearest to the ball" but "what does
the challenge mean by *controlled* possession and by *dead*", and the answer is encoded in the
following cues.

### 1.1 Who has the ball: control episodes and their outcomes

The challenge defines a change of possession as the opponent gaining **controlled** possession,
and names two proxies: a player keeps the ball for a few seconds, or a team completes passes. A
touch without control, a deflection, or a header that is not kept do not count. Per-frame
nearest-player evidence cannot express that, because whether a touch counts depends on what
happened afterwards.

We therefore cut the tracked play into **contact episodes**: runs of frames where the ball is
within 1.5 m of a player and moving at roughly that player's speed (the tracker's speed channel
is used; a ball flying past a player is not contact). Each episode is classified by its length
and by where the ball went next:

| episode | definition | evidence |
|---|---|---|
| controlled | contact lasting at least 2 s; or at least 0.6 s, whatever follows | strong, for that side |
| pass | shorter contact whose next contact is a team-mate within 4 s, ball not lost for more than 1 s in between | strong, for that side, over the contact and the flight |
| touch | contact shorter than 0.6 s whose next contact is an opponent | none |
| unresolved | contact shorter than 0.6 s followed by a lost ball or nothing for 4 s | none |

A contact must last at least 2 frames (80 ms) to count at all.

**The carry rule.** A deflection, block or header leaves the player at well over 3 m/s and so
crosses the 3 m contact disc quickly (at 5 m/s, in 0.6 s). A ball that stays inside the disc *and
moves with the player* longer than that was received or carried: that is control, even if the
next contact is an opponent's (a tackle, a misplaced pass) or never comes. Without such a rule a
player who carried the ball for 1.7 s and was then tackled contributed nothing, and over the full
match 85 touch and 57 unresolved episodes of at least a second were discarded that way. v30 set
the threshold at the physical bound of one second. The final version (v31) uses 0.6 s and a
2-frame minimum contact (v30: 3 frames); these two values were chosen on ten labelled *other*
matches by a rule fixed in advance (section 3), never on this match.

Touches and unresolved contacts carry no weight because the challenge says a simple touch
without control does not change possession: they are neutral evidence, not evidence for the
other side. A down-weighted per-frame contact term (weight 0.5 in log-odds, Gaussian control
radius 1.5 m, floor weight 0.25 per side so that a loose ball 3 m from everyone says nothing) is
added so that sustained but untidy contact still counts; inside a touch episode that term is
muted to 0.1 and inside an unresolved one to 0.3, so a touch that went straight to the opponent
can never accumulate into a change of possession.

Two further sources name the side at a restart. A kick out of a stoppage (ball leaving a
player's feet after two seconds of high dead-ball probability) credits the kicker's side for
0.8 s, because every restart begins a possession for the restarting side. The Smart Tagging
ATTACKING and DEFENSIVE TRANSITION tags credit the side that gained the ball for 3 s after the
tag starts, at 0.5 nats per frame.

### 1.2 Goalkeeper team correction

The tracker keeps one goalkeeper slot per team, and the two keepers sometimes swap slots: the
away keeper slot is inside the *home* penalty area on 9% of its first-half frames and 13% of its
second-half frames, which only the home keeper can be. A keeper holding the ball is then read as
the other side in contact with it, which produced whole false possessions.

`possession/keepers.py` applies a football rule before any feature is computed: **a goalkeeper
inside a penalty area belongs to the side defending that goal.** Each team's keeper slot is the
slot most often inside either penalty area; which goal each team defends in each half comes from
the side of that team's mean x in the tracking. Every frame on which a keeper slot sits in the
penalty area of the goal the other team defends is moved to an extra slot of the defending team.
Over the match this moves 15,945 frames of the away keeper slot and 1,501 of the home slot.
Nothing else in the tracking is touched, and no label is involved.

### 1.3 Dead ball

The ground truth marks a frame `D` wherever the corrected tracking says the ball is not in play.
Several cues feed the dead-ball probability of a frame:

- **Out of bounds.** Ball tracked more than 0.3 m outside the pitch: `P(D) = 0.9`.
- **Nobody is running.** The share of tracked players above 2 m/s is bimodal over the match:
  37% of graded frames sit in the lowest bin. Measured on frames that are dead by construction
  (the 3 s before each tagged set piece) versus frames inside tagged build-up and progression
  phases, a threshold of 0.05 separates 70% of dead frames from 84% of alive frames. A logistic
  ramp around that value maps the share to `P(D)` between 0.12 and 0.60.
- **Ball at rest** for about two seconds: `P(D) >= 0.6`.
- **Ignored ball.** Slow (under 3 m/s) and more than 3 m from every player for two seconds:
  `P(D) >= 0.6`. In play a loose ball is chased; a ball nobody goes to is waiting for a restart.
- **Ball walked along a line.** Tracked within 1 m of a touchline or goal line, slow, at a
  player's feet or in his hands: `P(D) >= 0.55`. That is a throw-in or goal kick being set up.
- **Restart hold.** Once the ball is tracked out, or vanishes within 2 m of a line while
  travelling towards it, the ball stays dead (`P(D) >= 0.5`) until the restart. The restart is
  a kick from a player's feet with the ball in bounds, and only if play follows it: a ball
  kicked back towards the spot and then left alone within 4 s was a retrieval, not a restart.
  While the ball is lost, the players clearly running again also ends the hold. A hold never
  ends within 3 s of the ball going out, never lasts beyond 25 s, and never runs past either
  published kick-off frame.

Smart Tagging adds hard windows. Kick-off tags are 5 s windows starting about 3.5 s before the
kick; every other set-piece tag is a 35 s window with the restart somewhere inside it, so the
restart is taken as the first kick from a player's feet in the window, the 15 s before it are
dead, and the 0.8 s after it favour the tagged restart side. A GOALS tag can start minutes
before the goal, since it covers the move that produced it, so the goal itself is located as
the start of the longest quiet stretch before the kick-off tag (ball lost or slow, nobody
running: the celebration and the walk back), and the ball is dead from there until the
kick-off is taken. No tag window is allowed to force a dead ball on a frame that visibly shows
play (tracked ball moving faster than 3 m/s with more than a tenth of the players running). The
two published kick-off frames (11,185 and 113,742) mark the ball in play. The second half is
graded from frame 110,808 but kicks off at 113,742, so those frames are written as `D`.

### 1.4 When the ball is lost

The ball is missing on 41% of graded frames, in long stretches (median 11 s). Three quarters of
those frames have nobody running and are stoppages; the running-share cue handles them. For the
rest the chain carries its last state forward, nudged by two weak priors: the lateral spread of
each side (the side with the ball plays wider; a single logistic weight fitted to pseudo-labels
produced by the tracking on frames where the ball is unambiguously at one side's feet, 84%
agreement on 29,234 such frames) and the Smart Tagging phase team (0.05 nats per frame). Both are
deliberately small so that they can only decide a long, otherwise silent stretch.

### 1.5 Decoding

Entering or leaving `D` costs 25 nats. A change of side costs 15 nats, **except at the start of
a control episode (controlled or pass) of the side gaining the ball, where it is free** (within
±0.2 s of the onset, since the tracked ball trails the video by about three frames).

This gating is the challenge's definition of a change of possession written into the decoder:
the side changes when the other side gains control, and nowhere else comes cheap. With one fixed
cost everywhere, a short possession inside the other side's play needs two full switches, so it
needs twice the evidence of a real hand-over and is overruled however clean the control was.
Gated, it needs only the evidence of its own control episodes. Elsewhere a change of side still
needs roughly 0.3 s of clean controlled contact or 1.5 s of untidy contact (strong episode
evidence is worth about 2 nats per frame); a stoppage needs about one second of dead-ball
evidence.

Two post-passes follow the decoder. Dead-ball spells shorter than 2 s are folded into the spell
that follows, because a stoppage cannot be that short. Then every dead-ball onset is moved back
to the moment the players began to stop: the decoder places the boundary where the balance of
evidence tips, which for a whistle is a second or two late, while the ball is dead from the
whistle. The onset becomes the last frame at which the share of players running was still at its
recent peak, provided the share then fell by at least 0.15 and no frame crossed had a player on
the ball.

The whole pipeline runs in about 15 s on a laptop.

## 2. Assets used, and dropped

- **Tactical ATD**: the only source of positional evidence. Used in full, with the goalkeeper
  team correction of section 1.2.
- **Smart Tagging**: set pieces and goals as hard dead-ball windows, phase teams as a weak prior.
  Phases are not possessions: 36 instances carry no team, and every throw-in is one of them.
- **Homography**: not used by the pipeline (the tracking is already in pitch coordinates); used
  only by the clip renderer to draw the tracking over the video for inspection.
- **Broadcast ATD**: evaluated and dropped. By matching player positions team against team we
  found the broadcast runs at a constant offset from the tactical video within each half
  (+4014 frames in the first half, +26920 in the second, with a few frames of jitter) and
  aligned 39% of its frames to within 1 m. The broadcast ball, however, is mostly visible when
  the tactical ball is too: used as a gap filler it covers 1.8% of graded frames and changed no
  check for the better, so the option (`possession/broadcast.py`) ships switched off.
- **Video** (either feed): not used for labels. No GPU was available, and the central difficulty
  is the semantics of control, not detection. We watched it, through the clip renderer, to debug
  the pipeline; that is how the goalkeeper slot swaps were found.

## 3. Training data and use of the public window

None from outside the provided game and none labelled by a person. The single fitted number is
the logistic slope of the team-shape prior, fitted to pseudo-labels that the tracking itself
produces (frames where one side's player is within 1.5 m of the ball and the other side's
nearest player is more than 3 m away).

The public 5-minute ground truth was used to confirm frame alignment and file format, and to
debug: to find *which mechanism* produced a wrong label (for example the discarded one-second
carries and the swapped keeper slots), never to set a number. We want to be explicit about the
history. Earlier development versions were scored against the public window as they were built.
For the final submission every constant has a stated reason that does not depend on this match's
labels: physical quantities (contact radius, penalty-area geometry), football rules (a touch is
neutral evidence; a stoppage lasts at least 2 s), label-free distributions of the full-match
tracking (the running-share threshold), or, for the carry time and the minimum contact length
only, a choice on ten labelled other matches (below). The three changes before that (sections 1.1, 1.2 and
1.5) were judged on label-free full-match checks (section 5), and kept even where a public-window
number moved the wrong way. The repository's own tools no longer read the public labels by
default: `possession/diagnose.py` no longer prints a public-window segment table, and
`possession/render_clip.py` draws a truth row only if a file is passed with `--gt`.

**Validation on other matches, and the last two settings.** Because the competition labels are
private and the public window cannot tell versions apart, we measured the decoder on ten *other*
matches with public labels. v30 was only measured on them. For the final version (v31), two
existing settings were then chosen on them, by a rule fixed and committed before any result was
read (below).

- Metrica's three public sample matches (https://github.com/metrica-sports/sample-data; games 1
  and 2 added 2020-04-01, game 3 added 2021-04-15, anonymised; none is the competition match).
  Labels are derived from Metrica's human event data with the competition definition
  (`possession/sample_game.py`): the owner changes only at an event in which a player plays the
  ball (pass, carry, recovery, shot, set piece, ball lost), never at a challenge; the ball is dead
  from a ball-out, goal, offside, hand ball or a foul that a restart follows, until the restart.
- The DFL/Sportec open dataset: seven 2022/23 Bundesliga and 2. Bundesliga matches (Bassek et
  al. 2025, *Scientific Data*, doi:10.1038/s41597-025-04505-y, CC BY 4.0, released with the
  authorisation of the DFL). Their corrected tracking carries a per-frame `BallPossession` and
  `BallStatus`; the label is `D` where the status is dead, else the owning side
  (`possession/dfl.py`). That is the same kind of source as the competition ground truth.

`possession/validate.py` runs the decoder exactly as `run.py` does on each match and scores it with
the official `evaluation/possession_metrics.py`. Smart Tagging does not exist for these matches,
and their tracking has fixed player identities, so the tagging terms and the keeper fix are off
there. Because their ball is tracked far more often (Metrica 59-68% of graded frames, DFL 100%),
each match is also scored with the ball removed in random 0.4-5 s gaps until it is visible on 59%
of graded frames, the level of the competition match (two random seeds). Results are in
section 5. The runs used only public data, as PBS jobs on a university cluster
(`hpc/dfl_convert.pbs`, `hpc/validate.pbs`, `hpc/grid.pbs`); the competition data stayed on our
own machine.

**How the two settings were chosen (v31).** `possession/grid.py` decodes every combination of
seven existing v30 settings (change-of-side cost 15/10/6 nats, gate window 0.2/0.5/1.0 s, control
after 2/1.25/0.75 s, carry time 1.0/0.6 s, contact radius 1.5/1.25/2.0 m, minimum contact 3/2
frames, dead-ball cost 25/18 nats; 648 combinations, no new logic) on the ten matches in the three
ball conditions. The rule, committed to the repository before the jobs ran (`calib/grid_rule.md`,
applied by `possession/grid_select.py`):

1. Score a setting on a match by its spellF1 averaged over the three ball conditions.
2. Pick, on a set of matches, among the settings within 0.003 of the best mean, the one that
   changes the fewest v30 values.
3. Estimate the gain honestly by leaving each match out in turn: pick on the other nine, score
   on the one left out.
4. Adopt a change only if that held-out gain averages at least +0.010, is positive on at least
   8 of the 10 matches, and is positive on average for the Metrica and the DFL matches separately;
   then only if the competition match still passes the label-free checks of section 5.

Every one of the ten held-out picks was the same: carry time 0.6 s and minimum contact 2 frames,
all else as in v30. The held-out gain was +0.020 on average (Metrica +0.023, DFL +0.019) and
positive on all ten matches (smallest +0.011). The label-free checks passed (section 5). The
public window was not scored for this decision, before or after it.

## 4. Possession rules implemented

- Possession changes only through a controlled contact or a completed pass by the other side;
  the decoder makes a change of side cheap only at such a moment.
- A contact of 0.6 s or more, moving with the player, is control.
- A touch, clearance or deflection that does not lead to a team-mate does not change possession.
- A loose or contested ball keeps the last side in possession.
- A goalkeeper in his own penalty area belongs to the side defending it.
- The ball is dead when it is out, at rest with nobody running, or between going out and the
  restart kick; every restart, including throw-ins, begins a new possession for the restarting
  side as soon as that side controls the ball.
- A goal is dead from the goal until the kick-off.

## 5. Results

Label-free checks on the full match, for the previously submitted version (v27), v30, and the
final version (v31):

| check | v27 | v30 | final (v31) |
|---|---|---|---|
| possessions (mean length) | 209 (15.9 s) | 242 (13.8 s) | 284 (11.8 s) |
| possessions shorter than 2 s | | 13% | 16% |
| graded frames dead | 47.7% | 47.5% | 47.5% |
| dead inside tagged build-up / progression / transition phases | 9.2% | 8.7% | 8.7% |
| dead inside the 3 s before a tagged set piece | 94.5% | 94.6% | 94.6% |
| same side as a tagged phase, where we say the ball is alive | 84.2% | 83.3% | 82.2% |
| team-shape pseudo-label agreement | 81.8% | 83.9% | 83.9% |

v31 changes only the split between the sides inside play: its dead-ball spells are those of v30.

Home share of alive time is 63%. Smart Tagging phases are not possessions (a phase can outlast
several changes of side), so the side agreement is a sanity bound rather than a target; the
official baseline scores 67% on it.

**On ten other matches with public labels** (section 3; official metric code; spellF1). v27 and
v30 were not tuned on these matches. v31's two settings were chosen on them, so its columns are
in-sample; its honest, held-out gain over v30 is +0.020 (section 3).

| match (true possessions) | v27, ball as recorded | v30, as recorded | v31, as recorded | v27, ball at 59% (mean of 2 seeds) | v30, at 59% | v31, at 59% |
|---|---|---|---|---|---|---|
| Metrica 1 (384) | 0.491 | 0.546 | **0.580** | 0.470 | 0.525 | **0.557** |
| Metrica 2 (360) | 0.533 | 0.592 | **0.612** | 0.533 | 0.592 | **0.612** |
| Metrica 3 (406) | 0.524 | 0.555 | **0.570** | 0.487 | 0.524 | **0.540** |
| DFL J03WMX (511) | 0.435 | 0.487 | **0.517** | 0.333 | 0.358 | **0.378** |
| DFL J03WN1 (500) | 0.419 | 0.469 | **0.493** | 0.282 | 0.321 | **0.337** |
| DFL J03WOH (402) | 0.463 | 0.519 | **0.552** | 0.354 | 0.392 | **0.408** |
| DFL J03WOY (456) | 0.452 | 0.508 | **0.533** | 0.355 | 0.371 | **0.395** |
| DFL J03WPY (421) | 0.479 | 0.535 | **0.546** | 0.364 | 0.400 | **0.414** |
| DFL J03WQQ (401) | 0.544 | 0.596 | **0.607** | 0.435 | 0.470 | **0.480** |
| DFL J03WR9 (370) | 0.459 | 0.534 | **0.558** | 0.390 | 0.430 | **0.451** |
| **mean** | 0.480 | 0.534 | **0.557** | 0.400 | 0.438 | **0.457** |

v30 beats v27, and v31 beats v30, on all ten matches in all three ball conditions (30 of 30 each;
v31's smallest gains, at 59% visibility, are +0.0001 and +0.0004). The removal of the ball is
random, so the 59% columns are a pessimistic stand-in for the competition match's real gaps,
which fall mostly in stoppages. With the ball as recorded, v31 finds 58-74% as many possessions as
the labels contain (v30 51-66%, v27 42-57%), for example 294 of 511 on J03WMX: short possessions
remain the main weakness (section 6).

For the record, a variant of v27 with two settings re-chosen on the three Metrica matches only
(change-of-side cost 6 nats, control after 0.75 s; branch `calib-sample-game3`) reached about the
level of v30 on the Metrica matches (0.561 against 0.565, chosen in-sample there) but less on the
DFL matches with the ball as recorded (0.497 against 0.521). We did not use it. The v31 grid
includes both of its changes (6 nats, 0.75 s) applied to v30, and the rule did not pick them.

On the public window, as a correctness check only (v31 was not scored on it; see section 3):

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| `baseline_smooth50` (official floor) | 0.6207 | 0.3256 | 26 |
| ours, v27 | 0.9010 | 0.5143 | 11 |
| ours, v30 | 0.9089 | 0.5286 | 11 |

With only 17 possessions in the window the segment-bootstrap intervals of the two versions of
ours overlap almost entirely (about 0.31 to 0.72 for both), so the window cannot tell them
apart; EVALUATION.md shows why it should not be asked to.

## 6. Limitations

- Possessions shorter than about 2 s are mostly missed. In the public window the short
  possessions we miss have no control episode in the tracking at all: a single brief touch, or
  a stretch where the ball is not tracked. A second ball observer would be needed for them.
- When the tracked ball sits 3 m or more from every player for several seconds, usually a
  tracking error, nothing can be said and the chain holds its last state.
- The first seconds of a possession that begins with an untidy recovery are often credited late.
- Dead-ball boundaries are placed to within about a second, not to the frame.
- Goalkeepers are frequently absent from the tracking, so a keeper holding the ball is often
  seen only as a slow ball with nobody near it. When a keeper *is* tracked but in the wrong
  slot, section 1.2 corrects it only inside the penalty areas.
- Metrica's ball track trails the video by about three frames, measured label-free from where
  the travelling ball meets a receiver. The ground truth runs on the tracker's clock, so no
  correction is applied for scoring; the clip renderer in `possession/render_clip.py` draws the
  ball three frames ahead purely for the eye.
- The broadcast alignment works but is unused; with more time it would become a second ball
  observer and a check on the tactical ball, which disagrees with it by more than 5 m on one
  frame in ten.
- The out-of-sample validation (section 5) measures the decoder without Smart Tagging and
  without the keeper fix, on tracking with fixed player identities, and on labels that are not
  Metrica's corrected tracking (human events for the Metrica matches, the DFL's own possession
  field for the DFL matches). It shows the direction and size of a change, not the competition
  score. v31's two settings were chosen against those labels, whose definition of possession is
  close to, but not exactly, the competition's; the rule asked for a gain on both kinds of labels
  for that reason.

## 7. Reproduction

```bash
pip install -r requirements-possession.txt        # Python 3.11 or newer
python make_submission.py --team "Más Bueno"      # quotes needed: the name has a space
```

Data in `./data` as described in DATA.md (or `MTAH_DATA_DIR`). The run parses the tracking
once into `cache/` and is deterministic: the same data gives a byte-identical file (SHA-256
`f34036f02e246b42f314e1de70d7d6b1cc15aca906d65a686d6963a32604de2a`; checked with the pinned
numpy 2.5.0 / pandas 3.0.3 and with numpy 2.3.5 / pandas 3.0.0).

To repeat the validation of section 5 (optional; `calib/validate_v30.jsonl` and
`calib/validate_v27_v29.jsonl` hold the runs of v30 and the earlier versions, `calib/grid/` the
grid of section 3):

```bash
# Metrica sample matches 1-3: the files listed in possession/sample_game.py, two folders.
python -m possession.validate --sample <games12> --game 1
python -m possession.validate --sample <game3>   --game 3 --ball-visible 0.59 --seed 0
# DFL: the 7 match-information and raw-position XML files from figshare article 28196177,
# converted once with kloppy 3.19.1 (pip install kloppy==3.19.1), then scored the same way.
python -m possession.dfl <dfl-xml-folder> <npz-folder>
python -m possession.validate --sample <npz-folder>/J03WOH.npz --ball-visible 0.59 --seed 1
# The settings grid of section 3 (one match and ball condition per run; Linux, uses fork) and
# the fixed selection rule over all 30 result files:
python -m possession.grid --sample <npz-folder>/J03WOH.npz --jobs 32 --out calib/grid/J03WOH__rec.jsonl
python -m possession.grid_select calib/grid/*.jsonl
```

## 8. Tools

The code was written with the help of an LLM coding assistant (Claude Code by Anthropic) running
on the team's own machine. The pipeline itself contains no LLM and no call to any service.
