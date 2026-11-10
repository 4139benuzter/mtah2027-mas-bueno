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
| controlled | contact lasting at least 2 s; or at least 1 s, whatever follows | strong, for that side |
| pass | shorter contact whose next contact is a team-mate within 4 s, ball not lost for more than 1 s in between | strong, for that side, over the contact and the flight |
| touch | contact shorter than 1 s whose next contact is an opponent | none |
| unresolved | contact shorter than 1 s followed by a lost ball or nothing for 4 s | none |

**The one-second rule.** A deflection, block or header leaves the player at well over 3 m/s and
so crosses the 3 m contact disc in under a second. A ball that stays inside the disc *and moves
with the player* for a full second was received or carried: that is control, even if the next
contact is an opponent's (a tackle, a misplaced pass) or never comes. Without this rule a player
who carried the ball for 1.7 s and was then tackled contributed nothing, and over the full match
85 touch and 57 unresolved episodes of at least a second were discarded that way. The threshold
is physical, not fitted to labels.

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
For the final submission every constant has a stated reason that does not depend on the labels:
physical quantities (contact radius, the one-second carry, penalty-area geometry), football rules
(a touch is neutral evidence; a stoppage lasts at least 2 s), or label-free distributions of the
full-match tracking (the running-share threshold). The last three changes (sections 1.1, 1.2 and
1.5) were judged on label-free full-match checks (section 5), and kept even where a public-window
number moved the wrong way.

## 4. Possession rules implemented

- Possession changes only through a controlled contact or a completed pass by the other side;
  the decoder makes a change of side cheap only at such a moment.
- A contact of a second or more, moving with the player, is control.
- A touch, clearance or deflection that does not lead to a team-mate does not change possession.
- A loose or contested ball keeps the last side in possession.
- A goalkeeper in his own penalty area belongs to the side defending it.
- The ball is dead when it is out, at rest with nobody running, or between going out and the
  restart kick; every restart, including throw-ins, begins a new possession for the restarting
  side as soon as that side controls the ball.
- A goal is dead from the goal until the kick-off.

## 5. Results

Label-free checks on the full match, for the previously submitted version and the final one:

| check | previous | final |
|---|---|---|
| possessions (mean length) | 209 (15.9 s) | 242 (13.8 s) |
| graded frames dead | 47.7% | 47.5% |
| dead inside tagged build-up / progression / transition phases | 9.2% | 8.7% |
| dead inside the 3 s before a tagged set piece | 94.5% | 94.6% |
| same side as a tagged phase, where we say the ball is alive | 84.2% | 83.3% |
| team-shape pseudo-label agreement | 81.8% | 83.9% |

Home share of alive time is 63%. Smart Tagging phases are not possessions (a phase can outlast
several changes of side), so the side agreement is a sanity bound rather than a target; the
official baseline scores 67% on it.

On the public window, as a correctness check only:

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| `baseline_smooth50` (official floor) | 0.6207 | 0.3256 | 26 |
| ours, previous | 0.9010 | 0.5143 | 11 |
| ours, final | 0.9089 | 0.5286 | 11 |

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

## 7. Reproduction

```bash
pip install -r requirements-possession.txt
python make_submission.py --team "Más Bueno"
```

Data in `./data` as described in DATA.md (or `MTAH_DATA_DIR`). The run parses the tracking
once into `cache/` and is deterministic: the same data gives a byte-identical file.

## 8. Tools

The code was written with the help of an LLM coding assistant (Claude Code by Anthropic) running
on the team's own machine. The pipeline itself contains no LLM and no call to any service.
