# Frame-level ball possession from automatic tracking

**More Than A Hack 2027, Phase 1. Team: Más Bueno (Aaron Binoj Amacadu, Karthik, James).**

## 1. Approach

We treat possession as a hidden state with three values, home (`H`), away (`A`) and dead ball
(`D`), that changes rarely, and decode the most probable state sequence over the 199,319 frames
with a Viterbi pass. Every piece of evidence is expressed as a per-frame log-probability of the
three states; a fixed cost is charged for every change of state. The chain therefore only
changes its answer when the evidence for the new state, accumulated over consecutive frames,
outweighs the cost. This is what turns noisy per-frame cues into whole possessions, which is what
the ranking metric scores.

The evidence comes from the provided automatic tracking (ATD) and the Smart Tagging, nothing
else. The design question throughout was not "which player is nearest to the ball" but "what does
the ground truth mean by *controlled* possession and by *dead*", and the answer is encoded in the
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
| controlled | contact lasting at least 2 s | strong, for that side |
| pass | shorter contact whose next contact is a team-mate within 4 s, ball not lost for more than 1 s in between | strong, for that side, over the contact and the flight |
| touch | shorter contact whose next contact is an opponent | none |
| unresolved | shorter contact followed by a lost ball or nothing for 4 s | none |

On the public window, episodes classed as controlled or pass agreed with the ground truth side
in about 90% of cases, touches in about two thirds (almost always because that side already had
the ball), and unresolved episodes in fewer than half, which is why the last two carry no
weight. A down-weighted per-frame contact term (weight 0.5 in log-odds, Gaussian control radius
1.5 m, floor weight 0.25 per side so that a loose ball 3 m from everyone says nothing) is added
so that sustained but untidy contact still counts; inside a touch episode that term is muted to
0.1 and inside an unresolved one to 0.3, so a touch that went straight to the opponent can never
accumulate into a change of possession.

Two further sources name the side at a restart. A kick out of a stoppage (ball leaving a
player's feet after two seconds of high dead-ball probability) credits the kicker's side for
0.8 s, because every restart begins a possession for the restarting side. The Smart Tagging
ATTACKING and DEFENSIVE TRANSITION tags credit the side that gained the ball for 3 s after the
tag starts, at 0.5 nats per frame.

### 1.2 Dead ball

The ground truth marks a frame `D` wherever the corrected tracking says the ball is not in play.
Four cues feed the dead-ball probability of a frame:

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
two published kick-off frames (11,185 and 113,742) mark the ball in play. Dead-ball spells shorter than 2 s are folded into the spell that follows,
because a stoppage cannot be that short. The second half is graded from frame 110,808 but kicks
off at 113,742, so those frames are written as `D`.

### 1.3 When the ball is lost

The ball is missing on 41% of graded frames, in long stretches (median 11 s). Three quarters of
those frames have nobody running and are stoppages; the running-share cue handles them. For the
rest the chain carries its last state forward, nudged by two weak priors: the lateral spread of
each side (the side with the ball plays wider; a single logistic weight fitted to pseudo-labels
produced by the tracking on frames where the ball is unambiguously at one side's feet, 82%
agreement on 28,820 such frames) and the Smart Tagging phase team (0.05 nats per frame). Both are
deliberately small so that they can only decide a long, otherwise silent stretch.

### 1.4 Decoding

Switch costs are fixed at 15 nats for a change of side and 25 nats for entering or leaving `D`.
With strong episode evidence worth about 2 nats per frame, a change of side needs roughly 0.3 s
of clean controlled contact or 1.5 s of untidy contact; a stoppage needs about one second of
dead-ball evidence.

Two post-passes follow the decoder. Dead-ball spells shorter than 2 s are folded into the spell
that follows. Then every dead-ball onset is moved back to the moment the players began to stop:
the decoder places the boundary where the balance of evidence tips, which for a whistle is a
second or two late, while the ground truth marks the whistle. The onset becomes the last frame
at which the share of players running was still at its recent peak, provided the share then
fell by at least 0.15 and no frame crossed had a player on the ball. On the public window this
took the two late stoppage onsets from 1.9 s and 2.2 s late to within two frames.

The whole pipeline runs in about 15 s on a laptop.

## 2. Assets used, and dropped

- **Tactical ATD**: the only source of positional evidence. Used in full.
- **Smart Tagging**: set pieces and goals as hard dead-ball windows, phase teams as a weak prior.
  Phases are not possessions: 36 instances carry no team, and every throw-in is one of them.
- **Homography**: not used. The tracking is already in pitch coordinates.
- **Broadcast ATD**: evaluated and dropped. By matching player positions team against team we
  found the broadcast runs at a constant offset from the tactical video within each half
  (+4014 frames in the first half, +26920 in the second, with a few frames of jitter) and
  aligned 39% of its frames to within 1 m. The broadcast ball, however, is mostly visible when
  the tactical ball is too: used as a gap filler it covers 1.8% of graded frames and changed no
  check for the better, so the option (`possession/broadcast.py`) ships switched off.
- **Video** (either feed): not used. No GPU was available, and the central difficulty is the
  semantics of control, not detection.

## 3. Training data

None from outside the provided game and none labelled by a person. The single fitted number is
the logistic slope of the team-shape prior, fitted to pseudo-labels that the tracking itself
produces (frames where one side's player is within 1.5 m of the ball and the other side's
nearest player is more than 3 m away). The public 5-minute ground truth was used only to confirm
alignment and to check that episode classes mean what we thought; no threshold was chosen on it.

## 4. Possession rules implemented

- Possession changes only through a controlled contact or a completed pass by the other side.
- A touch, clearance or deflection that does not lead to a team-mate does not change possession.
- A loose or contested ball keeps the last side in possession.
- The ball is dead when it is out, at rest with nobody running, or between going out and the
  restart kick; every restart, including throw-ins, begins a new possession for the restarting
  side as soon as that side controls the ball.
- A goal is dead from the goal until the kick-off.

## 5. Results on the public window

| submission | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| `baseline_smooth50` (official floor) | 0.6207 | 0.3256 | 26 |
| ours | 0.9010 | 0.5143 | 11 |

With only 17 possessions in the window, spellF1 differences below about 0.4 are inside the
bootstrap noise band, so the second column is encouraging rather than conclusive; the macroF1
gain, and in particular the dead-ball F1 of 0.93, is well outside the noise.

Across the full match our file has 209 possessions of mean 16 s, 48% of graded frames dead, and
a 63% home share of alive time. Three label-free checks: inside Metrica's own tagged build-up,
progression and transition phases we label 9% of frames dead; inside the 3 s before each tagged
set piece we label 94% dead; and where a tagged phase names a side and we say the ball is alive,
we name the same side 84% of the time (the baseline: 67%).

## 6. Limitations

- Possessions shorter than about 1.5 s are mostly missed; three of the seventeen in the public
  window are that short.
- When the tracked ball sits 3 m or more from every player for several seconds, usually a
  tracking error, nothing can be said and the chain holds its last state.
- The first seconds of a possession that begins with an untidy recovery are often credited late.
- Dead-ball boundaries are placed to within about a second, not to the frame.
- Goalkeepers are frequently absent from the tracking (a player is within 8 m of a goal line on
  under half of the graded frames), so a keeper holding the ball is seen only as a slow ball
  with nobody near it; the ground truth, derived from the same tracking family, treats such
  frames as dead, and so do we.
- Metrica's ball track trails the video by about three frames, measured label-free from where
  the travelling ball meets a receiver. The ground truth runs on the tracker's clock (our side
  changes are already early against it), so no correction is applied for scoring; the clip
  renderer in `possession/render_clip.py` draws the ball three frames ahead purely for the eye.
- Changes of side at an untidy recovery are often credited one to two seconds before the
  ground truth, which waits for clean control.
- The broadcast alignment works but is unused; with more time it would become a second ball
  observer and a check on the tactical ball, which disagrees with it by more than 5 m on one
  frame in ten.

## 7. Reproduction

```bash
pip install -r requirements-possession.txt
python make_submission.py --team Más Bueno
```

Data in `./data` as described in DATA.md (or `MTAH_DATA_DIR`). The run parses the tracking
once into `cache/` and is deterministic.

## 8. Tools

The code was written with the help of an LLM coding assistant (Claude Code by Anthropic) running
on the team's own machine. The pipeline itself contains no LLM and no call to any service.
