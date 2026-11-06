"""Align the broadcast tracking to the tactical timeline and lift its ball track across.

The broadcast ATD is on its own frame counter, but during live coverage it runs at a constant
offset from the tactical video within each half (the two feeds were cut from the same clock).
Replays and close-ups break the match. For every broadcast frame the offset is chosen by
comparing player positions, same team against same team, with the tactical frame it would map
to; a frame is accepted when the mean nearest-neighbour distance is under 1 m.

    from possession.broadcast import broadcast_ball
    ball_b, n_aligned = broadcast_ball()      # (n_tactical, 2) metres, NaN where nothing maps
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from possession.atd import load_tracking, CACHE_DIR, HOME  # noqa: E402

# Coarse offsets found by matching team-shape signatures; refined per frame below.
OFFSET_HINTS = (4014, 26920)
REFINE = 4            # frames either side of a hint
MAX_DIST = 1.0        # m, mean nearest-neighbour distance for an accepted alignment
MIN_PLAYERS = 6


def _nn_mean(Q, R):
    if len(Q) == 0 or len(R) == 0:
        return np.nan, 0
    d = np.hypot(Q[:, None, 0] - R[None, :, 0], Q[:, None, 1] - R[None, :, 1]).min(axis=1)
    return d.sum(), len(d)


def align(force=False):
    """(offset, distance) per broadcast frame; offset -1 where no alignment was accepted."""
    cache = os.path.join(CACHE_DIR, "broadcast_align.npz")
    if os.path.isfile(cache) and not force:
        z = np.load(cache)
        return z["offset"], z["dist"]
    T, B = load_tracking("tactical"), load_tracking("broadcast")
    PT, PB = T["players"], B["players"]
    hT, hB = T["team"] == HOME, B["team"] == HOME
    nT, nB = len(PT), len(PB)
    tracked = (~np.isnan(PB[:, :, 0])).sum(axis=1)
    cands = [o + k for o in OFFSET_HINTS for k in range(-REFINE, REFINE + 1)]
    offset = np.full(nB, -1, dtype=np.int64)
    dist = np.full(nB, np.nan)
    for b in range(nB):
        if tracked[b] < MIN_PLAYERS:
            continue
        best, best_o = np.inf, -1
        for o in cands:
            t = b + o
            if not (0 <= t < nT):
                continue
            tot, cnt = 0.0, 0
            for mb, mt in ((hB, hT), (~hB, ~hT)):
                Q = PB[b][mb]
                Q = Q[~np.isnan(Q[:, 0])]
                R = PT[t][mt]
                R = R[~np.isnan(R[:, 0])]
                s, c = _nn_mean(Q, R)
                if c:
                    tot += s
                    cnt += c
            if cnt >= MIN_PLAYERS and tot / cnt < best:
                best, best_o = tot / cnt, o
        if best_o >= 0 and best < MAX_DIST:
            offset[b], dist[b] = best_o, best
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, offset=offset, dist=dist)
    return offset, dist


def broadcast_ball(force=False):
    """Broadcast ball position on the tactical timeline, metres, NaN where unavailable."""
    offset, dist = align(force)
    B = load_tracking("broadcast")
    nT = GAMES[GAME]["n_video_frames"]
    ball = np.full((nT, 2), np.nan)
    count = np.zeros(nT, dtype=np.int64)
    ok = (offset >= 0) & ~np.isnan(B["ball"][:, 0])
    for b in np.flatnonzero(ok):
        t = b + offset[b]
        if 0 <= t < nT:
            if count[t] == 0:
                ball[t] = B["ball"][b]
            else:
                ball[t] = (ball[t] * count[t] + B["ball"][b]) / (count[t] + 1)
            count[t] += 1
    return ball, int((offset >= 0).sum())


if __name__ == "__main__":
    import time
    t0 = time.time()
    offset, dist = align(force="--force" in sys.argv)
    ok = offset >= 0
    print("aligned {:.1%} of broadcast frames in {:.0f}s; distance p50 {:.2f}".format(
        ok.mean(), time.time() - t0, float(np.nanmedian(dist[ok]))))
    ball_b, n_aligned = broadcast_ball()
    T = load_tracking("tactical")
    from possession.run import graded_mask
    gr = graded_mask(len(T["ball"]))
    tv = ~np.isnan(T["ball"][:, 0])
    bv = ~np.isnan(ball_b[:, 0])
    both = gr & tv & bv
    d = np.hypot(*(T["ball"][both] - ball_b[both]).T)
    print("graded frames: tactical ball {:.1%}, broadcast ball {:.1%}, union {:.1%}; "
          "agreement when both (m) p50 {:.2f} p90 {:.2f}".format(
              tv[gr].mean(), bv[gr].mean(), (tv | bv)[gr].mean(), *np.percentile(d, [50, 90])))
