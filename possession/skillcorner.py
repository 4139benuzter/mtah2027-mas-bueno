"""SkillCorner open data as a further set of labelled OTHER matches (validation only).

SkillCorner Open Data, https://github.com/SkillCorner/opendata (MIT licence; credit SkillCorner),
released with PySport: 20 matches of the Australian A-League 2024/25, broadcast tracking at 10 fps
(players outside the camera view are extrapolated; the ball is detected on about 55% of in-play
frames, close to the competition match's 59%), plus "dynamic events" with every player
possession and every game interruption. None of them is the competition match.

Labels follow the competition definition (OVERVIEW.md):
  * the owner changes only when the other side starts a controlled player possession
    (SkillCorner `player_possession` events); a loose or contested ball keeps the last owner;
  * the ball is dead (D) from the moment it leaves the pitch (first detected ball position past a
    line after the event that the next restart follows) -- or from the end of that event for a
    free kick or penalty -- until the restart is taken; the restart moment is the last frame
    before the receiving possession at which the ball was slow (< 2 m/s), else 1 s (throw-in) or
    1.5 s before the reception;
  * from the restart to the reception the restarting side owns the ball;
  * frames before the first possession or after the last event of each half are X;
  * a stretch of more than 5 s with no event of any kind, not already dead, is X (unknown). This
    rule was added after the first converted match showed long event-free stretches (up to
    148 s, ball visible on 0-10% of their frames) labelled as alive play; the 5 s threshold is
    the knee of the event-gap distribution, chosen from the events, not from any score.
The 10 fps tracking is resampled to 25 fps (linear between neighbouring samples, never across a
gap); labels take the nearest 10 fps sample.

    python -m possession.skillcorner <folder with the downloaded files> <out folder> [ids...]
"""

import glob
import json
import os
import re
import sys

import numpy as np

FPS_IN = 10
FPS_OUT = 25
SLOW = 2.0          # m/s, a ball at most this fast is at rest or held before a restart
SEARCH_S = 6.0      # how far to look for the ball leaving the pitch, or for the restart kick
MAX_DEAD_S = 120.0  # an interruption whose restart comes later than this is not linked
NO_BALL_OUT = ("free_kick", "penalty")
GAP_S = 5.0         # an event-free stretch longer than this is not observed live play (see below)


def _team(tid, home_id):
    return "H" if tid == home_id else "A"


def _other(label):
    return "A" if label == "H" else "H"


def _taker(event, home_id):
    """Side that takes the restart that `event` (the first possession after it) follows."""
    kind = event["game_interruption_before"]
    own = _team(event["team_id"], home_id)
    if kind.startswith("goal"):
        return own if kind.endswith("against") else _other(own)   # the conceding side kicks off
    return own if kind.endswith("_for") else _other(own)


def _ball_speed(ball):
    v = np.full(len(ball), np.nan)
    d = ball[1:] - ball[:-1]
    v[:-1] = np.hypot(d[:, 0], d[:, 1]) * FPS_IN
    return v


def build_labels(ev, n, ball, length, width, home_id):
    """(n,) labels at 10 fps from the dynamic events and the detected ball (corner origin, m)."""
    labels = np.full(n, "X", dtype="<U1")
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
    for period in sorted(pp["period"].dropna().unique()):
        part = pp[pp["period"] == period]
        g0 = int(part["frame_start"].min())
        g1 = int(ev.loc[ev["period"] == period, "frame_end"].max())
        starts = part["frame_start"].to_numpy(int)
        owners = np.array([_team(t, home_id) for t in part["team_id"]])
        idx = np.searchsorted(starts, np.arange(g0, g1 + 1), side="right") - 1
        labels[g0:g1 + 1] = owners[np.maximum(idx, 0)]

    speed = _ball_speed(ball)
    out = (ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width)
    afters = ev[ev["game_interruption_after"].notna()].sort_values("frame_end")
    befores = ev[ev["game_interruption_before"].notna()].sort_values("frame_start")
    search, max_dead = int(SEARCH_S * FPS_IN), int(MAX_DEAD_S * FPS_IN)
    windows = 0
    for _, nb in befores.iterrows():
        start_b = int(nb["frame_start"])
        prior = afters[(afters["frame_end"] < start_b) & (afters["period"] == nb["period"])]
        if prior.empty or start_b - int(prior["frame_end"].iloc[-1]) > max_dead:
            continue
        a = prior.iloc[-1]                        # the last event before this restart
        dead = int(a["frame_end"])
        if not str(a["game_interruption_after"]).startswith(NO_BALL_OUT):
            hit = np.flatnonzero(out[dead:min(start_b, dead + search)])
            if len(hit):
                dead += int(hit[0])
        kick = None
        for f in range(start_b - 1, max(dead, start_b - search), -1):
            if np.isfinite(speed[f]) and speed[f] < SLOW:
                kick = f + 1
                break
        if kick is None:
            back = 1.0 if "throw_in" in nb["game_interruption_before"] else 1.5
            kick = start_b - int(back * FPS_IN)
        kick = min(max(kick, dead + FPS_IN), start_b)     # at least 1 s dead
        graded = labels[dead:start_b] != "X"
        labels[dead:kick][graded[:kick - dead]] = "D"
        labels[kick:start_b][graded[kick - dead:]] = _taker(nb, home_id)
        windows += 1

    # Unknown stretches. Live play produces an event every second or so (median gap 1 s); a
    # stretch of more than GAP_S with no event of any kind is a stoppage that no interruption
    # event marks (injury, substitution, review) or play the broadcast did not show. Its owner
    # is unknown, so it becomes X unless it is already dead. The official scorer never scores X
    # and cuts possessions at it, for the labels and every prediction alike.
    gap = int(GAP_S * FPS_IN)
    for _, part in ev.groupby("period"):
        iv = part[["frame_start", "frame_end"]].sort_values("frame_start").to_numpy(int)
        reach = iv[0, 1]
        for s, e in iv[1:]:
            if s - reach > gap:
                span = labels[reach + 1:s]
                span[span != "D"] = "X"
            reach = max(reach, e)
    return labels, windows


def _resample(a, n_out):
    """10 fps -> 25 fps, linear between neighbouring samples, NaN across a gap."""
    t = np.arange(n_out) * FPS_IN / FPS_OUT
    lo = np.minimum(np.floor(t).astype(int), len(a) - 1)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = (t - lo).reshape((-1,) + (1,) * (a.ndim - 1))
    mixed = (1 - w) * a[lo] + w * a[hi]
    return np.where(w == 0, a[lo], mixed)


def convert(folder, match_id, out_path):
    import pandas as pd
    from possession.sample_game import _player_speed

    meta = json.load(open(os.path.join(folder, "{}_match.json".format(match_id)), encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    players = meta["players"]
    col = {p["id"]: i for i, p in enumerate(players)}
    team = np.array([0 if p["team_id"] == home_id else 1 for p in players], dtype=np.int8)

    lines = open(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)),
                 encoding="utf-8").read().splitlines()
    n = max(json.loads(line)["frame"] for line in lines[-50:]) + 1
    ball = np.full((n, 2), np.nan)
    xy = np.full((n, len(players), 2), np.nan)
    for line in lines:
        f = json.loads(line)
        if f.get("period") is None:
            continue
        i = f["frame"]
        b = f.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:
            ball[i] = (b["x"] + length / 2, b["y"] + width / 2)
        for p in f.get("player_data") or []:
            if p["player_id"] in col and p.get("x") is not None:
                xy[i, col[p["player_id"]]] = (p["x"] + length / 2, p["y"] + width / 2)

    ev = pd.read_csv(os.path.join(folder, "{}_dynamic_events.csv".format(match_id)),
                     low_memory=False)
    labels10, windows = build_labels(ev, n, ball, length, width, home_id)

    n_out = int(n * FPS_OUT / FPS_IN)
    t = np.minimum(np.round(np.arange(n_out) * FPS_IN / FPS_OUT).astype(int), n - 1)
    labels = labels10[t]
    ball25, xy25 = _resample(ball, n_out), _resample(xy, n_out)
    graded = labels != "X"
    edges = np.flatnonzero(np.diff(np.concatenate(([0], graded.astype(np.int8), [0]))))
    halves = [(k + 1, int(s), int(e) - 1) for k, (s, e) in enumerate(zip(edges[::2], edges[1::2]))]
    np.savez_compressed(out_path, ball=ball25, players=xy25, team=team,
                        speed=_player_speed(xy25, FPS_OUT), labels=labels,
                        kickoffs=np.array([h[1] for h in halves]), halves=np.array(halves),
                        pitch=np.array([length, width]))
    return out_path, windows


if __name__ == "__main__":
    from possession.dfl import load_npz
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ids = sorted({re.match(r"(\d+)_tracking", os.path.basename(p)).group(1)
                  for p in glob.glob(os.path.join(src, "*_tracking_extrapolated.jsonl"))})
    only = sys.argv[3:]
    for match_id in ids:
        if only and match_id not in only:
            continue
        path, windows = convert(src, match_id, os.path.join(out, "SC{}.npz".format(match_id)))
        trk, labels, info = load_npz(path)
        g = labels != "X"
        lab = labels[g]
        spells = int(np.sum((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"]))) + int(lab[0] in "HA")
        print("SC{}: graded {} D {:.1%} H {:.1%} possessions {} restarts {} ball visible {:.1%} "
              "halves {}".format(match_id, g.sum(), (lab == "D").mean(), (lab == "H").mean(), spells,
                                 windows, (~np.isnan(trk["ball"][g, 0])).mean(), info["halves"]),
              flush=True)
