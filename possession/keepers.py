"""Goalkeeper team correction: a keeper inside a penalty area belongs to the side defending it.

The tracker keeps one goalkeeper slot per team, and now and then the two keepers swap slots: the
away keeper slot turns up in the home penalty area (9% of its first-half frames, 13% of its
second-half frames), and the home slot occasionally in the away one. Each keeper wears his own
kit, so this is an identity error, not a team change, and it matters: a keeper holding the ball
at his feet is then read as the *other* team in contact with it.

The rule is football, not fitted to anything: only the defending goalkeeper stands in his own
penalty area as a keeper. Every frame on which a keeper slot sits inside the penalty area of the
goal the *other* team defends is moved to an extra slot of the defending team. Which goal each
team defends in each half comes from the tracking itself (the side of the team's mean x).
Nothing else in the tracking is touched.
"""

import numpy as np

from .atd import HOME, AWAY

BOX_DEPTH = 16.5     # m, penalty area depth from the goal line
BOX_HALF_WIDTH = 20.16  # m, half the penalty area width (40.32 m)


def _in_box(x, y, goal_x, length, width):
    depth = np.abs(x - goal_x)
    return (depth <= BOX_DEPTH) & (np.abs(y - width / 2) <= BOX_HALF_WIDTH) & \
        (x >= -2) & (x <= length + 2)


def keeper_slots(players, team, length, width):
    """Per team, the slot most often inside either penalty area."""
    x, y = players[:, :, 0], players[:, :, 1]
    tracked = ~np.isnan(x)
    in_any = (_in_box(x, y, 0.0, length, width) | _in_box(x, y, length, length, width)) & tracked
    share = in_any.sum(axis=0) / np.maximum(tracked.sum(axis=0), 1)
    return {side: int(np.flatnonzero(team == side)[np.argmax(share[team == side])])
            for side in (HOME, AWAY)}


def fix_keepers(trk, halves, length, width):
    """Return a copy of trk with mis-teamed keeper frames moved to two appended slots.

    halves: [(half, first_frame, last_frame)] in video frames, e.g. games.half_video_blocks.
    Adds trk["keeper_fix"] = {"slots": {...}, "moved": {HOME: frames, AWAY: frames}}.
    """
    players, speed, team = trk["players"], trk["speed"], trk["team"]
    n, n_slots, _ = players.shape
    slots = keeper_slots(players, team, length, width)
    extra_p = np.full((n, 2, 2), np.nan)
    extra_s = np.full((n, 2), np.nan)
    # appended slot 0 is an extra HOME keeper slot, slot 1 an extra AWAY one
    extra_team = np.array([HOME, AWAY], dtype=team.dtype)
    players = players.copy()
    speed = speed.copy()
    moved = {HOME: 0, AWAY: 0}
    for _, first, last in halves:
        sl = slice(first, last + 1)
        mean_x = {side: np.nanmean(players[sl][:, team == side, 0]) for side in (HOME, AWAY)}
        goal = {side: (0.0 if mean_x[side] < mean_x[1 - side] else length) for side in (HOME, AWAY)}
        for side in (HOME, AWAY):
            other = 1 - side
            s = slots[side]
            x, y = players[sl, s, 0], players[sl, s, 1]
            wrong = _in_box(x, y, goal[other], length, width) & ~np.isnan(x)
            idx = np.flatnonzero(wrong) + first
            extra_p[idx, other] = players[idx, s]
            extra_s[idx, other] = speed[idx, s]
            players[idx, s] = np.nan
            speed[idx, s] = np.nan
            moved[side] += len(idx)
    out = dict(trk)
    out["players"] = np.concatenate([players, extra_p], axis=1)
    out["speed"] = np.concatenate([speed, extra_s], axis=1)
    out["team"] = np.concatenate([team, extra_team])
    out["keeper_fix"] = {"slots": slots, "moved": moved}
    return out
