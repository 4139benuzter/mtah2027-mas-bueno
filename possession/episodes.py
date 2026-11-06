"""Control episodes and their outcomes: the possession rules as evidence.

The challenge defines a change of possession as the opponent gaining *controlled* possession,
and offers two proxies: a player keeps the ball for a few seconds, or a team strings passes
together. A touch without control -- a clearance that goes straight back, a deflection, a header
won but not kept -- does not count.

Per-frame "nearest player" evidence cannot express that, because whether a touch was control
depends on what happened next. So the tracking is first cut into contact episodes (ball at a
player's feet, moving with him) and the flights between them, and each episode is classified
by its own length and by where the ball went afterwards:

    controlled   the player kept it for >= hold_s                -> strong evidence
    pass         short contact, next contact is a team-mate      -> strong evidence, incl. flight
    touch        short contact, next contact is an opponent      -> no evidence at all
    unresolved   short contact, ball lost or nothing follows     -> medium evidence

The result is a per-frame share in [0, 1] for home possession that the decoder consumes.
"""

import numpy as np

from .atd import HOME, AWAY


class EpisodeParams:
    radius = 1.5          # m, ball within this of a player counts as contact
    rel_slack = 2.0       # m/s, ball may outrun the player by this
    rel_scale = 4.0       # m/s, fade of the "moving with him" factor
    with_min = 0.3        # minimum "moving with him" factor for contact
    min_contact = 3       # frames, shorter runs are noise
    merge_gap = 5         # frames, same-team gaps shorter than this are bridged
    hold_s = 2.0          # seconds of contact that is control regardless of outcome
    max_flight_s = 4.0    # a flight longer than this resolves nothing
    max_lost_s = 1.0      # ball may vanish this long inside a flight and still link
    strong = 0.88         # home share for strong evidence (or 1 - strong for away)
    medium = 0.50         # unresolved contacts are no better than a coin flip against the truth
    light = 0.58          # a touch is usually by the side already in possession, so a little credit
    neutral = 0.50


def contact_team(feat, inplay, p=EpisodeParams):
    """(n,) int8: HOME / AWAY where a player of that side is in contact with the ball, else -1."""
    d_home, d_away = feat["d_home"], feat["d_away"]
    v_ball = np.nan_to_num(feat["v_ball_s"], nan=0.0)
    rel_h = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_home"], nan=0.0) - p.rel_slack)
    rel_a = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_away"], nan=0.0) - p.rel_slack)
    with_h = np.exp(-(rel_h / p.rel_scale) ** 2)
    with_a = np.exp(-(rel_a / p.rel_scale) ** 2)
    ok_h = inplay & np.isfinite(d_home) & (d_home < p.radius) & (with_h > p.with_min)
    ok_a = inplay & np.isfinite(d_away) & (d_away < p.radius) & (with_a > p.with_min)
    team = np.full(len(d_home), -1, dtype=np.int8)
    home_closer = np.nan_to_num(d_home, nan=np.inf) <= np.nan_to_num(d_away, nan=np.inf)
    team[ok_h & (~ok_a | home_closer)] = HOME
    team[ok_a & (~ok_h | ~home_closer)] = AWAY
    return team


def episodes_from(team, p=EpisodeParams):
    """[(start, stop, team)] contact runs, half-open, bridged over short same-team gaps."""
    n = len(team)
    runs = []
    t = 0
    while t < n:
        if team[t] < 0:
            t += 1
            continue
        k = team[t]
        s = t
        while t < n and team[t] == k:
            t += 1
        runs.append([s, t, int(k)])
    merged = []
    for run in runs:
        if merged and merged[-1][2] == run[2] and run[0] - merged[-1][1] <= p.merge_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [tuple(r) for r in merged if r[1] - r[0] >= p.min_contact]


def classify(eps, visible, fps, p=EpisodeParams):
    """Attach an outcome to every episode: 'controlled', 'pass', 'touch' or 'unresolved'."""
    out = []
    hold = int(round(p.hold_s * fps))
    max_flight = int(round(p.max_flight_s * fps))
    max_lost = int(round(p.max_lost_s * fps))
    lost_cum = np.concatenate(([0], np.cumsum(~visible)))
    for i, (s, e, k) in enumerate(eps):
        if e - s >= hold:
            out.append((s, e, k, "controlled", e))
            continue
        nxt = eps[i + 1] if i + 1 < len(eps) else None
        if nxt is None or nxt[0] - e > max_flight or lost_cum[nxt[0]] - lost_cum[e] > max_lost:
            out.append((s, e, k, "unresolved", e))
        elif nxt[2] == k:
            out.append((s, e, k, "pass", nxt[0]))
        else:
            out.append((s, e, k, "touch", e))
    return out


def home_share(feat, inplay, fps, p=EpisodeParams):
    """(n,) P(home has the ball | tracking) from the episode rules, 0.5 where nothing is known."""
    n = len(inplay)
    share = np.full(n, p.neutral)
    team = contact_team(feat, inplay, p)
    eps = classify(episodes_from(team, p), feat["visible"], fps, p)
    counts = {"controlled": 0, "pass": 0, "touch": 0, "unresolved": 0}
    for s, e, k, kind, until in eps:
        counts[kind] += 1
        level = {"controlled": p.strong, "pass": p.strong, "touch": p.light,
                 "unresolved": p.medium}[kind]
        share[s:until] = level if k == HOME else 1.0 - level
    return share, counts, eps
