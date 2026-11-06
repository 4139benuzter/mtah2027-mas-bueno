"""Per-frame signals derived from the tracking. All arrays are indexed by video frame."""

import numpy as np

from .atd import HOME, AWAY


def moving_mean(x, half):
    """Centred moving mean ignoring NaN, window +/- half frames."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0.0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    s = cs[hi] - cs[lo]
    n = cn[hi] - cn[lo]
    out = np.full(len(x), np.nan)
    out[n > 0] = s[n > 0] / n[n > 0]
    return out


def ball_speed(ball, fps):
    """m/s from centred finite differences where both neighbours are tracked, else NaN."""
    n = len(ball)
    v = np.full(n, np.nan)
    d = ball[2:] - ball[:-2]
    ok = ~np.isnan(d).any(axis=1)
    v[1:-1][ok] = np.hypot(d[ok, 0], d[ok, 1]) * fps / 2.0
    return v


def missing_stretches(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def compute(trk, pitch, fps):
    ball = trk["ball"]
    players = trk["players"]
    team = trk["team"]
    speed = trk["speed"]
    length, width = pitch
    n = len(ball)

    visible = ~np.isnan(ball[:, 0])
    margin = 0.3
    out_of_bounds = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin) |
                               (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))

    # Distances from the ball to every tracked player, split by team.
    diff = players - ball[:, None, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])          # NaN where either is missing
    home_mask = team == HOME
    with np.errstate(all="ignore"):
        dist_home = np.where(home_mask[None, :], dist, np.nan)
        dist_away = np.where(~home_mask[None, :], dist, np.nan)
        d_home = np.nanmin(dist_home, axis=1)
        d_away = np.nanmin(dist_away, axis=1)
        # Speed of the nearest player of each team (NaN where unknown).
        i_home = np.nanargmin(np.nan_to_num(dist_home, nan=np.inf), axis=1)
        i_away = np.nanargmin(np.nan_to_num(dist_away, nan=np.inf), axis=1)
    rows = np.arange(n)
    v_home = np.where(np.isfinite(d_home), speed[rows, i_home], np.nan)
    v_away = np.where(np.isfinite(d_away), speed[rows, i_away], np.nan)

    v_ball = ball_speed(ball, fps)
    v_ball_s = moving_mean(v_ball, 2)

    # Collective activity: median speed of the tracked players, smoothed over one second.
    with np.errstate(all="ignore"):
        activity = np.nanmedian(speed, axis=1)
    activity_s = moving_mean(activity, fps // 2)
    n_tracked = (~np.isnan(players[:, :, 0])).sum(axis=1)
    # Share of tracked players running (> 2 m/s). Near zero only when play has stopped.
    with np.errstate(all="ignore"):
        frac_run = np.nanmean(speed > 2.0, axis=1)
    frac_run_s = moving_mean(frac_run, fps // 2)

    # Ball at rest for a sustained spell: fraction of +/- 1 s with speed < 0.5 m/s.
    slow = np.where(np.isnan(v_ball), np.nan, (v_ball < 0.5).astype(float))
    rest_frac = moving_mean(slow, fps)

    # Team shape: lateral spread of each side. The side with the ball plays wide, the side
    # without it stays compact. Used as weak evidence when the ball is not tracked.
    with np.errstate(all="ignore"):
        y_home = np.where(home_mask[None, :], players[:, :, 1], np.nan)
        y_away = np.where(~home_mask[None, :], players[:, :, 1], np.nan)
        x_home = np.where(home_mask[None, :], players[:, :, 0], np.nan)
        x_away = np.where(~home_mask[None, :], players[:, :, 0], np.nan)
        sy_home, sy_away = np.nanstd(y_home, axis=1), np.nanstd(y_away, axis=1)
        sx_home, sx_away = np.nanstd(x_home, axis=1), np.nanstd(x_away, axis=1)
        n_home = (~np.isnan(y_home)).sum(axis=1)
        n_away = (~np.isnan(y_away)).sum(axis=1)
    shape = np.where((n_home >= 7) & (n_away >= 7), sy_home - sy_away, np.nan)
    shape_s = moving_mean(shape, fps)
    area = np.where((n_home >= 7) & (n_away >= 7),
                    np.log((sx_home * sy_home + 1e-6) / (sx_away * sy_away + 1e-6)), np.nan)
    area_s = moving_mean(area, fps)

    # Ball lost within 2 m of a line while travelling towards it: it has probably gone out.
    vx = moving_mean(np.concatenate(([np.nan], (ball[2:, 0] - ball[:-2, 0]) * fps / 2, [np.nan])), 2)
    vy = moving_mean(np.concatenate(([np.nan], (ball[2:, 1] - ball[:-2, 1]) * fps / 2, [np.nan])), 2)
    outward = np.maximum.reduce([
        np.where(ball[:, 0] < 2.0, -vx, -np.inf), np.where(ball[:, 0] > length - 2.0, vx, -np.inf),
        np.where(ball[:, 1] < 2.0, -vy, -np.inf), np.where(ball[:, 1] > width - 2.0, vy, -np.inf)])
    lost_near_line = np.zeros(n, dtype=bool)
    for start, stop in missing_stretches(~visible):
        if start == 0:
            continue
        last = start - 1
        if out_of_bounds[last] or (np.isfinite(outward[last]) and outward[last] > 1.0):
            lost_near_line[start:stop] = True

    # Restart hold: once the ball has gone out (tracked outside the lines, or lost at a line) it
    # stays dead until it is put back into play. The restart is the first sustained kick -- the
    # ball tracked in bounds above 5 m/s for three frames -- or, failing that, the players
    # clearly running again. A restart cannot happen within 3 s of the ball going out, and the
    # hold gives up after 15 s.
    fast = visible & ~out_of_bounds & (np.nan_to_num(v_ball_s, nan=0.0) > 5.0)
    kicked = fast.copy()
    kicked[2:] &= fast[1:-1] & fast[:-2]
    running = np.nan_to_num(frac_run_s, nan=0.0) > 0.2
    trigger = out_of_bounds | lost_near_line
    hold = np.zeros(n, dtype=bool)
    min_hold, timeout = 3 * fps, 25 * fps
    t = 0
    while t < n:
        if trigger[t]:
            stop = min(n, t + timeout)
            release = np.flatnonzero((kicked | running)[t + min_hold:stop])
            stop = t + min_hold + int(release[0]) if len(release) else stop
            hold[t:stop] = True
            t = stop
        else:
            t += 1

    return {
        "visible": visible, "oob": out_of_bounds,
        "d_home": d_home, "d_away": d_away, "v_home": v_home, "v_away": v_away,
        "v_ball": v_ball, "v_ball_s": v_ball_s,
        "activity": activity_s, "frac_run": frac_run_s, "n_tracked": n_tracked,
        "rest_frac": rest_frac,
        "shape": shape_s, "area": area_s,
        "lost_near_line": lost_near_line, "hold": hold,
    }
