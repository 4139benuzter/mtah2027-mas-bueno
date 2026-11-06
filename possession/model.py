"""Three-state possession decoder: H / A / D as a hidden Markov chain over frames.

Emissions come from the tracking (who is close to the ball, is it in play, is the ball under
control or in flight, is anybody running, how are the teams shaped) and from the Smart Tagging
(which team the phase belongs to). Transitions carry a fixed switch cost, which is what turns
frame-level flicker into whole possessions: the chain only changes state when the evidence for
the new state, accumulated over consecutive frames, outweighs the cost.

No constant here is fitted to any ground truth. The switch costs are expressed as "seconds of
typical evidence", the control radius and speeds are football quantities, the running-share
threshold sits in the trough of the bimodal full-match histogram of that feature, and the one
learned weight (team shape -> possession) is fitted to pseudo-labels the tracking itself produces
on frames where the ball is unambiguously at a player's feet.
"""

import numpy as np

from .episodes import home_share

H, A, D = 0, 1, 2
N_STATES = 3


class Params:
    p_dead_floor = 0.08      # dead-ball probability floor during visible play
    oob_dead = 0.90          # P(D) when the ball is tracked outside the pitch
    rest_dead = 0.60         # P(D) when the ball has been at rest for ~2 s
    hold_dead = 0.60         # P(D) between the ball going out and the restart kick
    run_dead_lo = 0.12       # P(D) when plenty of players are running
    run_dead_hi = 0.60       # P(D) when nobody is running
    run_threshold = 0.05     # share of tracked players above 2 m/s; trough of the histogram
    run_scale = 0.02
    shape_weight = 0.15      # scale of the team-shape log-odds while the ball is lost
    shape_weight_visible = 0.03  # ... and while it is visible
    tag_weight = 0.05        # strength of the Smart Tagging team prior (log-odds units)
    restart_lead_s = 3.5     # a set-piece tag starts ~3.5 s before the restart
    sigma_control = 1.5      # m, per-frame contact radius
    eps = 0.25               # floor weight for each team in the per-frame term
    v_rel_slack = 2.0        # m/s
    v_rel_scale = 4.0        # m/s
    contact_weight = 0.5     # weight of the per-frame contact term against the episode term
    switch_team = 15.0       # nats
    switch_dead = 25.0       # nats
    min_dead_s = 2.0         # dead-ball spells shorter than this are not stoppages


def fit_shape_logit(feat, graded):
    """Logistic slope of P(home has ball) against the lateral-spread difference.

    Pseudo-labels: frames where the ball is tracked, in play, within 1.5 m of one side's player
    and more than 3 m from the other's. One-dimensional Newton iterations, no intercept (the
    feature is antisymmetric in the teams, so the prior must be too).
    """
    d_home, d_away, s = feat["d_home"], feat["d_away"], feat["shape"]
    clear = (feat["visible"] & ~feat["oob"] & graded & np.isfinite(s) &
             (np.fmin(d_home, d_away) < 1.5) & (np.fmax(d_home, d_away) > 3.0))
    x = s[clear]
    y = (d_home < d_away)[clear].astype(np.float64)
    k = 0.0
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-k * x))
        grad = np.sum((y - p) * x)
        hess = -np.sum(p * (1 - p) * x * x) - 1e-6
        k -= grad / hess
    acc = np.mean((x * k > 0) == (y > 0.5))
    return float(k), float(acc), int(clear.sum())


def emissions(feat, tag_prior, set_pieces, goals, graded, n, fps, p=Params):
    """(n, 3) log-probabilities, plus a dict of diagnostics."""
    visible, oob = feat["visible"], feat["oob"]
    d_home, d_away = feat["d_home"], feat["d_away"]
    inplay = visible & ~oob
    lost = ~visible

    # --- dead-ball probability ----------------------------------------------------------------
    run = np.nan_to_num(feat["frac_run"], nan=0.3)
    quiet = 1.0 / (1.0 + np.exp((run - p.run_threshold) / p.run_scale))
    pd = p.run_dead_lo + (p.run_dead_hi - p.run_dead_lo) * quiet
    pd = np.maximum(pd, p.p_dead_floor)
    rest = np.nan_to_num(feat["rest_frac"], nan=0.0)
    pd = np.where(rest > 0.9, np.maximum(pd, p.rest_dead), pd)
    pd = np.where(feat["hold"], np.maximum(pd, p.hold_dead), pd)
    pd = np.where(oob, np.maximum(pd, p.oob_dead), pd)
    v_ball = np.nan_to_num(feat["v_ball_s"], nan=0.0)

    # --- team split: control episodes and their outcomes (possession/episodes.py), plus a
    # down-weighted per-frame contact term so sustained but untidy contact still counts ------
    share_ep, ep_counts, _ = home_share(feat, inplay, fps)
    c_home = np.exp(-(np.nan_to_num(d_home, nan=1e3) / p.sigma_control) ** 2)
    c_away = np.exp(-(np.nan_to_num(d_away, nan=1e3) / p.sigma_control) ** 2)
    v_ball = np.nan_to_num(feat["v_ball_s"], nan=0.0)
    rel_home = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_home"], nan=0.0) - p.v_rel_slack)
    rel_away = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_away"], nan=0.0) - p.v_rel_slack)
    w_h = np.where(inplay, p.eps + c_home * np.exp(-(rel_home / p.v_rel_scale) ** 2), 1.0)
    w_a = np.where(inplay, p.eps + c_away * np.exp(-(rel_away / p.v_rel_scale) ** 2), 1.0)
    logit = np.log(share_ep / (1 - share_ep)) + p.contact_weight * np.log(w_h / w_a)
    share_h = 1.0 / (1.0 + np.exp(-logit))

    e = np.empty((n, N_STATES))
    e[:, H] = (1 - pd) * share_h
    e[:, A] = (1 - pd) * (1 - share_h)
    e[:, D] = pd
    log_e = np.log(e)

    # --- team shape: wide side has the ball -------------------------------------------------
    k, acc, n_clear = fit_shape_logit(feat, graded)
    shape_llr = np.clip(k * np.nan_to_num(feat["shape"], nan=0.0), -2.0, 2.0)
    w_shape = np.where(lost, p.shape_weight, p.shape_weight_visible)
    log_e[:, H] += 0.5 * w_shape * shape_llr
    log_e[:, A] -= 0.5 * w_shape * shape_llr

    # --- Smart Tagging team prior (soft) ----------------------------------------------------
    log_e[:, H] += p.tag_weight * tag_prior
    log_e[:, A] -= p.tag_weight * tag_prior

    # --- set pieces: dead before the restart, restart team after it -------------------------
    lead = int(round(p.restart_lead_s * fps))
    for start, end, team, kind in set_pieces:
        a, b = max(0, start), min(n, end)
        if b <= a:
            continue
        moving = np.flatnonzero((v_ball[a:b] > 2.0) & visible[a:b])
        restart = a + (int(moving[0]) if len(moving) else min(lead, b - a - 1))
        log_e[a:restart, D] += 2.0
        if team is not None:
            t = H if team == "H" else A
            log_e[restart:b, t] += 1.5
    # Goals: the GOALS tag runs from the goal to just after the kick-off tag begins, so the ball
    # is dead from the tag start until the kick-off inside it is taken (tag start + lead).
    kickoffs = sorted(s for s, _, _, kind in set_pieces if kind.startswith("kick off"))
    for start, end, team in goals:
        inside = [kf for kf in kickoffs if start <= kf <= end]
        stop = min(n, (inside[-1] + lead) if inside else end)
        log_e[start:stop, D] += 3.0

    diag = {"shape_k": k, "shape_acc": acc, "shape_n": n_clear, "episodes": ep_counts,
            "pd_mean_graded": float(pd[graded].mean()), "hold_share": float(feat["hold"][graded].mean())}
    return log_e, diag


def switch_costs(log_e, feat, graded, fps, p=Params):
    """Fixed costs in nats. With strong episode evidence worth ~1.8 nats per frame, a change of
    team needs about half a second of controlled contact; a stoppage about a second of
    nobody-running evidence."""
    return p.switch_team, p.switch_dead


def viterbi(log_e, cost_team, cost_dead):
    """Most probable state path under fixed switch costs."""
    n = len(log_e)
    cost = np.zeros((N_STATES, N_STATES))
    cost[H, A] = cost[A, H] = cost_team
    cost[H, D] = cost[A, D] = cost[D, H] = cost[D, A] = cost_dead
    trans = -cost
    score = log_e[0].copy()
    back = np.zeros((n, N_STATES), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + trans              # [from, to]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(N_STATES)] + log_e[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def merge_short_dead(path, fps, p=Params):
    """Fold dead-ball spells shorter than min_dead_s into the spell that follows them.

    A stoppage takes several seconds by nature (the ball has to be fetched and the restart
    taken), so a sub-2 s D is a flicker in the evidence, not a dead ball.
    """
    path = path.copy()
    n = len(path)
    min_len = int(round(p.min_dead_s * fps))
    breaks = np.flatnonzero(path[1:] != path[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n]))
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if path[a] == D and b - a < min_len:
            fill = path[b] if b < n else (path[a - 1] if a > 0 else D)
            path[a:b] = fill
    return path
