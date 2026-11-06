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
KICKOFF_FRAMES = (11185, 113742)   # published in DATA.md / FAQ.md: boundaries, not labels


class Params:
    p_dead_floor = 0.08      # dead-ball probability floor during visible play
    oob_dead = 0.90          # P(D) when the ball is tracked outside the pitch
    rest_dead = 0.60         # P(D) when the ball has been at rest for ~2 s
    hold_dead = 0.50         # P(D) between the ball going out and the restart kick; clear play overrides it
    run_dead_lo = 0.12       # P(D) when plenty of players are running
    run_dead_hi = 0.60       # P(D) when nobody is running
    run_threshold = 0.05     # share of tracked players above 2 m/s; trough of the histogram
    run_scale = 0.02
    shape_weight = 0.15      # scale of the team-shape log-odds while the ball is lost
    shape_weight_visible = 0.03  # ... and while it is visible
    tag_weight = 0.05        # strength of the Smart Tagging team prior (log-odds units)
    restart_lead_s = 3.5     # a kick-off tag starts ~3.5 s before the restart
    setpiece_dead_cap_s = 15 # at most this much of a set-piece window before the restart is forced dead
    sigma_control = 1.5      # m, per-frame contact radius
    eps = 0.25               # floor weight for each team in the per-frame term
    v_rel_slack = 2.0        # m/s
    v_rel_scale = 4.0        # m/s
    contact_weight = 0.5     # weight of the per-frame contact term against the episode term
    contact_weight_touch = 0.1       # ... inside a touch episode
    contact_weight_unresolved = 0.3  # ... inside an unresolved episode
    use_broadcast = False    # broadcast fill evaluated and dropped: no gain on any check
    switch_team = 15.0       # nats
    switch_dead = 25.0       # nats
    min_dead_s = 2.0         # dead-ball spells shorter than this are not stoppages
    carry_dead = 0.55        # P(D) while the ball is walked along a line
    ignored_dead = 0.60      # P(D) while a slow ball sits far from everyone
    restart_pd = 0.45        # mean P(D) over the 2 s before a kick for it to count as a restart
    restart_weight = 1.5     # nats per frame for the restarting side ...
    restart_boost_s = 0.8    # ... for this long after the kick
    transition_s = 3.0       # seconds after a TRANSITION tag start that favour the gaining side
    transition_weight = 0.5  # nats per frame


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


def emissions(feat, tag_prior, set_pieces, goals, transitions, graded, n, fps, p=Params):
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
    pd = np.where(feat["line_carry"], np.maximum(pd, p.carry_dead), pd)
    pd = np.where(feat["ignored"], np.maximum(pd, p.ignored_dead), pd)
    v_ball = np.nan_to_num(feat["v_ball_s"], nan=0.0)

    # --- team split: control episodes and their outcomes (possession/episodes.py), plus a
    # down-weighted per-frame contact term so sustained but untidy contact still counts ------
    share_ep, ep_counts, _, kind_of = home_share(feat, inplay, fps)
    c_home = np.exp(-(np.nan_to_num(d_home, nan=1e3) / p.sigma_control) ** 2)
    c_away = np.exp(-(np.nan_to_num(d_away, nan=1e3) / p.sigma_control) ** 2)
    v_ball = np.nan_to_num(feat["v_ball_s"], nan=0.0)
    rel_home = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_home"], nan=0.0) - p.v_rel_slack)
    rel_away = np.maximum(0.0, v_ball - np.nan_to_num(feat["v_away"], nan=0.0) - p.v_rel_slack)
    w_h = np.where(inplay, p.eps + c_home * np.exp(-(rel_home / p.v_rel_scale) ** 2), 1.0)
    w_a = np.where(inplay, p.eps + c_away * np.exp(-(rel_away / p.v_rel_scale) ** 2), 1.0)
    # The per-frame term is muted inside episodes whose outcome says the contact was not
    # control: a touch that went straight to the opponent must not accumulate into a switch.
    cw = np.full(n, p.contact_weight)
    cw[kind_of == 3] = p.contact_weight_touch
    cw[kind_of == 4] = p.contact_weight_unresolved
    logit = np.log(share_ep / (1 - share_ep)) + cw * np.log(w_h / w_a)
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

    # --- restarts: the side that kicks the ball out of a stoppage has possession ---------------
    pd_cum = np.concatenate(([0.0], np.cumsum(pd)))
    look = 2 * fps
    boost = int(round(p.restart_boost_s * fps))
    n_restart_kicks = 0
    for kf in np.flatnonzero(feat["kick_team"] >= 0):
        if kf < look:
            continue
        if (pd_cum[kf] - pd_cum[kf - look]) / look > p.restart_pd:
            t = H if feat["kick_team"][kf] == 0 else A
            # takeoff is detected five frames after the ball leaves the foot: start there
            log_e[max(0, kf - 5):kf - 5 + boost, t] += p.restart_weight
            n_restart_kicks += 1

    # --- the two published kick-offs: the ball is in play from that frame -------------------
    for kf in KICKOFF_FRAMES:
        log_e[kf:kf + 2 * fps, H] += 2.0
        log_e[kf:kf + 2 * fps, A] += 2.0

    # --- Smart Tagging transitions: the gaining side, for a few seconds -------------------
    for start, end, team in transitions:
        a, b = max(0, start), min(n, start + int(round(p.transition_s * fps)))
        log_e[a:b, H if team == "H" else A] += p.transition_weight

    # --- Smart Tagging team prior (soft) ----------------------------------------------------
    log_e[:, H] += p.tag_weight * tag_prior
    log_e[:, A] -= p.tag_weight * tag_prior

    # Frames that visibly show play: tracked ball moving with players running. No tag window
    # is allowed to force a dead ball on them.
    clear_play = inplay & (v_ball > 3.0) & (run > 0.10)

    # --- set pieces: dead before the restart, restart team after it -------------------------
    lead = int(round(p.restart_lead_s * fps))
    for start, end, team, kind in set_pieces:
        a, b = max(0, start), min(n, end)
        if b <= a:
            continue
        # The restart is the first kick from a player's feet inside the window (a ball thrown
        # back onto the pitch does not count); failing that, the typical lead into the window.
        kicks = np.flatnonzero(feat["kicked"][a:b])
        restart = a + (int(kicks[0]) if len(kicks) else min(lead, b - a - 1))
        # Windows vary from 5 s (kick-offs) to over half a minute (free kicks, goal kicks):
        # the stoppage is the part before the restart, capped, and the restarting side is
        # credited only for the first moments after it.
        # ... and never on frames that visibly show play (ball tracked, moving, players running).
        seg = slice(max(a, restart - p.setpiece_dead_cap_s * fps), restart)
        log_e[seg, D] += 2.0 * ~clear_play[seg]
        if team is not None:
            t = H if team == "H" else A
            log_e[restart:min(b, restart + int(round(p.restart_boost_s * fps))), t] += 1.5
    # Goals: a GOALS tag can start minutes before the goal (it covers the move that led to it)
    # and ends just after the kick-off tag begins. The goal itself is the start of the longest
    # quiet stretch before that kick-off -- ball lost or slow, nobody running: the celebration
    # and the walk back -- and the ball is dead from there until the kick-off is taken.
    kickoffs = sorted(s for s, _, _, kind in set_pieces if kind.startswith("kick off"))
    quiet = (~visible | (v_ball < 3.0)) & (run < 0.15)
    for start, end, team in goals:
        inside = [kf for kf in kickoffs if start <= kf <= end]
        ko = inside[-1] if inside else end
        best_len, best_start, cur = 0, start, None
        for f in range(start, ko):
            if quiet[f]:
                if cur is None:
                    cur = f
                if f - cur + 1 > best_len:
                    best_len, best_start = f - cur + 1, cur
            else:
                cur = None
        stop = min(n, ko + lead)
        log_e[best_start:stop, D] += 3.0 * ~clear_play[best_start:stop]

    diag = {"shape_k": k, "shape_acc": acc, "shape_n": n_clear, "episodes": ep_counts,
            "restart_kicks": n_restart_kicks,
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
