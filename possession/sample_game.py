"""Metrica's public sample matches as labelled calibration and test sets from OTHER matches.

RULES.md allows labelled data about other games in any amount, provided the organisers can reach
it and the technical document explains it. Metrica publishes three anonymised matches at
https://github.com/metrica-sports/sample-data, each with tracking and human-coded events:

  game 3  FIFA/EPTS text layout like the competition tracking, but positions only (no speed
          channel); "Team A" v "Team B", competition "Demo", metadata dated 2020-05-08, added to
          that repository 2021-04-15, final score 0-2.
  game 1, game 2  Metrica CSV layout (one file per team), added 2020-04-01; 3-0 and 3-2.
All three are used by possession/calibrate.py: leave-one-match-out to estimate the gain on an
unseen match, then all three together for the final choice of decoder settings.

None of them is the competition match (LaLiga 2024-25). Nothing here reads the competition match
or its public ground truth.

    from possession.sample_game import load
    trk, labels, info = load(folder, game=3)   # trk shaped like possession.atd.load_tracking

Files (download from .../sample-data/master/data/Sample_Game_<g>/ into one folder):
    game 3: Sample_Game_3_tracking.txt  Sample_Game_3_metadata.xml  Sample_Game_3_events.json
    game g: Sample_Game_<g>_RawTrackingData_Home_Team.csv  ..._Away_Team.csv  ..._RawEventsData.csv

Label rules (the competition definition, applied to the human event stream):
  * A team owns the ball from the start frame of any event in which one of its players plays the
    ball (PASS, CARRY, RECOVERY, SHOT, SET PIECE, BALL LOST -- a lost ball was owned until it was
    lost). When events start on the same frame, BALL LOST yields to the others (at a theft the
    winner's RECOVERY can share the loser's frame). CHALLENGE and BALL OUT never change the owner:
    a touch or a duel is not control.
  * The ball is dead from the end frame of BALL OUT, of a SHOT that went OUT or was a GOAL, and of
    an OFFSIDE; from the start frame of FAULT RECEIVED and of a HAND BALL; and from the start of a
    foul CHALLENGE (FAULT, no ADVANTAGE) only if a restart follows before anyone plays the ball.
    If no event at all happens for more than GAP_S seconds before a restart, the ball is dead from
    the end of the last event. The dead spell ends at the next restart (SET PIECE, or a PASS coded
    GOAL KICK) or, if none is coded, at the next event in which a player plays the ball.
  * Only frames from each half's kick-off to the end of that half's last event are graded;
    everything else is 'X', exactly as in the competition ground truth.
"""

import csv
import io
import json
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1
FPS = 25
PITCH = (105.0, 68.0)    # metadata FieldSize of Sample Game 3; Metrica's sample pitch
GAP_S = 4.0              # an event-free stretch longer than this before a restart is dead ball

OWNING = {"PASS", "CARRY", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST"}


def _files(folder, game):
    if game == 3:
        names = {"raw": "Sample_Game_3_tracking.txt", "meta": "Sample_Game_3_metadata.xml",
                 "events": "Sample_Game_3_events.json"}
    else:
        stem = "Sample_Game_{}_".format(game)
        names = {"home": stem + "RawTrackingData_Home_Team.csv",
                 "away": stem + "RawTrackingData_Away_Team.csv",
                 "events": stem + "RawEventsData.csv"}
    paths = {k: os.path.join(folder, v) for k, v in names.items()}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise IOError("Sample Game {} files not found:\n  {}".format(game, "\n  ".join(missing)))
    return paths


def _player_speed(xy, fps):
    """m/s per player column: positions smoothed over +/-2 frames, centred difference over +/-2."""
    n = len(xy)
    k = 2
    pad = np.full((k, xy.shape[1], 2), np.nan)
    p = np.concatenate([pad, xy, pad])
    stack = np.stack([p[i:i + n] for i in range(2 * k + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sm = np.nanmean(stack, axis=0)
    v = np.full(xy.shape[:2], np.nan)
    d = sm[2 * k:] - sm[:-2 * k]
    v[k:n - k] = np.hypot(d[:, :, 0], d[:, :, 1]) * fps / (2 * k)
    v[np.isnan(xy[:, :, 0])] = np.nan
    return v


def _to_metres(xy, ball):
    length, width = PITCH
    xy[..., 0] *= length
    xy[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return xy, ball


# -- game 3: FIFA/EPTS ----------------------------------------------------------------------------

def _slot_layout(meta_path):
    """Team ids (home first), player -> team, and per frame range the player id of each slot."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    blocks = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        slots = [channel_player[r] for r in refs if r.endswith("_x")]
        blocks.append((int(spec.get("startFrame")), int(spec.get("endFrame")), slots))
    return team_ids, player_team, blocks


def _load_epts(paths):
    """One column per PLAYER (not per slot), so a column never changes team across substitutions;
    a player is NaN on frames where he is not on the pitch. Home is the first <Team>."""
    team_ids, player_team, blocks = _slot_layout(paths["meta"])
    with open(paths["raw"], "r") as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    values = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                         na_values=["NaN"], engine="c").values
    if not np.array_equal(values[:, 0].astype(np.int64), np.arange(1, len(values) + 1)):
        raise ValueError("frame ids are not 1..n contiguous")
    n = len(values)
    n_slots = (values.shape[1] - 3) // 2
    slots_xy = values[:, 1:1 + 2 * n_slots].reshape(n, n_slots, 2)
    ball = values[:, 1 + 2 * n_slots:1 + 2 * n_slots + 2].copy()
    players = sorted({pid for _, _, slots in blocks for pid in slots})
    col = {pid: i for i, pid in enumerate(players)}
    xy = np.full((n, len(players), 2), np.nan)
    for start, end, slots in blocks:
        rows = slice(start - 1, end)
        for s, pid in enumerate(slots):
            xy[rows, col[pid]] = slots_xy[rows, s]
    xy, ball = _to_metres(xy, ball)
    team = np.array([HOME if player_team[pid] == team_ids[0] else AWAY for pid in players],
                    dtype=np.int8)
    side = {team_ids[0]: "H", team_ids[1]: "A"}
    events = []
    for e in json.load(open(paths["events"], encoding="utf-8"))["data"]:
        sub = e.get("subtypes") or []
        sub = [sub] if isinstance(sub, dict) else sub
        events.append({"period": e["period"], "start": e["start"]["frame"] - 1,
                       "end": e["end"]["frame"] - 1, "order": e["index"],
                       "team": side.get((e.get("team") or {}).get("id")),
                       "kind": e["type"]["name"], "sub": {s["name"] for s in sub}})
    return {"ball": ball, "players": xy, "team": team}, events


# -- games 1 and 2: Metrica CSV ---------------------------------------------------------------------

def _read_team_csv(path):
    """(frame ids, (n, players, 2) normalised positions, (n, 2) ball) from one team's CSV."""
    with open(path, newline="") as handle:
        header = [next(csv.reader(handle)) for _ in range(3)][2]
    table = pd.read_csv(path, skiprows=3, header=None, dtype=np.float64,
                        na_values=["NaN"]).values
    names = header
    player_cols = [i for i, h in enumerate(names) if h.startswith("Player")]
    ball_col = names.index("Ball")
    xy = np.stack([table[:, [c, c + 1]] for c in player_cols], axis=1)
    return table[:, 1].astype(np.int64), xy, table[:, [ball_col, ball_col + 1]].copy()


def _load_csv_game(paths):
    fh, xy_h, ball = _read_team_csv(paths["home"])
    fa, xy_a, _ = _read_team_csv(paths["away"])
    if not (np.array_equal(fh, fa) and np.array_equal(fh, np.arange(1, len(fh) + 1))):
        raise ValueError("home/away frame ids are not the same 1..n")
    xy = np.concatenate([xy_h, xy_a], axis=1)
    xy, ball = _to_metres(xy, ball)
    team = np.array([HOME] * xy_h.shape[1] + [AWAY] * xy_a.shape[1], dtype=np.int8)
    side = {"Home": "H", "Away": "A"}
    events = []
    for i, r in enumerate(csv.DictReader(open(paths["events"], newline=""))):
        events.append({"period": int(r["Period"]), "start": int(r["Start Frame"]) - 1,
                       "end": int(r["End Frame"]) - 1, "order": i, "team": side.get(r["Team"]),
                       "kind": r["Type"].strip().upper(),
                       "sub": {s.strip().upper() for s in r["Subtype"].split("-") if s.strip()}})
    return {"ball": ball, "players": xy, "team": team}, events


# -- labels ---------------------------------------------------------------------------------------

def labels_from_events(events, n, fps=FPS):
    """(labels, info): (n,) 'H','A','D','X' from a list of normalised events (see module doc)."""
    labels = np.full(n, "X", dtype="<U1")
    info = {"halves": []}
    gap = int(round(GAP_S * fps))
    for period in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == period),
                     key=lambda e: (e["start"], e["kind"] == "BALL LOST", e["order"]))
        first = min(e["start"] for e in evs if e["kind"] == "SET PIECE" and "KICK OFF" in e["sub"])
        last = max(e["end"] for e in evs)
        own, restart, dead_from, fouls = {}, {}, set(), []
        for e in evs:
            kind, sub, t, s, f = e["kind"], e["sub"], e["team"], e["start"], e["end"]
            if kind in OWNING and t is not None:
                own.setdefault(s, t)
            if (kind == "SET PIECE" or (kind == "PASS" and "GOAL KICK" in sub)) and t is not None:
                restart.setdefault(s, t)
            if kind == "BALL OUT" or (kind == "SHOT" and sub & {"OUT", "GOAL"}):
                dead_from.add(f)
            elif kind == "BALL LOST" and "OFFSIDE" in sub:
                dead_from.add(f)
            elif kind == "BALL LOST" and "HAND BALL" in sub:
                dead_from.add(s)
            elif kind == "FAULT RECEIVED":
                dead_from.add(s)
            elif kind == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
                fouls.append(s)
        plays = sorted(f for f in own if f not in restart)
        restarts = sorted(restart)
        for s in fouls:                      # a foul stops play only if a restart comes next
            nxt_play = next((f for f in plays if f > s), None)
            nxt_restart = next((f for f in restarts if f > s), None)
            if nxt_restart is not None and (nxt_play is None or nxt_restart <= nxt_play):
                dead_from.add(s)
        for r in restarts:                   # a long event-free stretch before a restart is dead
            before = [e["end"] for e in evs if e["start"] < r]
            if before and r - max(before) > gap:
                dead_from.add(max(before) + 1)

        team = restart.get(first) or own.get(first)
        dead = False
        for frame in range(first, last + 1):
            if frame in dead_from and frame != first:
                dead = True
            if frame in restart:
                dead, team = False, restart[frame]
            elif frame in own:
                if dead and frame not in dead_from:
                    dead = False             # play resumed without a coded restart
                if not dead:
                    team = own[frame]
            labels[frame] = "D" if dead else team
        info["halves"].append((period, first, last))
    info["kickoffs"] = tuple(h[1] for h in info["halves"])
    return labels, info


def load(folder, game=3):
    """(trk, labels, info) for one sample match; trk also gets per-player speeds.
    A path ending in .npz is a DFL match converted by possession/dfl.py instead."""
    if str(folder).endswith(".npz"):
        from possession.dfl import load_npz
        return load_npz(folder)
    paths = _files(folder, game)
    trk, events = _load_epts(paths) if game == 3 else _load_csv_game(paths)
    trk["speed"] = _player_speed(trk["players"], FPS)
    labels, info = labels_from_events(events, len(trk["ball"]))
    return trk, labels, info


def summary(labels, fps=FPS):
    """Label-free description of a label array: shares, possessions, mean possession length."""
    graded = labels != "X"
    g = labels[graded]
    runs, prev = [], None
    for ch in labels:
        if ch in "HA" and ch != prev:
            runs.append(1)
        elif ch in "HA":
            runs[-1] += 1
        prev = ch if ch != "X" else None
    return {"graded": int(graded.sum()),
            "H": round(float((g == "H").mean()), 3), "A": round(float((g == "A").mean()), 3),
            "D": round(float((g == "D").mean()), 3), "possessions": len(runs),
            "mean_possession_s": round(float(np.mean(runs) / fps), 2) if runs else 0.0,
            "median_possession_s": round(float(np.median(runs) / fps), 2) if runs else 0.0}


if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    game = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    trk, labels, info = load(folder, game)
    vis = ~np.isnan(trk["ball"][:, 0])
    graded = labels != "X"
    print("game {}: frames {}  players {} (home {})  ball visible {:.1%} all / {:.1%} graded".format(
        game, len(trk["ball"]), trk["players"].shape[1], int((trk["team"] == HOME).sum()),
        vis.mean(), vis[graded].mean()))
    print("halves", info["halves"])
    print(summary(labels))
