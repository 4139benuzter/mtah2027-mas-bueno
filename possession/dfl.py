"""DFL / Sportec open data as a VALIDATION set from other matches (never used to choose anything).

Bassek et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data, doi:10.1038/s41597-025-04505-y; files on figshare (article 28196177), licence
CC BY 4.0, released with the authorisation of the DFL. Seven matches of the 2022/23 Bundesliga and
2. Bundesliga, 25 fps. Their corrected tracking carries a per-frame BallPossession (home/away) and
BallStatus (alive/dead) -- the same kind of fields from which the competition ground truth is
derived ("D where the corrected tracking marks the ball dead, otherwise the team it names as
owning it"). None of them is the competition match.

    python -m possession.dfl <folder with the DFL xml files> <out folder>   # needs kloppy
    trk, labels, info = load_npz("<out folder>/J03WOH.npz")                # numpy only

Conversion: kloppy (3.19.1) reads the match information and the raw positions with
coordinates="sportec" (metres, origin at the centre) and only_alive=False; we shift to a corner
origin, give each player his own column, compute speeds from positions as for Metrica's sample
matches, and write labels 'D' where the ball is dead, else 'H'/'A' from the owning team. The two
halves are separated by 25 frames of 'X' so that no possession straddles half-time. The ball is
never missing in this data; possession/validate.py can remove it at random to the competition
match's visibility (59% of graded frames).
"""

import glob
import os
import re
import sys

import numpy as np

GAP = 25   # frames of X between the halves


def convert(folder, match_id, out_path):
    from kloppy import sportec
    from kloppy.domain import BallState
    from possession.sample_game import _player_speed

    meta = glob.glob(os.path.join(folder, "*matchinformation*{}*.xml".format(match_id)))[0]
    raw = glob.glob(os.path.join(folder, "*positions_raw_observed*{}*.xml".format(match_id)))[0]
    ds = sportec.load_tracking(meta_data=meta, raw_data=raw, coordinates="sportec",
                               only_alive=False)
    home, away = ds.metadata.teams
    players = [p for t in (home, away) for p in t.players]
    col = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), dtype=np.int8)
    half_len = ds.metadata.pitch_dimensions.pitch_length / 2.0
    half_wid = ds.metadata.pitch_dimensions.pitch_width / 2.0

    frames = list(ds.frames)
    periods = sorted({f.period.id for f in frames})
    n = len(frames) + GAP * (len(periods) - 1)
    xy = np.full((n, len(players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    labels = np.full(n, "X", dtype="<U1")
    kickoffs, halves = [], []
    row, last_period = 0, None
    for f in frames:
        if last_period is not None and f.period.id != last_period:
            halves[-1] = (halves[-1][0], halves[-1][1], row - 1)
            row += GAP
        if f.period.id != last_period:
            kickoffs.append(row)
            halves.append((f.period.id, row, None))
            last_period = f.period.id
        for p, d in f.players_data.items():
            if d.coordinates is not None and p.player_id in col:
                xy[row, col[p.player_id]] = (d.coordinates.x + half_len, d.coordinates.y + half_wid)
        if f.ball_coordinates is not None:
            ball[row] = (f.ball_coordinates.x + half_len, f.ball_coordinates.y + half_wid)
        if f.ball_state == BallState.DEAD:
            labels[row] = "D"
        elif f.ball_owning_team is not None:
            labels[row] = "H" if f.ball_owning_team == home else "A"
        row += 1
    halves[-1] = (halves[-1][0], halves[-1][1], row - 1)
    np.savez_compressed(out_path, ball=ball, players=xy, team=team,
                        speed=_player_speed(xy, 25), labels=labels,
                        kickoffs=np.array(kickoffs), halves=np.array(halves),
                        pitch=np.array([2 * half_len, 2 * half_wid]))
    return out_path


def load_npz(path):
    """(trk, labels, info) in the shape possession.sample_game.load returns."""
    d = np.load(path)
    trk = {"ball": d["ball"], "players": d["players"], "team": d["team"], "speed": d["speed"]}
    info = {"kickoffs": tuple(int(k) for k in d["kickoffs"]),
            "halves": [tuple(int(v) for v in h) for h in d["halves"]],
            "pitch": tuple(float(v) for v in d["pitch"])}
    return trk, d["labels"], info


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    ids = sorted({re.search(r"DFL-MAT-(\w+)\.xml", p).group(1)
                  for p in glob.glob(os.path.join(src, "*positions_raw_observed*.xml"))})
    only = sys.argv[3:]
    for match_id in ids:
        if only and match_id not in only:
            continue
        path = convert(src, match_id, os.path.join(out, match_id + ".npz"))
        trk, labels, info = load_npz(path)
        graded = labels != "X"
        print("{}: frames {} players {} pitch {} alive {:.1%} halves {}".format(
            match_id, len(labels), trk["players"].shape[1], info["pitch"],
            (labels[graded] != "D").mean(), info["halves"]), flush=True)
