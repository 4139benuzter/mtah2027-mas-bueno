"""Parse the raw Metrica FIFA/EPTS tracking export into numpy arrays, cached on disk.

Raw line:  "<frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<ball_x,ball_y,ball_s>"
Coordinates are normalised 0-1; we convert to metres with the pitch from games.py.

    from possession.atd import load_tracking
    trk = load_tracking()              # tactical, cached in cache/
    trk["ball"]      (n, 2)  metres, NaN where the ball is not tracked
    trk["players"]   (n, 46, 2) metres, NaN where the slot is empty
    trk["speed"]     (n, 46) m/s from the export (NaN where empty)
    trk["team"]      (46,)  0 = home (Barcelona), 1 = away (Athletic)
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}".format(meta_path))
    home_id = team_ids[0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_slots, n_frames):
    """Stream the raw file through pandas after flattening the separators.

    Every line has 1 + 3*n_slots + 3 numeric fields once ':' and ';' become ','. pandas' C
    parser handles 'NaN' natively, so a 200 MB file loads in a few seconds.
    """
    with open(raw_path, "r") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(pd.io.common.StringIO(text), header=None, dtype=np.float64,
                        na_values=["NaN"], engine="c")
    values = table.values
    expected_cols = 1 + 3 * n_slots + 3
    if values.shape[1] != expected_cols:
        raise ValueError("expected {} columns, got {}".format(expected_cols, values.shape[1]))
    frame_ids = values[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, len(frame_ids) + 1)):
        raise ValueError("frame ids are not 1..n contiguous")
    if len(frame_ids) != n_frames:
        raise ValueError("got {} records, expected {}".format(len(frame_ids), n_frames))
    body = values[:, 1:1 + 3 * n_slots].reshape(len(values), n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:1 + 3 * n_slots + 3]
    return body, ball


def load_tracking(source=DEFAULT_SOURCE, game=GAME, force=False):
    """Cached numpy version of the ATD, in metres."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "atd_{}.npz".format(source))
    if os.path.isfile(cache) and not force:
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, None, source)
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    n_frames = source_frames(source, game)
    length, width = GAMES[game]["pitch"]

    body, ball = parse_raw(raw_path, n_slots, n_frames)
    players = body[:, :, :2].copy()
    players[:, :, 0] *= length
    players[:, :, 1] *= width
    speed = body[:, :, 2].copy()
    ball_xy = ball[:, :2].copy()
    ball_xy[:, 0] *= length
    ball_xy[:, 1] *= width

    out = {"ball": ball_xy, "ball_s": ball[:, 2], "players": players, "speed": speed,
           "team": team}
    np.savez_compressed(cache, **out)
    return out


if __name__ == "__main__":
    trk = load_tracking(force="--force" in sys.argv)
    n = len(trk["ball"])
    vis = ~np.isnan(trk["ball"][:, 0])
    n_players = (~np.isnan(trk["players"][:, :, 0])).sum(axis=1)
    print("frames {}  ball visible {:.1%}  mean tracked players {:.1f}".format(
        n, vis.mean(), n_players.mean()))
    print("home slots {}  away slots {}".format(int((trk["team"] == 0).sum()),
                                                int((trk["team"] == 1).sum())))
    ball_s = trk["ball_s"][~np.isnan(trk["ball_s"])]
    if len(ball_s):
        print("ball speed column: {:.1%} non-NaN, median {:.2f}".format(
            len(ball_s) / n, float(np.median(ball_s))))
    else:
        print("ball speed column: all NaN (compute it from positions)")
