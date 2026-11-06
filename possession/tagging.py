"""Smart Tagging -> per-frame priors.

Each instance is a phase (BUILD UP, PROGRESSION, ...) or a set piece, with a Team tag. Phases are
not possessions, so they are used as a soft prior on the team, never as a label.

    from possession.tagging import load_instances, team_prior, set_pieces, goals
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

HOME_NAME = "barcelona"
AWAY_NAME = "atletic club"

# Codes where Team is the side with the ball, and codes where Team is the side without it.
OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition", "long ball",
             "ball in final third", "ball in the box", "players in the box"}
DEFENSIVE = {"defending in attacking third", "defending in middle third",
             "defending in defensive third", "defensive transition"}


def load_instances(game=GAME):
    fps = GAMES[game]["fps"]
    path = require(["smart_tagging"], game)[0]
    out = []
    for inst in ET.parse(path).iter("instance"):
        start_s = float(inst.findtext("start"))
        end_s = float(inst.findtext("end"))
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().lower(),
            "start": int(round(start_s * fps)),
            "end": int(round(end_s * fps)),
            "team": _side(tags.get("Team")),
            "type": (tags.get("Type") or "").strip().lower(),
            "half": tags.get("Half"),
        })
    return out


def _side(name):
    """'H', 'A' or None from the club name in the Team tag."""
    if not name:
        return None
    name = name.strip().lower()
    if name == HOME_NAME:
        return "H"
    if name == AWAY_NAME:
        return "A"
    return None


def team_prior(instances, n_frames):
    """(n,) in [-1, 1]: +1 favours home possession, -1 away, 0 no information."""
    prior = np.zeros(n_frames, dtype=np.float64)
    for inst in instances:
        if inst["team"] is None:
            continue
        sign = 1.0 if inst["team"] == "H" else -1.0
        if inst["code"] in OFFENSIVE:
            pass
        elif inst["code"] in DEFENSIVE:
            sign = -sign
        else:
            continue
        a, b = max(0, inst["start"]), min(n_frames, inst["end"])
        prior[a:b] += sign
    return np.clip(prior, -1.0, 1.0)


def set_pieces(instances):
    """[(start, end, team, type)] for every SET PIECES instance, team may be None."""
    return [(i["start"], i["end"], i["team"], i["type"])
            for i in instances if i["code"] == "set pieces"]


def transitions(instances):
    """[(start, end, gaining_team)] from ATTACKING / DEFENSIVE TRANSITION tags."""
    out = []
    for i in instances:
        if i["team"] is None:
            continue
        if i["code"] == "attacking transition":
            out.append((i["start"], i["end"], i["team"]))
        elif i["code"] == "defensive transition":
            out.append((i["start"], i["end"], "A" if i["team"] == "H" else "H"))
    return out


def goals(instances):
    return [(i["start"], i["end"], i["team"]) for i in instances if i["code"] == "goals"]


if __name__ == "__main__":
    inst = load_instances()
    n = GAMES[GAME]["n_video_frames"]
    prior = team_prior(inst, n)
    print("instances {}  frames with team prior {:.1%}  (home-leaning {:.1%}, away-leaning {:.1%})"
          .format(len(inst), (prior != 0).mean(), (prior > 0).mean(), (prior < 0).mean()))
    sp = set_pieces(inst)
    types = {}
    for _, _, team, kind in sp:
        types[kind] = types.get(kind, 0) + 1
    print("set pieces {}: {}".format(len(sp), types))
    print("set pieces with a team: {}".format(sum(1 for s in sp if s[2] is not None)))
    print("goals: {}".format([(a, b, t) for a, b, t in goals(inst)]))
