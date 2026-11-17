"""Decoder settings re-chosen on public balls degraded to match the competition ball (v34).

The rule is calib/domain_decoder_rule.md (fixed before any result). Only labelled OTHER matches
are scored; the competition match and its public window are never read here. Works on Windows
(no fork): each job computes the features of one match in one ball condition once, then decodes
every setting exactly as possession/grid.py (and so possession/validate.py) does.

    python -m possession.grid_dom run [--jobs 14]          # GRID_DOM on the degraded ten matches
    python -m possession.grid_dom select                    # conditions 1-3 and the final setting
    python -m possession.grid_dom check [--jobs 14]         # conditions 4-6 for the final setting
"""

import glob
import itertools
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import possession.model as M  # noqa: E402
from possession import degrade, features, sample_game  # noqa: E402
from possession import grid as G  # noqa: E402
from possession.model import emissions, viterbi, merge_short_dead, snap_dead_onsets, gated_team_costs  # noqa: E402
from possession.validate import FPS, drop_ball  # noqa: E402

EXT = os.environ.get("MTAH_EXTERNAL", os.path.join(os.path.dirname(ROOT), "external"))
OUT = os.path.join(ROOT, "cache", "grid_dom")
CHOICE = os.path.join(ROOT, "calib", "domain_choice.json")
# First value of every list = v31.
GRID_DOM = {
    "radius": [1.5, 2.0, 2.5],
    "sigma_control": [1.5, 2.5],
    "min_contact": [2, 3, 5],
    "merge_gap": [5, 12],
    "carry_s": [0.6, 1.0],
    "gate_window_s": [0.2, 0.5],
    "switch_team": [15.0, 10.0, 6.0],
    "rel_slack": [2.0, 4.0],
}
V31 = {k: v[0] for k, v in GRID_DOM.items()}
CONDS = (("rec", None, 0), ("v59s0", 0.59, 0), ("v59s1", 0.59, 1))
# Amendment 1 of calib/domain_decoder_rule.md: the scored conditions are the degraded ball as
# recorded with three seeds of the degradation noise (no random removal: it cuts the stoppage
# holds before nohold sees them and makes in-play gaps the competition ball does not have).
CONDS_DOM = (("drec0", 0), ("drec1", 1), ("drec2", 2))
DOM_DIR = "dom2"
ALT = {"sub": {"k": 8, "h": 8, "noise": 0.4, "lag": 4}, "gauss": {"s": 6.27, "noise": 1.0, "lag": 2},
       "box": {"h": 8, "noise": 0.32, "lag": 1}, "kalman": {"q": 1.9, "r": 1.0, "noise": 0.3, "lag": 3}}
EXCLUDED_SC = {"SC2017461"}
NEAR_BEST = 0.003


def configs():
    return [dict(zip(GRID_DOM, values)) for values in itertools.product(*GRID_DOM.values())]


def fine_matches():
    return ["metrica1", "metrica2", "metrica3"] + sorted(glob.glob(os.path.join(EXT, "dfl-npz", "*.npz")))


def sc_matches():
    paths = sorted(glob.glob(os.path.join(EXT, "skillcorner-npz", "*.npz")))
    return [p for p in paths if os.path.basename(p)[:-4] not in EXCLUDED_SC]


def name_of(match):
    return match if match.startswith("metrica") else os.path.basename(match)[:-4]


def load(match):
    if match.startswith("metrica"):
        g = int(match[-1])
        folder = os.path.join(EXT, "metrica-sample-game3" if g == 3 else "metrica-sample-game12")
        return sample_game.load(folder, g)
    return sample_game.load(match)


def prepare(match, vis, seed, deg, dseed=0):
    """Features of one match in one ball condition. deg: 'dom', 'raw' or 'alt:<family>'.
    dseed: seed of the degradation noise (dom only)."""
    trk, labels, info = load(match)
    if deg == "dom":
        assert not vis, "amendment 1: no random removal before the domain degradation"
    if vis:
        trk["ball"] = drop_ball(trk["ball"], labels != "X", vis, seed)
    if deg == "dom":
        choice = json.load(open(CHOICE))
        trk["ball"] = degrade.apply(trk["ball"], choice["family"], choice["params"], seed=dseed)
    elif deg.startswith("alt:"):
        fam = deg[4:]
        trk["ball"] = degrade.apply(trk["ball"], fam, ALT[fam])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat = features.compute(trk, info.get("pitch", sample_game.PITCH), FPS, kickoffs=info["kickoffs"])
    return feat, labels, info


def score(cfg, feat, labels, info, gt):
    """spellF1 of one setting, decoded exactly as possession/grid.py decodes."""
    n = len(labels)
    graded = labels != "X"
    p = G.apply(cfg)
    M.KICKOFF_FRAMES = info["kickoffs"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_e, diag = emissions(feat, np.zeros(n), [], [], [], graded, n, FPS, p)
    team_costs = gated_team_costs(diag["eps"], n, FPS, p) if p.gate_switches else None
    path = merge_short_dead(viterbi(log_e, p.switch_team, p.switch_dead, team_costs), FPS, p)
    path = snap_dead_onsets(path, feat, FPS, p)
    pred = np.full(n, pm.X, dtype=np.int8)
    pred[graded] = path[graded]
    return float(pm.evaluate(gt, pred)["spell_f1_mean"])


def run_job(job):
    match, cond, vis, seed, deg, dseed, cfgs, out = job
    if os.path.isfile(out):
        return out
    feat, labels, info = prepare(match, vis, seed, deg, dseed)
    gt = pm.label_codes(labels)
    with open(out + ".tmp", "w") as handle:
        for cfg in cfgs:
            row = {"match": name_of(match), "cond": cond, "deg": deg, "cfg": cfg,
                   "spell_f1": score(cfg, feat, labels, info, gt)}
            handle.write(json.dumps(row) + "\n")
    os.replace(out + ".tmp", out)
    return out


def run_all(jobs, n_jobs):
    with ProcessPoolExecutor(n_jobs) as pool:
        for out in pool.map(run_job, jobs):
            print("done", os.path.relpath(out, ROOT), flush=True)


def key(cfg):
    return tuple(sorted(cfg.items()))


def table(deg, conds):
    """s[cfg_key][match] = mean spellF1 over the conditions; also the cfg dicts."""
    raw, cfgs = {}, {}
    for path in glob.glob(os.path.join(OUT, deg.replace(":", "_"), "*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["cond"] not in conds:
                continue
            k = key(r["cfg"])
            cfgs[k] = r["cfg"]
            raw.setdefault(k, {}).setdefault(r["match"], []).append(r["spell_f1"])
    s = {k: {m: float(np.mean(v)) for m, v in per.items() if len(v) == len(conds)} for k, per in raw.items()}
    return s, cfgs


def changes(cfg):
    return sum(cfg[k] != V31[k] for k in V31)


def pick(s, cfgs, subset):
    J = {k: np.mean([v[m] for m in subset]) for k, v in s.items()}
    best = max(J.values())
    near = [k for k in J if J[k] >= best - NEAR_BEST]
    return min(near, key=lambda k: (changes(cfgs[k]), -J[k])), J


def cmd_run(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    os.makedirs(os.path.join(OUT, DOM_DIR), exist_ok=True)
    cfgs = configs()
    assert cfgs[0] == V31 and len(cfgs) == 864
    jobs = [(m, c, None, 0, "dom", ds, cfgs, os.path.join(OUT, DOM_DIR, "{}__{}.jsonl".format(name_of(m), c)))
            for m in fine_matches() for c, ds in CONDS_DOM]
    run_all(jobs, n_jobs)


def cmd_select(_):
    s, cfgs = table(DOM_DIR, [c for c, _ in CONDS_DOM])
    matches = sorted({m for per in s.values() for m in per})
    assert len(matches) == 10, matches
    s = {k: v for k, v in s.items() if len(v) == 10}
    d = key(V31)
    print("{} settings complete on 10 matches x 3 degraded conditions".format(len(s)))
    gains = {}
    for m in matches:
        k, _ = pick(s, cfgs, [x for x in matches if x != m])
        gains[m] = s[k][m] - s[d][m]
        print("  held out {:10s} pick {}  gain {:+.4f}".format(
            m, {a: b for a, b in cfgs[k].items() if b != V31[a]}, gains[m]), flush=True)
    g = np.array([gains[m] for m in matches])
    met = np.array([gains[m] for m in matches if m.startswith("metrica")])
    dfl = np.array([gains[m] for m in matches if not m.startswith("metrica")])
    ok1, ok2, ok3 = g.mean() >= 0.010, (g > 0).sum() >= 8, met.mean() > 0 and dfl.mean() > 0
    final, J = pick(s, cfgs, matches)
    print("leave one match out: mean gain {:+.4f} ({}), wins {}/10 ({}), Metrica {:+.4f} DFL {:+.4f} ({})".format(
        g.mean(), ok1, (g > 0).sum(), ok2, met.mean(), dfl.mean(), ok3))
    print("final setting (pick on all ten): {}  J {:.4f} (v31 {:.4f}, best {:.4f})".format(
        {a: b for a, b in cfgs[final].items() if b != V31[a]}, J[final], J[d], max(J.values())))
    json.dump({"gains": gains, "final": cfgs[final], "J_final": J[final], "J_v31": J[d],
               "rule_1_3": bool(ok1 and ok2 and ok3)}, open(os.path.join(OUT, "select.json"), "w"), indent=1)
    print("RULE_1_3", "PASS" if ok1 and ok2 and ok3 else "FAIL")


def cmd_check(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    sel = json.load(open(os.path.join(OUT, "select.json")))
    final = sel["final"]
    if key(final) == key(V31):
        print("final setting is v31: nothing to adopt")
        print("RULE_4_6", "FAIL")
        return
    cfgs = [V31, final]
    jobs = []
    for fam in ALT:
        os.makedirs(os.path.join(OUT, "alt_" + fam), exist_ok=True)
        jobs += [(m, "rec", None, 0, "alt:" + fam, 0, cfgs, os.path.join(OUT, "alt_" + fam, name_of(m) + "__rec.jsonl"))
                 for m in fine_matches()]
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    jobs += [(m, c, v, s, "raw", 0, cfgs, os.path.join(OUT, "raw", "{}__{}.jsonl".format(name_of(m), c)))
             for m in fine_matches() for c, v, s in CONDS]
    os.makedirs(os.path.join(OUT, "sc"), exist_ok=True)
    jobs += [(m, "rec", None, 0, "sc", 0, cfgs, os.path.join(OUT, "sc", name_of(m) + "__rec.jsonl")) for m in sc_matches()]
    run_all(jobs, n_jobs)
    kv, kf = key(V31), key(final)

    def gain(deg, conds):
        s, _ = table(deg, conds)
        return {m: s[kf][m] - s[kv][m] for m in s[kv]}

    alt = {fam: np.mean(list(gain("alt:" + fam, ["rec"]).values())) for fam in ALT}
    ok4 = np.mean(list(alt.values())) > 0 and min(alt.values()) >= -0.005
    raw = np.mean(list(gain("raw", [c for c, _, _ in CONDS]).values()))
    ok5 = raw >= -0.010
    sc = np.array(list(gain("sc", ["rec"]).values()))
    g = sel["gains"]
    met = np.mean([v for m, v in g.items() if m.startswith("metrica")])
    dfl = np.mean([v for m, v in g.items() if not m.startswith("metrica")])
    three = np.mean([met, dfl, sc.mean()])
    ok6 = three >= 0.005
    print("4. other degradations, as recorded: {} -> mean {:+.4f}, min {:+.4f} ({})".format(
        {f: round(v, 4) for f, v in alt.items()}, np.mean(list(alt.values())), min(alt.values()), ok4))
    print("5. raw balls, three conditions: mean gain {:+.4f} ({})".format(raw, ok5))
    print("6. three groups: {:+.4f} (Metrica {:+.4f}, DFL {:+.4f}, SkillCorner {:+.4f}, wins {}/{}) ({})".format(
        three, met, dfl, sc.mean(), (sc > 0).sum(), len(sc), ok6))
    print("RULE_4_6", "PASS" if ok4 and ok5 and ok6 else "FAIL")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"run": cmd_run, "select": cmd_select, "check": cmd_check}[cmd](sys.argv[2:])
