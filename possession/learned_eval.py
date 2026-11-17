"""Held-out test of the learned emissions (v35) on labelled OTHER matches; rule in
calib/learned_rule.md (fixed before any result). Public data only; the competition match and its
public window are never read here. Training needs LightGBM (not needed to regenerate the CSV).

    python -m possession.learned_eval build  [--jobs 14]   # features of every match and seed
    python -m possession.learned_eval select [--jobs 14]   # nested leave-one-match-out, rule 1-3
    python -m possession.learned_eval final  [--jobs 14]   # model on all ten, SkillCorner, export
"""

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
from possession import degrade, features, learned as LM  # noqa: E402
from possession import grid_dom as GD  # noqa: E402

OUT = os.path.join(ROOT, "cache", "learned")
SEEDS = (0, 1, 2)
STRIDE = 10                       # training rows: every 10th graded frame, offset by the seed
LGB = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.1, "num_leaves": 31,
       "min_data_in_leaf": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
       "lambda_l2": 1.0, "max_bin": 63, "seed": 0, "deterministic": True, "verbose": -1}
ROUNDS = 120
GRID = [{"T": t, "switch_team": st, "switch_dead": sd, "gate": g}
        for t, st, sd, g in itertools.product((1.0, 0.5, 2.0), (8.0, 4.0, 15.0), (25.0, 10.0), (1, 0))]
CHOICE = json.load(open(os.path.join(ROOT, "calib", "domain_choice.json")))
THREADS = int(os.environ.get("LGB_THREADS", "14"))


def fine_names():
    return [GD.name_of(m) for m in GD.fine_matches()]


def set_path(name, seed):
    return os.path.join(OUT, "sets", "{}__s{}.npz".format(name, seed))


def build_one(job):
    match, seed, degrade_it = job
    name = GD.name_of(match)
    out = set_path(name, seed)
    if os.path.isfile(out):
        return out
    trk, labels, info = GD.load(match)
    if degrade_it:
        trk["ball"] = degrade.apply(trk["ball"], CHOICE["family"], CHOICE["params"], seed=seed)
    graded = labels != "X"
    pitch = info.get("pitch", (105.0, 68.0))
    halves = [(a, b) for _, a, b in info["halves"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feat, log_e0, eps, share, kind, path0 = LM.v31_inputs(trk, pitch, info["kickoffs"], graded)
        f = LM.frame_features(trk, feat, log_e0, path0, share, kind, eps, pitch, halves)
    names = sorted(f)
    X = np.stack([f[k] for k in names], axis=1)
    np.savez_compressed(out, X=X, names=np.array(names), gt=pm.label_codes(labels).astype(np.int8),
                        path0=path0.astype(np.int8), ball=trk["ball"],
                        eps=json.dumps([[int(s), int(e), int(k), kind, int(u)] for s, e, k, kind, u in eps]))
    return out


def cmd_build(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    os.makedirs(os.path.join(OUT, "sets"), exist_ok=True)
    jobs = [(m, s, True) for m in GD.fine_matches() for s in SEEDS]
    jobs += [(m, 0, False) for m in GD.sc_matches()]
    with ProcessPoolExecutor(n_jobs) as pool:
        for out in pool.map(build_one, jobs):
            print("built", os.path.basename(out), flush=True)


def load_set(name, seed):
    d = np.load(set_path(name, seed))
    return {"X": d["X"], "names": [str(x) for x in d["names"]], "gt": d["gt"], "path0": d["path0"]}


def train(names_in):
    """LightGBM booster trained on the given matches (all seeds), with team-swap augmentation."""
    import lightgbm as lgb
    key = "excl_" + "_".join(sorted(set(fine_names()) - set(names_in))) if set(names_in) != set(fine_names()) else "all"
    path = os.path.join(OUT, "models", key + ".txt")
    if os.path.isfile(path):
        return lgb.Booster(model_file=path)
    Xs, ys = [], []
    for name in names_in:
        for seed in SEEDS:
            d = load_set(name, seed)
            frames = np.arange(len(d["gt"]))
            rows = (d["gt"] != pm.X) & (frames % STRIDE == seed % STRIDE)
            X, y = d["X"][rows], d["gt"][rows].astype(np.int64)
            Xs += [X, LM.swap_teams(X, d["names"])]
            ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
    params = dict(LGB, num_threads=THREADS)
    booster = lgb.train(params, lgb.Dataset(np.vstack(Xs), np.concatenate(ys)), num_boost_round=ROUNDS)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)
    return booster


def predict_sets(booster, name, tag, seeds=SEEDS):
    """Save P for every seed of a match under cache/learned/prob/<tag>__<name>__s<seed>.npy."""
    for seed in seeds:
        out = os.path.join(OUT, "prob", "{}__{}__s{}.npy".format(tag, name, seed))
        if not os.path.isfile(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, booster.predict(load_set(name, seed)["X"]).astype(np.float32))


def decode_job(job):
    """spellF1 for every GRID setting (and v31's own path) on one match and seed."""
    name, seed, prob_path, match = job
    out = prob_path.replace(".npy", ".scores.json")
    if os.path.isfile(out):
        return out
    d = np.load(set_path(name, seed))
    trk, labels, info = GD.load(match)
    trk["ball"] = d["ball"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import possession.model as M
        M.KICKOFF_FRAMES = tuple(info["kickoffs"])
        feat = features.compute(trk, info.get("pitch", (105.0, 68.0)), 25, kickoffs=info["kickoffs"])
        eps = [tuple(e) for e in json.loads(str(d["eps"]))]
        prob = np.load(prob_path).astype(np.float64)
        gt = d["gt"]
        graded = gt != pm.X
        res = {}
        for i, cfg in enumerate(GRID):
            path = LM.decode(prob, 0.0, feat, eps, cfg)
            pred = np.full(len(gt), pm.X, dtype=np.int8)
            pred[graded] = path[graded]
            res[i] = pm.evaluate(gt, pred)["spell_f1_mean"]
        pred = np.full(len(gt), pm.X, dtype=np.int8)
        pred[graded] = d["path0"][graded]
        res["v31"] = pm.evaluate(gt, pred)["spell_f1_mean"]
    json.dump(res, open(out, "w"))
    return out


def run_decodes(jobs, n_jobs):
    with ProcessPoolExecutor(n_jobs) as pool:
        list(pool.map(decode_job, jobs))


def match_of(name):
    return {GD.name_of(m): m for m in GD.fine_matches() + GD.sc_matches()}[name]


def scores(tag, name, seeds=SEEDS):
    """{grid index or 'v31': mean spellF1 over seeds} for one match under one model tag."""
    per = [json.load(open(os.path.join(OUT, "prob", "{}__{}__s{}.scores.json".format(tag, name, s))))
           for s in seeds]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def pick(table):
    """Grid index with the best mean over the matches in table (list of score dicts); ties to the
    earlier grid entry (T 1, v31's switch costs and gate come first)."""
    J = [np.mean([t[str(i)] for t in table]) for i in range(len(GRID))]
    return int(np.argmax(J)), J


def cmd_select(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    names = fine_names()
    assert len(names) == 10
    for a, b in itertools.combinations(names, 2):          # models without {a, b}
        tag = "excl_" + "_".join(sorted((a, b)))
        booster = train([x for x in names if x not in (a, b)])
        predict_sets(booster, a, tag)
        predict_sets(booster, b, tag)
        print("pair model", tag, flush=True)
    for m in names:                                         # models without m
        booster = train([x for x in names if x != m])
        predict_sets(booster, m, "excl_" + m)
        print("single model excl_" + m, flush=True)
    jobs = []
    for a, b in itertools.combinations(names, 2):
        tag = "excl_" + "_".join(sorted((a, b)))
        for x in (a, b):
            jobs += [(x, s, os.path.join(OUT, "prob", "{}__{}__s{}.npy".format(tag, x, s)), match_of(x)) for s in SEEDS]
    jobs += [(m, s, os.path.join(OUT, "prob", "excl_{}__{}__s{}.npy".format(m, m, s)), match_of(m))
             for m in names for s in SEEDS]
    run_decodes(jobs, n_jobs)
    gains, picks = {}, {}
    for m in names:
        inner = [scores("excl_" + "_".join(sorted((m, x))), x) for x in names if x != m]
        i, _ = pick(inner)
        s = scores("excl_" + m, m)
        gains[m] = s[str(i)] - s["v31"]
        picks[m] = GRID[i]
        print("  held out {:9s} decode {}  v31 {:.4f}  v35 {:.4f}  gain {:+.4f}".format(
            m, GRID[i], s["v31"], s[str(i)], gains[m]), flush=True)
    g = np.array([gains[m] for m in names])
    met = np.array([gains[m] for m in names if m.startswith("metrica")])
    dfl = np.array([gains[m] for m in names if not m.startswith("metrica")])
    ok1, ok2, ok3 = g.mean() >= 0.020, (g > 0).sum() >= 8, met.mean() > 0 and dfl.mean() > 0
    print("leave one match out: mean gain {:+.4f} ({}), wins {}/10 ({}), Metrica {:+.4f} DFL {:+.4f} ({})".format(
        g.mean(), ok1, (g > 0).sum(), ok2, met.mean(), dfl.mean(), ok3))
    json.dump({"gains": gains, "picks": picks, "rule_1_3": bool(ok1 and ok2 and ok3)},
              open(os.path.join(OUT, "select.json"), "w"), indent=1)
    print("RULE_1_3", "PASS" if ok1 and ok2 and ok3 else "FAIL")


# --- Amendment 1 of calib/learned_rule.md: stress test with gaps in live play -----------------------
# The competition ball is missing on 9.2% of graded frames where more than 20% of players run
# (label-free); the degraded DFL balls on 2-4%. A model that learned "missing ball = dead ball"
# would lose its gain there. Each held-out match gets extra gaps (0.4-3 s) on such frames until
# it reaches the competition's rate; v35 (held-out model, its own decode pick) and v31 are both
# re-run on that ball.
STRESS_RATE = 0.092


def live_gaps(ball, run, graded, rate, seed):
    rng = np.random.default_rng(1000 + seed)
    b = ball.copy()
    live = graded & (run > 0.2)
    target = rate * live.sum()
    missing = int((live & np.isnan(b[:, 0])).sum())
    cand = np.flatnonzero(live & np.isfinite(b[:, 0]))
    while missing < target and len(cand):
        s = int(rng.choice(cand))
        seg = slice(s, min(len(b), s + int(rng.integers(10, 76))))
        missing += int((live[seg] & np.isfinite(b[seg, 0])).sum())
        b[seg] = np.nan
    return b


# v35c (calib/learned_c_rule.md): one decoding setting fixed in advance, instead of the pick.
FIXED = os.environ.get("FIXED_INDEX")


def stress_dir():
    return "stress" if FIXED is None else "stress_fix" + FIXED


def stress_job(job):
    match, seed = job
    name = GD.name_of(match)
    out = os.path.join(OUT, stress_dir(), "{}__s{}.json".format(name, seed))
    if os.path.isfile(out):
        return out
    import lightgbm as lgb
    d = np.load(set_path(name, seed))
    trk, labels, info = GD.load(match)
    graded = labels != "X"
    pitch = info.get("pitch", (105.0, 68.0))
    halves = [(a, b) for _, a, b in info["halves"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trk["ball"] = d["ball"]
        run = np.nan_to_num(features.compute(trk, pitch, 25, kickoffs=info["kickoffs"])["frac_run"], nan=0.0)
        trk["ball"] = live_gaps(d["ball"], run, graded, STRESS_RATE, seed)
        feat, log_e0, eps, share, kind, path0 = LM.v31_inputs(trk, pitch, info["kickoffs"], graded)
        f = LM.frame_features(trk, feat, log_e0, path0, share, kind, eps, pitch, halves)
    X = np.stack([f[k] for k in sorted(f)], axis=1)
    booster = lgb.Booster(model_file=os.path.join(OUT, "models", "excl_" + name + ".txt"))
    prob = booster.predict(X, num_threads=2)
    cfg = (json.load(open(os.path.join(OUT, "select.json")))["picks"][name] if FIXED is None
           else GRID[int(FIXED)])
    path = LM.decode(prob, 0.0, feat, eps, cfg)
    gt = pm.label_codes(labels)
    res = {}
    for tag, p in (("v35", path), ("v31", path0)):
        pred = np.full(len(gt), pm.X, dtype=np.int8)
        pred[graded] = p[graded]
        res[tag] = pm.evaluate(gt, pred)["spell_f1_mean"]
    live = graded & (run > 0.2)
    res["live_missing"] = float(np.isnan(trk["ball"][live, 0]).mean())
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    return out


def cmd_stress(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    jobs = [(m, s) for m in GD.fine_matches() for s in SEEDS]
    with ProcessPoolExecutor(n_jobs) as pool:
        list(pool.map(stress_job, jobs))
    gains, rates = {}, {}
    for m in GD.fine_matches():
        name = GD.name_of(m)
        per = [json.load(open(os.path.join(OUT, stress_dir(), "{}__s{}.json".format(name, s)))) for s in SEEDS]
        gains[name] = float(np.mean([p["v35"] - p["v31"] for p in per]))
        rates[name] = float(np.mean([p["live_missing"] for p in per]))
    json.dump({"gains": gains, "live_missing": rates}, open(os.path.join(OUT, stress_dir() + ".json"), "w"), indent=1)
    print("stress written")


def cmd_fixed(args):
    """Conditions 1-4b for one GRID setting fixed in advance (v35c), from the saved scores:
    held-out match m predicted by the model trained on the other nine; SkillCorner by the model
    trained on all ten. No selection happens here."""
    i = int(FIXED)
    names = fine_names()
    gains = {m: scores("excl_" + m, m)[str(i)] - scores("excl_" + m, m)["v31"] for m in names}
    sc = [GD.name_of(m) for m in GD.sc_matches()]
    d_sc = {n: scores("all", n, seeds=(0,))[str(i)] - scores("all", n, seeds=(0,))["v31"] for n in sc}
    met = np.mean([v for m, v in gains.items() if m.startswith("metrica")])
    dfl = np.mean([v for m, v in gains.items() if not m.startswith("metrica")])
    json.dump({"cfg": GRID[i], "gains": gains, "sc_gains": d_sc, "metrica": met, "dfl": dfl,
               "sc": float(np.mean(list(d_sc.values()))),
               "three": float(np.mean([met, dfl, np.mean(list(d_sc.values()))]))},
              open(os.path.join(OUT, "fixed_{}.json".format(i)), "w"), indent=1)
    print("fixed written")


def export(booster, cfg, names):
    dump = booster.dump_model()
    return {"features": names, "num_class": 3, "trees": dump["tree_info"], "decode": cfg,
            "lightgbm": {"params": LGB, "rounds": ROUNDS}}


def cmd_final(args):
    n_jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 14
    names = fine_names()
    table = [scores("excl_" + m, m) for m in names]          # leave-one-match-out predictions
    i, J = pick(table)
    if FIXED is not None:                                     # v35c: FIXED_INDEX=8, v31's own setting
        i = int(FIXED)                                        # (calib/learned_c_rule.md); nothing picked
    cfg = GRID[i]
    print("decode on all ten (leave-one-out predictions):", cfg, "J {:.4f}".format(J[i]),
          "v31 {:.4f}".format(np.mean([t["v31"] for t in table])))
    booster = train(names)
    sc = [GD.name_of(m) for m in GD.sc_matches()]
    for name in sc:
        predict_sets(booster, name, "all", seeds=(0,))
    run_decodes([(n, 0, os.path.join(OUT, "prob", "all__{}__s0.npy".format(n)), match_of(n)) for n in sc], n_jobs)
    d_sc = np.array([scores("all", n, seeds=(0,))[str(i)] - scores("all", n, seeds=(0,))["v31"] for n in sc])
    sel = json.load(open(os.path.join(OUT, "select.json")))
    g = sel["gains"]
    met = np.mean([g[m] for m in names if m.startswith("metrica")])
    dfl = np.mean([g[m] for m in names if not m.startswith("metrica")])
    three = np.mean([met, dfl, d_sc.mean()])
    print("SkillCorner ({}): mean gain {:+.4f}, wins {}/{}".format(len(sc), d_sc.mean(), (d_sc > 0).sum(), len(sc)))
    print("three groups: {:+.4f} (Metrica {:+.4f}, DFL {:+.4f}, SkillCorner {:+.4f})".format(three, met, dfl, d_sc.mean()))
    print("RULE_4", "PASS" if three >= 0.005 else "FAIL")
    model = export(booster, cfg, load_set(names[0], 0)["names"])
    json.dump(model, open(os.path.join(ROOT, "calib", "learned_model.json"), "w"))
    X = load_set(names[0], 0)["X"][:20000]
    diff = np.abs(LM.predict_json(json.load(open(os.path.join(ROOT, "calib", "learned_model.json"))), X)
                  - booster.predict(X)).max()
    print("numpy inference vs LightGBM: max abs difference {:.2e}".format(diff))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"build": cmd_build, "select": cmd_select, "final": cmd_final, "stress": cmd_stress,
     "fixed": cmd_fixed}[cmd](sys.argv[2:])
