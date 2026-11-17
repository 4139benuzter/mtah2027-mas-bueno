"""Hybrid (calib/hybrid_rule.md): D from v31, sides from v35c. Held-out public test. Public data only."""
import json, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import possession_metrics as pm
from possession import learned_eval as LE, learned as LM, features, grid_dom as GD

D = 2

def one(job):
    name, seed, match = job
    d = np.load(LE.set_path(name, seed))
    trk, labels, info = GD.load(match)
    trk["ball"] = d["ball"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import possession.model as M
        M.KICKOFF_FRAMES = tuple(info["kickoffs"])
        feat = features.compute(trk, info.get("pitch", (105.0, 68.0)), 25, kickoffs=info["kickoffs"])
        eps = [tuple(e) for e in json.loads(str(d["eps"]))]
        prob = np.load(os.path.join(LE.OUT, "prob", "excl_{0}__{0}__s{1}.npy".format(name, seed))).astype(np.float64)
        gt = d["gt"]; graded = gt != pm.X
        p31 = np.asarray(d["path0"])
        p35 = LM.decode(prob, 0.0, feat, eps, LE.GRID[8])
        hyb = np.where(p31 == D, D, np.where(p35 != D, p35, p31))
        out = {}
        for tag, path in (("v31", p31), ("v35c", p35), ("hyb", hyb)):
            pred = np.full(len(gt), pm.X, dtype=np.int8); pred[graded] = path[graded]
            r = pm.evaluate(gt, pred)
            out[tag] = (r["spell_f1_mean"], r["macro_f1"])
    return name, seed, out

if __name__ == "__main__":
    names = LE.fine_names()
    jobs = [(n, s, LE.match_of(n)) for n in names for s in LE.SEEDS]
    with ProcessPoolExecutor(30) as pool:
        res = list(pool.map(one, jobs))
    per = {}
    for n, s, o in res:
        per.setdefault(n, []).append(o)
    g = {}
    print("match      v31 spell  v35c spell  hyb spell   gain   | v31 macro v35c macro hyb macro", flush=True)
    for n in names:
        a = {t: np.mean([o[t] for o in per[n]], axis=0) for t in ("v31", "v35c", "hyb")}
        g[n] = a["hyb"][0] - a["v31"][0]
        print("%-10s %9.4f %11.4f %10.4f %+8.4f | %9.4f %10.4f %9.4f" % (n, a["v31"][0], a["v35c"][0], a["hyb"][0], g[n], a["v31"][1], a["v35c"][1], a["hyb"][1]), flush=True)
    met = np.mean([g[n] for n in names if n.startswith("metrica")]); dfl = np.mean([g[n] for n in names if not n.startswith("metrica")])
    print("RULE1 mean gain %+.4f  RULE2 wins %d/10  RULE3 Metrica %+.4f DFL %+.4f" % (np.mean(list(g.values())), sum(v > 0 for v in g.values()), met, dfl), flush=True)
