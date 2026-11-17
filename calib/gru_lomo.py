"""BiGRU over v35c's 67 per-frame features; leave-one-match-out on the ten public matches; public data only.
Outputs held-out spellF1 for: v31, GRU (decoded like v35c, GRID[8]), GRU-hybrid (D from v31, sides from GRU),
and v35c-hybrid for reference."""
import json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import possession_metrics as pm
from possession import learned_eval as LE, learned as LM, features, grid_dom as GD

D = 2
WIN, BATCH, STEPS, HID = 750, 48, 800, 64
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gru_out"); os.makedirs(OUTD, exist_ok=True)


def load(name, seed):
    d = np.load(LE.set_path(name, seed), allow_pickle=True)
    return d["X"].astype(np.float32), d["gt"].astype(np.int64), list(d["names"])


def train_fold(train_names, test_name, dev):
    import torch, torch.nn as nn
    torch.manual_seed(0); np.random.seed(0)
    seqs = []
    for n in train_names:
        for s in LE.SEEDS:
            X, y, names = load(n, s)
            seqs.append((X, y))
            Xs = LM.swap_teams(X, names).astype(np.float32)
            ys = np.where(y == 0, 1, np.where(y == 1, 0, y))
            seqs.append((Xs, ys))
    allX = np.concatenate([q[0] for q in seqs])
    mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6
    del allX
    def prep(X):
        Z = (X - mu) / sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seqs = [(torch.from_numpy(prep(X)), torch.from_numpy(y)) for X, y in seqs]

    class Net(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(f, HID), nn.ReLU())
            self.gru = nn.GRU(HID, HID, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * HID, 3)
        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    net = Net(seqs[0][0].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=STEPS)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    rng = np.random.default_rng(0)
    for step in range(STEPS):
        xb, yb = [], []
        for _ in range(BATCH):
            X, y = seqs[rng.integers(len(seqs))]
            a = int(rng.integers(0, len(y) - WIN))
            xb.append(X[a:a + WIN]); yy = y[a:a + WIN].clone(); yy[yy > 2] = -1; yb.append(yy)
        xb = torch.stack(xb).to(dev); yb = torch.stack(yb).to(dev)
        loss = lossf(net(xb).reshape(-1, 3), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        for s in LE.SEEDS:
            X, _, _ = load(test_name, s)
            Z = torch.from_numpy(prep(X)); n = Z.shape[0]; S = WIN // 2
            starts = list(range(0, max(n - WIN, 0) + 1, S))
            if starts[-1] + WIN < n: starts.append(n - WIN)
            acc = np.zeros((n, 3), dtype=np.float64); cnt = np.zeros(n)
            for b in range(0, len(starts), 256):
                st = starts[b:b + 256]
                xb = torch.stack([Z[a:a + WIN] for a in st]).to(dev)
                pb = torch.softmax(net(xb), -1).float().cpu().numpy()
                for a, q in zip(st, pb):
                    acc[a:a + WIN] += q; cnt[a:a + WIN] += 1
            p = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
            np.save(os.path.join(OUTD, "gru_excl_%s__%s__s%d.npy" % (test_name, test_name, s)), p)
    return float(loss.item())


def score(job):
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
        gt = d["gt"]; graded = gt != pm.X
        p31 = np.asarray(d["path0"])
        pl = np.load(os.path.join(LE.OUT, "prob", "excl_{0}__{0}__s{1}.npy".format(name, seed))).astype(np.float64)
        pg = np.load(os.path.join(OUTD, "gru_excl_{0}__{0}__s{1}.npy".format(name, seed))).astype(np.float64)
        p35 = LM.decode(pl, 0.0, feat, eps, LE.GRID[8])
        pgr = LM.decode(np.clip(pg, 1e-6, 1), 0.0, feat, eps, LE.GRID[8])
        out = {}
        for tag, path in (("v31", p31), ("hyb35", np.where(p31 == D, D, np.where(p35 != D, p35, p31))),
                          ("gru", pgr), ("hybgru", np.where(p31 == D, D, np.where(pgr != D, pgr, p31)))):
            pred = np.full(len(gt), pm.X, dtype=np.int8); pred[graded] = path[graded]
            out[tag] = pm.evaluate(gt, pred)["spell_f1_mean"]
    return name, seed, out


if __name__ == "__main__":
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, torch.__version__, flush=True)
    names = LE.fine_names()
    t0 = time.time()
    for m in names:
        loss = train_fold([n for n in names if n != m], m, dev)
        print("fold %s done, last loss %.3f, %.0f s" % (m, loss, time.time() - t0), flush=True)
    jobs = [(n, s, LE.match_of(n)) for n in names for s in LE.SEEDS]
    with ProcessPoolExecutor(16) as pool:
        res = list(pool.map(score, jobs))
    per = {}
    for n, s, o in res:
        per.setdefault(n, []).append(o)
    tags = ("v31", "hyb35", "gru", "hybgru")
    print("match      " + "  ".join("%8s" % t for t in tags), flush=True)
    tot = {t: [] for t in tags}
    for n in names:
        a = {t: np.mean([o[t] for o in per[n]]) for t in tags}
        for t in tags: tot[t].append(a[t])
        print("%-10s " % n + "  ".join("%8.4f" % a[t] for t in tags), flush=True)
    print("MEAN       " + "  ".join("%8.4f" % np.mean(tot[t]) for t in tags), flush=True)
    for t in ("gru", "hybgru"):
        g = np.array(tot[t]) - np.array(tot["hyb35"])
        print("%s vs hyb35: mean %+.4f, wins %d/10" % (t, g.mean(), (g > 0).sum()), flush=True)
