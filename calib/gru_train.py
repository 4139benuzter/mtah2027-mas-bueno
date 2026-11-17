"""Train the BiGRU on all ten public matches (same recipe as the held-out test) and export it for numpy inference."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
import gru_lomo as G
from possession import learned_eval as LE, learned as LM

names = LE.fine_names()
torch.manual_seed(0); np.random.seed(0)
seqs = []
for n in names:
    for s in LE.SEEDS:
        X, y, nm = G.load(n, s)
        seqs.append((X, y))
        seqs.append((LM.swap_teams(X, nm).astype(np.float32), np.where(y == 0, 1, np.where(y == 1, 0, y))))
allX = np.concatenate([q[0] for q in seqs]); mu = np.nanmean(allX, 0); sd = np.nanstd(allX, 0) + 1e-6; del allX
prep = lambda X: np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
T = [(torch.from_numpy(prep(X)), torch.from_numpy(y)) for X, y in seqs]
class Net(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f, G.HID), nn.ReLU())
        self.gru = nn.GRU(G.HID, G.HID, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * G.HID, 3)
    def forward(self, x):
        h, _ = self.gru(self.inp(x)); return self.out(h)
dev = "cuda"
net = Net(T[0][0].shape[1]).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=G.STEPS)
lossf = nn.CrossEntropyLoss(ignore_index=-1); rng = np.random.default_rng(0)
for step in range(G.STEPS):
    xb, yb = [], []
    for _ in range(G.BATCH):
        X, y = T[rng.integers(len(T))]; a = int(rng.integers(0, len(y) - G.WIN))
        xb.append(X[a:a + G.WIN]); yy = y[a:a + G.WIN].clone(); yy[yy > 2] = -1; yb.append(yy)
    xb = torch.stack(xb).to(dev); yb = torch.stack(yb).to(dev)
    loss = lossf(net(xb).reshape(-1, 3), yb.reshape(-1)); opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
net.eval().cpu()
out = {k: v.detach().numpy() for k, v in net.state_dict().items()}
out.update(mu=mu, sd=sd, win=np.array(G.WIN), layers=np.array(2))
np.savez(os.path.join("calib", "gru_model.npz"), **out)
# numpy vs torch check on one window batch
from possession import gru as GR
m = GR.load(os.path.join("calib", "gru_model.npz"))
X, _, _ = G.load(names[0], 0); Xs = X[:3000]
with torch.no_grad():
    Z = torch.from_numpy(prep(Xs))
    pt = torch.softmax(net(Z[None, :G.WIN]), -1)[0].numpy()
pn = GR._forward(m, np.nan_to_num((Xs[:G.WIN] - m["mu"]) / m["sd"])[None])[0]
print("final loss %.3f  numpy vs torch max abs diff %.2e" % (loss.item(), np.abs(pt - pn).max()))
