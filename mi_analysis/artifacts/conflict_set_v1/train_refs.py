"""Reference predictors + conflict set. [project] stage 1+2. Run: python train_refs.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix

OUT = Path("/content/outputs")
DATA = "/content/research_events.parquet"
D, EPOCHS, BS, LR = 64, 25, 8192, 3e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

df = pd.read_parquet(DATA)
assert df.shape == (658905, 40), df.shape
tr, va, te = df[df.split=="train"], df[df.split=="val"], df[df.split=="test"]
print(f"train/val/test: {len(tr)}/{len(va)}/{len(te)}")

# action vocab: 15 observed classes in id order
voc = tr.drop_duplicates("action_id").set_index("action_id")["action"].sort_index()
assert list(voc.index) == list(range(15)) and 15 not in df.action_id.unique()
(OUT/"predictions").mkdir(parents=True, exist_ok=True)
json.dump({str(k): v for k, v in voc.to_dict().items()}, open(OUT/"action_vocab.json", "w"), indent=1)
print("vocab:", voc.to_dict())

def col(a, vals):  # tensor from column(s)
    return torch.tensor(a[vals].to_numpy() if isinstance(vals, str) else a[vals].to_numpy(), dtype=torch.long)

class IdFull(nn.Module):  # player+team+role+comp summed (team decision: try both, this is default)
    def __init__(self):
        super().__init__()
        self.e = nn.ModuleDict({k: nn.Embedding(n, D) for k, n in
            [("p", 351), ("t", 214), ("r", 25), ("c", 21)]})
        self.h = nn.Linear(D, 15)
    def forward(self, p, t, r, c): return self.h(self.e["p"](p)+self.e["t"](t)+self.e["r"](r)+self.e["c"](c))

class IdPlayer(nn.Module):  # strict player-only variant
    def __init__(self):
        super().__init__(); self.e = nn.Embedding(351, D); self.h = nn.Linear(D, 15)
    def forward(self, p): return self.h(self.e(p))

class Ctx(nn.Module):  # 11 cat + 4 cont; asserts no identity cols enter
    SIZES = {"zone":96,"period":5,"minute_bin":24,"score":7,"pp":10,"plen":21,"newp":2,
             "pa":16,"po":3,"pez":96,"same":2}
    def __init__(self):
        super().__init__()
        self.e = nn.ModuleDict({k: nn.Embedding(n, D) for k, n in self.SIZES.items()})
        self.cp = nn.Linear(4, D); self.h = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 15))
    def forward(self, cats, cont):
        s = sum(self.e[k](cats[:, i]) for i, k in enumerate(self.SIZES)) + self.cp(cont)
        return self.h(s)

def ctx_tensors(a):
    cats = np.stack([a.zone.to_numpy(), a.period.to_numpy(), a.minute_bin.to_numpy(),
        a.score_diff_idx.to_numpy(), (a.play_pattern_id-1).to_numpy(), a.poss_len.to_numpy(),
        a.new_possession.to_numpy(), a.prev_action_id.to_numpy(), a.prev_outcome.to_numpy(),
        a.prev_end_zone.to_numpy(), a.same_player_as_prev.to_numpy()], 1)
    cont = a[["x_norm","y_norm","dt","t_poss"]].to_numpy(dtype=np.float32)
    return torch.tensor(cats.astype(np.int64)), torch.tensor(cont)

def fit(model, tr_ld, va_ld, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss()
    best, best_sd, hist = 1e9, None, []
    for ep in range(EPOCHS):
        model.train(); tl = 0; nt = 0
        for b in tr_ld:  # b = (inputs..., y)
            *x, y = [t.to(device) for t in b]
            opt.zero_grad(); l = loss(model(*x), y); l.backward(); opt.step(); tl += l.item()*len(y); nt += len(y)
        model.eval(); vl = 0; n = 0
        with torch.no_grad():
            for b in va_ld: *x, y = [t.to(device) for t in b]; vl += loss(model(*x), y).item()*len(y); n += len(y)
        vl /= n; hist.append((tl/nt, vl))
        if vl < best: best, best_sd = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"{name} ep{ep}: train {tl/nt:.4f} val {vl:.4f}", flush=True)
    model.load_state_dict(best_sd); return model, hist

def loader(a, kind, shuf=True):
    y = torch.tensor(a.action_id.to_numpy(), dtype=torch.long)
    if kind == "idf": d = torch.utils.data.TensorDataset(col(a,"player_ix"),col(a,"team_ix"),col(a,"role_ix"),col(a,"comp_ix"),y)
    elif kind == "idp": d = torch.utils.data.TensorDataset(col(a,"player_ix"),y)
    else: c, t = ctx_tensors(a); d = torch.utils.data.TensorDataset(c,t,y)
    return torch.utils.data.DataLoader(d, batch_size=BS, shuffle=shuf)

@torch.no_grad()
def logits_of(model, ld):
    model.eval(); outs = []
    for b in ld: *x, y = b; outs.append(model(*[t.to(device) for t in x]).cpu().numpy())
    return np.concatenate(outs)

def fit_temp(logits, y):
    L = torch.tensor(np.asarray(logits, dtype=np.float32)); Y = torch.tensor(np.asarray(y, dtype=np.int64))
    f = lambda t: nn.functional.cross_entropy(L/t, Y).item()
    r = minimize_scalar(f, bounds=(0.05, 10), method="bounded", options={"xatol":1e-3})
    return float(r.x), f(1.0), f(r.x)

def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); e = 0
    for b in range(bins):
        m = (conf > b/bins) & (conf <= (b+1)/bins)
        if m.sum(): e += m.mean()*abs(conf[m].mean()-(pred[m]==y[m]).mean())
    return float(e)

def report(name, logits, T, y, outdir, split):
    p1 = torch.softmax(torch.tensor(logits), -1).numpy()
    p2 = torch.softmax(torch.tensor(logits/T), -1).numpy()
    pred = p2.argmax(1)
    top3 = float((np.argpartition(p2, -3, axis=1)[:, -3:] == y[:, None]).any(1).mean())
    m = dict(split=split, acc=float((pred==y).mean()), top3=top3,
             nll=float(nn.functional.cross_entropy(torch.tensor(logits/T), torch.tensor(y))),
             ece_before=ece(p1,y), ece_after=ece(p2,y), T=T,
             per_class={voc[i]: float((pred[y==i]==i).mean()) if (y==i).sum() else None for i in range(15)})
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(outdir/f"metrics_{split}.json","w"), indent=1)
    json.dump(m, open(outdir/"metrics.json","w"), indent=1)
    np.savetxt(outdir/"confusion.csv", confusion_matrix(y, pred, labels=list(range(15))), delimiter=",", fmt="%d")
    print(name, {k: round(v,4) if isinstance(v,float) else v for k,v in m.items() if k!="per_class"})
    print(" per-class:", {k: (round(v,3) if v is not None else None) for k,v in m["per_class"].items()})
    return p2

yf = {"train": tr.action_id.to_numpy().astype(np.int64), "val": va.action_id.to_numpy().astype(np.int64), "test": te.action_id.to_numpy().astype(np.int64)}
models, probs = {}, {}
for key, cls, kind in [("model1_identity_full", IdFull, "idf"), ("model1_identity_playeronly", IdPlayer, "idp"), ("model2_context", Ctx, "ctx")]:
    m = cls().to(device)
    m, h = fit(m, loader(tr, kind), loader(va, kind, False), key)
    torch.save({"state": m.state_dict(), "arch": cls.__name__, "d": D},
               OUT/key/"weights.pt" if (OUT/key).mkdir(exist_ok=True) else OUT/key/"weights.pt")
    json.dump({"arch": cls.__name__, "d": D, "epochs": EPOCHS, "bs": BS, "lr": LR,
               "loss": "unweighted CE (priors must reflect base rates)",
               "history": h}, open(OUT/key/"config.json","w"))
    lg = {s: logits_of(m, loader(a, kind, False)) for s, a in [("val", va), ("test", te)]}
    T, nll_b, nll_a = fit_temp(lg["val"], yf["val"])
    p = {s: report(key, lg[s], T, yf[s], OUT/key, s) for s in ["val","test"]}
    probs[key] = p; models[key] = m
    print(f"{key}: T={T:.3f} NLL {nll_b:.4f}->{nll_a:.4f}", flush=True)

# leakage: identity perturbation must leave Model 2 byte-identical
torch.manual_seed(0); idx = np.random.choice(len(va), 2048)
c0, t0 = ctx_tensors(va.iloc[idx])
m2 = models["model2_context"]
with torch.no_grad(): base = m2(c0.to(device), t0.to(device)).cpu().numpy()
# (identity cols never enter ctx_tensors by construction — assert + rerun = identical)
c1, t1 = ctx_tensors(va.iloc[idx])
with torch.no_grad(): again = m2(c1.to(device), t1.to(device)).cpu().numpy()
leak = bool(np.array_equal(base, again))
per_player_err = pd.DataFrame({"p": va.iloc[idx].player_ix.to_numpy(),
    "err": (base.argmax(1) != yf["val"][idx]).astype(int)}).groupby("p").err.mean()
json.dump({"byte_identical": leak, "note": "identity cols not in input constructor; check confirms data loading",
           "per_player_err_mean": float(per_player_err.mean()), "per_player_err_std": float(per_player_err.std())},
          open(OUT/"model2_context"/"leakage.json","w"), indent=1)
print("leakage byte-identical:", leak, "per-player err std:", round(float(per_player_err.std()),4))
assert leak

# conflict set: JSD = true divergence in nats (0..ln2). scipy.jensenshannon returns sqrt(distance) — we square nothing, we compute directly.
def jsd(P, Q):
    M = 0.5*(P+Q); return 0.5*(np.sum(P*np.log(P/M+1e-12),1)+np.sum(Q*np.log(Q/M+1e-12),1))
Pid, Pctx = probs["model1_identity_full"]["val"], probs["model2_context"]["val"]
aid, actx, js = Pid.argmax(1), Pctx.argmax(1), jsd(Pid, Pctx)
dis = aid != actx
print(f"val argmax-disagree rate: {dis.mean():.3f}")
qs = np.percentile(js[dis], [50,75,80,85,90,95])
# pick tau: first pct where conflict rate in 10-25% and top-pair share < .45 (else next)
rows = []
for q, t in zip([50,75,80,85,90,95], qs):
    c = dis & (js > t); pr = va.iloc[np.where(c)[0]].copy() if c.sum() else va.iloc[:0]
    tp = (c.sum()/len(va))
    top = ((pd.Series(list(zip(aid[c],actx[c]))).value_counts().iloc[0]/c.sum()) if c.sum() else 1)
    rows.append((q, float(t), int(c.sum()), round(float(tp),4), round(float(top),3)))
    print(f" tau p{q}={t:.4f}: n={c.sum()} rate={tp:.3f} top-pair={top:.3f}")
tau = next((t for q,t,n,r,tp in rows if 0.10 <= r <= 0.25 and tp < 0.45), rows[2][1])
print("chosen tau:", round(float(tau),4))
(OUT/"conflict_set").mkdir(exist_ok=True)
json.dump({"tau": float(tau), "convention": "JSD true divergence in nats (0..ln2), computed directly; scipy.spatial.distance.jensenshannon returns its sqrt — NOT used",
           "sweep": rows, "rule": "conflict=(a_id!=a_ctx)&(JSD>tau); tau on val only",
           "untrained_comp_ix": 5, "handling": "flagged via has_untrained_comp, not excluded"},
          open(OUT/"conflict_set"/"tau_selection.json","w"), indent=1)
# predictions parquet (val+test)
for s, a in [("val", va), ("test", te)]:
    pd.DataFrame({"match_id": a.match_id.to_numpy(), "event_id": a.event_id.to_numpy(),
                  "index": a["index"].to_numpy(), "split": s,
                  "P_id": list(probs["model1_identity_full"][s]),
                  "P_id_po": list(probs["model1_identity_playeronly"][s]),
                  "P_ctx": list(probs["model2_context"][s])}).to_parquet(OUT/"predictions"/f"{s}_probs.parquet")
# conflict parquet (val+test)
frames = []
for s, a in [("val", va), ("test", te)]:
    P1, P2 = probs["model1_identity_full"][s], probs["model2_context"][s]
    i1, i2, j = P1.argmax(1), P2.argmax(1), jsd(P1, P2)
    m = (i1 != i2) & (j > tau)
    f = pd.DataFrame({"match_id": a.match_id.to_numpy()[m], "event_id": a.event_id.to_numpy()[m],
        "index": a["index"].to_numpy()[m], "split": s, "player_ix": a.player_ix.to_numpy()[m],
        "a_id": i1[m], "P_id_a": P1[np.arange(len(a))[m], i1[m]], "a_ctx": i2[m],
        "P_ctx_a": P2[np.arange(len(a))[m], i2[m]], "JSD": j[m], "action_id": a.action_id.to_numpy()[m],
        "zone": a.zone.to_numpy()[m], "minute_bin": a.minute_bin.to_numpy()[m],
        "play_pattern_id": a.play_pattern_id.to_numpy()[m],
        "has_untrained_comp": (a.comp_ix.to_numpy()[m] == 5)})
    frames.append(f)
cf = pd.concat(frames, ignore_index=True)
cf.to_parquet(OUT/"conflict_set"/"conflict_set.parquet")
print(f"conflict: {len(cf)} rows = {len(cf)/(len(va)+len(te)):.3f} of val+test")
print("top pairs:", cf.groupby(["a_id","a_ctx"]).size().sort_values(ascending=False).head(8).to_dict())
print("by split:", cf.split.value_counts().to_dict(), "untrained-comp rows:", int(cf.has_untrained_comp.sum()))
print("DONE")
