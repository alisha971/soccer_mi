"""Decision-rule comparison for tuning finalists (Track A 04 S2.5).

Compares mean val NLL over seeds: bootstrap 95% CI of per-event diff (match-level)
AND complete seed separation. Torch-free. Run with venv python (numpy/pandas/sklearn).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
import src.metrics as metrics


def per_event_nll(run_dir):
    z = np.load(Path(run_dir) / "val_predictions.npz")
    return z["event_rows"], metrics.per_event_nll(z["logits"].astype(np.float64),
                                                  _TARGETS[z["event_rows"]])


def config_mean(dirs):
    stacked, ref_rows = [], None
    seeds = []
    for d in dirs:
        rows, nll = per_event_nll(d)
        if ref_rows is None:
            ref_rows = rows
        assert (rows == ref_rows).all(), f"row mismatch in {d}"
        stacked.append(nll)
        seeds.append(json.load(open(Path(d) / "final_results.json"))["val_metrics"]["nll"])
    return np.stack(stacked).mean(0), np.array(seeds), ref_rows


def compare(a_dirs, b_dirs, a_name, b_name):
    a_mean, a_seeds, rows = config_mean(a_dirs)
    b_mean, b_seeds, _ = config_mean(b_dirs)
    diff = a_mean - b_mean
    ci = metrics.match_bootstrap_mean(diff, _MATCH[rows], n_resamples=2000, seed=0)
    return {"a": a_name, "b": b_name, "n_events": int(len(diff)),
            "a_mean": float(a_seeds.mean()), "a_sd": float(a_seeds.std(ddof=1)),
            "b_mean": float(b_seeds.mean()), "b_sd": float(b_seeds.std(ddof=1)),
            "a_seeds": [round(v, 4) for v in a_seeds], "b_seeds": [round(v, 4) for v in b_seeds],
            "diff": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "separated": bool((a_seeds.max() < b_seeds.min()) or (b_seeds.max() < a_seeds.min())),
            "a_better": bool(ci["ci_high"] < 0 and (a_seeds.max() < b_seeds.min()))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "selection.json"))
    a = ap.parse_args()
    ev = pd.read_parquet(REPO / "artifacts" / "events_prepared.parquet")
    _TARGETS = ev["action_id"].to_numpy().astype("int64")
    _MATCH = ev["match_id"].to_numpy()

    T = lambda *p: str(ROOT / "runs" / Path(*p))  # noqa
    R = lambda *p: str(REPO / "runs" / Path(*p))  # noqa
    base = "transformer_T300_pre_action_L2_H2_D128_F512_K16_g-all"
    t3 = [f"{base}_lr0.0003_wd0.01_s{s}" for s in [0, 1, 2]]
    t1 = [f"{base}_lr0.001_wd0.01_s{s}" for s in [0, 1, 2]]
    inc = [f"{base}_lr0.0003_wd0.1_s{s}" for s in [0, 1, 2]]
    out = {"wd001_lr3e4_vs_inc": compare([T(d) for d in t3], [R(d) for d in inc], "lr3e-4_wd0.01", "lr3e-4_wd0.1"),
           "wd001_lr1e3_vs_inc": compare([T(d) for d in t1], [R(d) for d in inc], "lr1e-3_wd0.01", "lr3e-4_wd0.1"),
           "lr1e3_vs_lr3e4_wd001": compare([T(d) for d in t1], [T(d) for d in t3], "lr1e-3_wd0.01", "lr3e-4_wd0.01")}
    # tie rule: indistinguishable -> lower LR; same LR -> keep wd 0.1
    out["recommendation"] = {"lr": 3e-4, "wd": 0.1, "reason": "set after verdict"}
    json.dump(out, open(a.out, "w"), indent=2)
    for k, v in out.items():
        if k == "recommendation":
            continue
        print(f"{k}: diff={v['diff']:.4f} CI=[{v['ci_low']:.4f},{v['ci_high']:.4f}] "
              f"separated={v['separated']} a_better={v['a_better']}")
