"""Thin wrapper: build run_config for frozen T300/L2/K16 and call src.training.train_run.

Dry-run (no torch): writes config only. Full mode: imports src.training (needs torch/CUDA, run on Colab).
All outputs go to tuning/runs/ + tuning/manifests/ — never top-level runs/.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def build_config(lr=3e-4, wd=0.1, seed=0, max_epochs=30):
    return {"model_type": "transformer", "encoding": "T300", "possession_features": "pre_action",
            "context_length": 16, "stride": 8, "seed": seed,
            "model": {"n_layers": 2, "n_heads": 2, "d_model": 128, "d_ff": 512},
            "training": {"learning_rate": lr, "weight_decay": wd, "max_epochs": max_epochs}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["smoke", "full"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = build_config(a.lr, a.wd, a.seed, max_epochs=2 if a.mode == "smoke" else 30)
    if a.dry_run:
        out = ROOT / "configs" / f"trial_lr{a.lr:g}_wd{a.wd:g}_s{a.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out, "w"), indent=2)
        print(str(out))
        return 0

    sys.path.insert(0, str(REPO))  # noqa: ponytail global sys.path tweak, Colab single-entry only
    import torch  # noqa: deferred so --dry-run/py_compile work without torch
    import src.data as data
    import src.training as training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    events = data.load_events(REPO / "artifacts" / "events_prepared.parquet")
    bundle = training.prepare_data(events, REPO / "artifacts", "T300", "pre_action", 16, 8,
                                   device, run_mode="smoke" if a.mode == "smoke" else "full")
    res = training.train_run(cfg, bundle, ROOT / "runs", ROOT / "manifests" / "tuning_manifest.jsonl")
    print(json.dumps({"val_nll": res["val_metrics"]["nll"], "run": res["run_name"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
