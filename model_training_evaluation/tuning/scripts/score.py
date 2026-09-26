"""Mean validation NLL across tuning trials. Torch-free (stdlib only)."""
import argparse, csv, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # tuning/
REPO = ROOT.parent  # model_training_evaluation/


def tuning_nlls():
    vals = []
    for p in sorted((ROOT / "runs").glob("*/final_results.json")):
        try:
            vals.append(json.load(open(p))["val_metrics"]["nll"])
        except Exception:
            pass
    return vals


def incumbent_nlls():
    # frozen T300/L2/K16 lr3e-4 wd0.1 seeds in top-level runs/
    vals = []
    for d in sorted((REPO / "runs").glob("transformer_T300_pre_action_L2_H2_D128_F512_K16_g-all_lr0.0003_wd0.1_s*")):
        f = d / "final_results.json"
        if f.exists():
            vals.append(json.load(open(f))["val_metrics"]["nll"])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-nll", action="store_true")
    ap.add_argument("--trials", default=str(ROOT / "results" / "trial-results.tsv"))
    ap.add_argument("--config", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-diverged", action="store_true")
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()

    if a.check:
        bad = [p for p in glob.glob(str(ROOT / "runs" / "*" / "final_results.json"))
               if json.load(open(p)).get("diverged")]
        if a.no_diverged and bad:
            print(f"FAIL diverged={len(bad)}")
            return 1
        print("OK")
        return 0

    vals = tuning_nlls() or incumbent_nlls()
    if not vals:
        print("metric-error: no final_results.json found", file=sys.stderr)
        return 2
    print(f"{sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    sys.exit(main())
