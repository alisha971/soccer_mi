"""
src/analysis.py — loading run results and comparing conditions (notebook 04).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.metrics as metrics


def load_final_results(run_dir):
    with open(Path(run_dir) / "final_results.json") as handle:
        return json.load(handle)


def load_val_event_nll(run_dir, targets):
    """Per-event validation NLL of a finished run, ordered by event row."""
    predictions = np.load(Path(run_dir) / "val_predictions.npz")
    rows = predictions["event_rows"]
    order = np.argsort(rows)
    nll = metrics.per_event_nll(predictions["logits"][order], targets[rows[order]])
    return rows[order], nll


def condition_event_nll(run_dirs, targets):
    """Per-event NLL for each seed of one condition. Returns (event_rows, [n_seeds, n_events])."""
    all_nll = []
    reference_rows = None
    for run_dir in run_dirs:
        rows, nll = load_val_event_nll(run_dir, targets)
        if reference_rows is None:
            reference_rows = rows
        assert np.array_equal(rows, reference_rows), "runs were evaluated on different events"
        all_nll.append(nll)
    return reference_rows, np.stack(all_nll)


def compare_conditions(runs_a, runs_b, targets, match_ids, event_mask=None, n_resamples=2000, seed=0):
    """Is A better (lower NLL) than B beyond noise? See notebook 04, Section 2.5."""
    rows_a, nll_a = condition_event_nll(runs_a, targets)
    rows_b, nll_b = condition_event_nll(runs_b, targets)
    assert np.array_equal(rows_a, rows_b)
    if event_mask is None:
        keep = np.ones(len(rows_a), dtype=bool)
    else:
        keep = event_mask[rows_a]
    seed_means_a = nll_a[:, keep].mean(axis=1)
    seed_means_b = nll_b[:, keep].mean(axis=1)
    difference = nll_a[:, keep].mean(axis=0) - nll_b[:, keep].mean(axis=0)
    interval = metrics.match_bootstrap_mean(difference, match_ids[rows_a][keep], n_resamples, seed)
    ci_below_zero = interval["ci_high"] < 0
    separated = seed_means_a.max() < seed_means_b.min()
    return {
        "n_events": int(keep.sum()),
        "nll_a_mean": float(seed_means_a.mean()), "nll_a_seed_sd": float(seed_means_a.std(ddof=1)) if len(seed_means_a) > 1 else 0.0,
        "nll_b_mean": float(seed_means_b.mean()), "nll_b_seed_sd": float(seed_means_b.std(ddof=1)) if len(seed_means_b) > 1 else 0.0,
        "difference_a_minus_b": interval["mean"],
        "difference_ci_low": interval["ci_low"], "difference_ci_high": interval["ci_high"],
        "seeds_completely_separated": bool(separated),
        "a_better_beyond_noise": bool(ci_below_zero and separated),
    }


def smallest_within_noise(candidates, targets, match_ids):
    """candidates: list of (label, run_dirs) ordered simplest -> most complex.

    Returns (chosen_label, table). The best candidate is the one with the lowest mean NLL;
    the chosen one is the first candidate the best does NOT beat beyond noise.
    """
    means = []
    for label, run_dirs in candidates:
        _, nll = condition_event_nll(run_dirs, targets)
        means.append(float(nll.mean(axis=1).mean()))
    best_position = int(np.argmin(means))
    best_label, best_runs = candidates[best_position]
    rows = []
    chosen = None
    for position, (label, run_dirs) in enumerate(candidates):
        if position == best_position:
            comparison = None
            beaten = False
        else:
            comparison = compare_conditions(best_runs, run_dirs, targets, match_ids)
            beaten = comparison["a_better_beyond_noise"]
        if chosen is None and not beaten:
            chosen = label
        row = {"candidate": label, "mean val NLL": round(means[position], 4),
               "is best": position == best_position, "best beats it beyond noise": beaten}
        if comparison is not None:
            row["best - candidate (CI)"] = (f"{comparison['difference_a_minus_b']:+.4f} "
                                            f"[{comparison['difference_ci_low']:+.4f}, {comparison['difference_ci_high']:+.4f}]")
            row["seeds separated"] = comparison["seeds_completely_separated"]
        rows.append(row)
    return chosen, pd.DataFrame(rows).set_index("candidate")


def seed_summary(run_dirs):
    rows = []
    for run_dir in run_dirs:
        result = load_final_results(run_dir)
        rows.append({"run": Path(run_dir).name, "seed": result["run_config"]["seed"],
                     "best epoch": result["best_epoch"], "epochs run": result["epochs_run"],
                     "diverged": result["diverged"], "val NLL": result["val_metrics"]["nll"],
                     "val accuracy": result["val_metrics"]["accuracy"],
                     "val macro-F1": result["val_metrics"]["macro_f1"], "train NLL": result["train_nll"],
                     "minutes": result["runtime_seconds"] / 60})
    return pd.DataFrame(rows)


def load_history(run_dir):
    return pd.read_csv(Path(run_dir) / "history.csv")
