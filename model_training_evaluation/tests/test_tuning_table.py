"""Consistency test for the hyperparameter sweep table.

Every row of tuning/results/hyperparameter_sweep.csv is cross-checked
against its run's final_results.json (and trial JSON). Fails on any
transcription mismatch. Stdlib-only core; numpy check skipped if absent.
"""
import csv
import json
from pathlib import Path

TUNING = Path(__file__).resolve().parents[1] / "tuning"


def load_rows():
    with open(TUNING / "results" / "hyperparameter_sweep.csv") as handle:
        return list(csv.DictReader(handle))


def test_row_count():
    assert len(load_rows()) == 18


def run_dir_of(trial_id):
    stored = json.load(open(TUNING / "trials" / f"{trial_id}.json"))["run_dir"]
    return TUNING.parent / stored


def test_rows_match_artifacts():
    for row in load_rows():
        run_dir = run_dir_of(row["trial_id"])
        final = json.load(open(run_dir / "final_results.json"))
        metrics, settings = final["val_metrics"], final["training_settings"]
        assert float(row["learning_rate"]) == settings["learning_rate"]
        assert float(row["weight_decay"]) == settings["weight_decay"]
        assert int(row["batch_size"]) == settings["batch_size"]
        assert float(row["warmup_fraction"]) == settings["warmup_fraction"]
        assert int(row["seed"]) == final["run_config"]["seed"]
        assert abs(float(row["val_nll_nats"]) - metrics["nll"]) < 5e-5
        assert abs(float(row["val_accuracy"]) - metrics["accuracy"]) < 5e-5
        assert abs(float(row["val_macro_f1"]) - metrics["macro_f1"]) < 5e-5
        assert abs(float(row["val_ece"]) - metrics["ece"]) < 5e-5
        assert int(row["best_epoch"]) == final["best_epoch"]
        assert (row["diverged"] == "True") == bool(final["diverged"])
        assert int(row["runtime_seconds"]) == round(final["runtime_seconds"])


def test_prediction_files_cover_reported_events():
    try:
        import numpy as np
    except ImportError:
        return
    for row in load_rows():
        run_dir = run_dir_of(row["trial_id"])
        final = json.load(open(run_dir / "final_results.json"))
        packed = np.load(run_dir / "val_predictions.npz")
        assert len(packed["event_rows"]) == final["val_metrics"]["n_events"]
