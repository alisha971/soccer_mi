"""
mi_analysis/src/paths.py — folder layout and run-mode settings of Track B (written by MB01).

Track B reads model_training_evaluation (Track A) and writes only below mi_analysis/. In smoke
mode every output goes to mi_analysis/smoke/, so a code check can never be mistaken for a result.

RUN_SETTINGS holds every number a notebook reads: reference_epochs (MB01 reference training),
batch_windows (evaluation windows per forward pass), n_bootstrap (match-level resamples behind
every interval), n_draws (draws averaged for resample ablation and the random-vector and
random-player controls), n_permutations (the C3 habit-correlation null), n_random_circuits
(size-matched random circuits, pseudo-heads and random neuron sets), probe_train_events (events
sampled per split for the MB05 probes), max_events (per-population cap, smoke only), probe_seeds
(repeats of the MB05 probes) and scrub_draws (resamples per node in the MB11b causal scrub).
"""
from pathlib import Path
import json

DATASET_SHA256 = "2fe8019fb59f8060f77ae0b81eba3df2077a3fd50c02f601ed7f401b3855fc5b"

SUBFOLDERS = [
    "artifacts/conflict/references",
    "artifacts/conflict/predictions",
    "artifacts/populations",
    "artifacts/caches",
    "config",
    "results",
    "figures",
    "logs",
    "manifests",
]

RUN_SETTINGS = {
    "full": {
        "reference_epochs": 25,
        "batch_windows": 256,
        "n_bootstrap": 2000,
        "n_draws": 5,
        "n_permutations": 1000,
        "n_random_circuits": 100,
        "probe_train_events": 20000,
        "max_events": 100000,
        "probe_seeds": 3,
        "scrub_draws": 5,
    },
    "smoke": {
        "reference_epochs": 2,
        "batch_windows": 64,
        "n_bootstrap": 200,
        "n_draws": 2,
        "n_permutations": 50,
        "n_random_circuits": 10,
        "probe_train_events": 1500,
        "max_events": 200,
        "probe_seeds": 2,
        "scrub_draws": 2,
    },
}


def output_dir(mi_dir, run_mode):
    """Folder that receives every output of this run mode; subfolders are created here."""
    out = Path(mi_dir) if run_mode == "full" else Path(mi_dir) / "smoke"
    for sub in SUBFOLDERS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def track_a_paths(track_a_dir):
    """Every Track A file that Track B reads. Nothing here is ever written."""
    root = Path(track_a_dir)
    hyper = root / "experiments" / "hyperparameter_optimisation"
    result = {
        "root": root,
        "src": root / "src",
        "config": root / "config",
        "artifacts": root / "artifacts",
        "results": hyper / "results",
        "events": root / "artifacts" / "events_prepared.parquet",
        "encoding": root / "artifacts" / "encodings" / "T300",
        "audit_report": root / "artifacts" / "audit_report.json",
        "frozen_config": root / "config" / "frozen_config.json",
        "selected_config": hyper / "results" / "12_selected_config.json",
        "final_runs": hyper / "results" / "13_final_runs.csv",
        "final_validation": hyper / "results" / "13_final_validation.json",
        "restricted_runs": hyper / "ablations" / "results" / "16_restricted_runs.csv",
        "runs": hyper / "runs",
        "ablation_runs": hyper / "ablations" / "runs",
    }
    plan_path = Path(__file__).resolve().parents[1] / "config/analysis_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()).get("model_family") == "benchmark_wd0.1":
        benchmark = root / "benchmarks/track_a_4layer"
        local_config = plan_path.parent
        result.update(results=benchmark / "results", selected_config=benchmark / "config/frozen_config.json",
                      final_runs=local_config / "benchmark_final_runs.csv",
                      final_validation=benchmark / "results/06_4L_validation_report.json",
                      restricted_runs=local_config / "benchmark_restricted_runs.csv", runs=benchmark / "runs")
    return result


def rebase_run_dir(recorded, track_a_dir):
    """Re-root a run directory recorded on another machine onto this checkout.

    Track A stored absolute paths from the machine it trained on. Only the part from "experiments/"
    onwards identifies the run, so that tail is re-attached here. A path that already exists is
    returned unchanged.
    """
    recorded = Path(str(recorded).replace("\\", "/"))
    if recorded.exists():
        return recorded
    parts = recorded.parts
    if "benchmarks" in parts:
        return Path(track_a_dir).joinpath(*parts[parts.index("benchmarks"):])
    if "experiments" in parts:
        tail = parts[parts.index("experiments"):]
        return Path(track_a_dir).joinpath(*tail)
    return recorded

SCHEMA_VERSION = 2


def check_modules(**modules):
    """Raise unless every module carries the functions these notebooks need."""
    expected = {"paths": ["rebase_run_dir", "SCHEMA_VERSION"],
                "fingerprint": ["verify_files", "compare_snapshot", "write_snapshot"],
                "mi_setup": ["restricted_run_dir", "reference_means"],
                "circuits": ["all_senders", "all_receivers", "receiver_gradients", "copy_from", "class_donors"],
                "patching": ["sweep_sites", "patch_batch"]}
    missing = []
    for name, module in modules.items():
        for attribute in expected.get(name, []):
            if not hasattr(module, attribute):
                missing.append(f"{name}.{attribute}")
    if missing:
        raise RuntimeError(
            "BLOCKED: mi_analysis/src/ is older than these notebooks expect (missing "
            + ", ".join(missing) + "). The %%writefile cells are the source of truth and the "
            "files on disk have gone stale, usually after syncing between Drive and a local "
            "checkout. Re-run MB01 section 0, MB02 section 0, MB06 and MB07 up to their "
            "%%writefile cells, then restart the kernel.")
