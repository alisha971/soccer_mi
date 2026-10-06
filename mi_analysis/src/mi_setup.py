"""
mi_analysis/src/mi_setup.py — load the frozen SEL-4L model and locate conflict events (MB02).

Everything is built from model_training_evaluation's own code (src.data, src.training) so that inputs, windows and
evaluation views are exactly the ones SEL-4L was trained and evaluated with.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.data as data
import src.training as training
import mi_src.paths as paths

ENCODING = "T300"
POSSESSION_FEATURES = "pre_action"
CONTEXT_LENGTH = 16
STRIDE = 8
CHECKED_FIELDS = [("model", "n_layers"), ("model", "n_heads"), ("model", "d_model"), ("model", "d_ff"),
                  ("context_length",), ("encoding",), ("possession_features",),
                  ("training", "learning_rate"), ("training", "weight_decay")]


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json(path, content):
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2, default=str)


def build_bundle(events, track, device):
    """Encoded inputs, training windows and train/val evaluation views at K = 16 (model_training_evaluation code)."""
    return training.prepare_data(events, track["artifacts"], ENCODING, POSSESSION_FEATURES,
                                 CONTEXT_LENGTH, STRIDE, device, "full")


def sel4l_runs(track):
    """The five SEL-4L final runs and the primary seed; run directories re-rooted onto this checkout."""
    runs = pd.read_csv(track["final_runs"]).sort_values("seed").reset_index(drop=True)
    runs["run_dir"] = [str(paths.rebase_run_dir(value, track["root"])) for value in runs["run_dir"]]
    missing = [value for value in runs["run_dir"] if not Path(value).exists()]
    if missing:
        raise RuntimeError(f"BLOCKED: {len(missing)} SEL-4L run directories do not exist here, "
                           f"for example {missing[0]}.")
    final_validation = load_json(track["final_validation"])
    return runs, int(final_validation["primary_seed"])


def restricted_run_dir(track, model_name, seed=None):
    """Directory of one notebook-16 restricted run, re-rooted onto this checkout."""
    table = pd.read_csv(track["restricted_runs"])
    rows = table[table["model"] == model_name]
    if seed is not None:
        rows = rows[rows["seed"] == seed]
    if len(rows) == 0:
        raise RuntimeError(f"BLOCKED: no restricted run called {model_name!r} in {track['restricted_runs']}.")
    rows = rows.sort_values("seed")
    return paths.rebase_run_dir(rows["run_dir"].iloc[0], track["root"])


def run_dir_for_seed(runs, seed):
    return Path(runs.loc[runs["seed"] == seed, "run_dir"].iloc[0])


def config_mismatches(run_config, selected):
    problems = []
    for field in CHECKED_FIELDS:
        a = run_config
        b = selected
        for key in field:
            a = a.get(key) if isinstance(a, dict) else None
            b = b.get(key) if isinstance(b, dict) else None
        if a != b:
            problems.append(f"{'.'.join(field)}: run {a} vs selected {b}")
    return problems


def load_model(run_dir, bundle):
    """Load best.pt with strict=True, switch to evaluation mode and freeze every parameter."""
    model = training.load_trained_model(run_dir, bundle)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluation_view(events, split):
    return data.build_evaluation_view(events, split, CONTEXT_LENGTH, STRIDE)


def locate(view, event_rows):
    """(window number, position) at which each event is scored in an evaluation view."""
    lookup = {}
    for index, row in enumerate(view["event_rows"]):
        lookup[int(row)] = index
    windows = np.empty(len(event_rows), dtype=np.int64)
    positions = np.empty(len(event_rows), dtype=np.int64)
    for i, row in enumerate(event_rows):
        index = lookup[int(row)]
        windows[i] = view["window_index"][index]
        positions[i] = view["position"][index]
    return windows, positions


def window_inputs(bundle, view, window_numbers):
    """Model inputs (dict name -> [B, K]) for a list of evaluation-view windows."""
    rows = torch.as_tensor(view["windows"][np.asarray(window_numbers)], device=bundle["device"])
    inputs, targets = bundle["encoded"].batch(rows)
    return inputs, targets


def batch_slices(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def unlock_test_view(events, out_dir):
    """Build the test evaluation view ONLY if locked predictions exist; log every unlock (MB12)."""
    locked = Path(out_dir) / "config" / "locked_predictions.json"
    if not locked.exists():
        raise RuntimeError("BLOCKED: the test split stays locked until config/locked_predictions.json exists.")
    from datetime import datetime, timezone
    import hashlib
    digest = hashlib.sha256(locked.read_bytes()).hexdigest()
    with open(Path(out_dir) / "logs" / "test_access_log.jsonl", "a") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                 "locked_predictions_sha256": digest}) + "\n")
    return evaluation_view(events, "test")


def event_batches(bundle, view, table, batch_size):
    """Yield (row slice, inputs, positions, a_ctx, a_id) for a table with columns
    window, position, a_ctx, a_id (one row per scored event)."""
    device = bundle["device"]
    for part in batch_slices(len(table), batch_size):
        chunk = table.iloc[part]
        inputs, _ = window_inputs(bundle, view, chunk["window"].to_numpy())
        positions = torch.as_tensor(chunk["position"].to_numpy(), device=device)
        a_ctx = torch.as_tensor(chunk["a_ctx"].to_numpy(), device=device)
        a_id = torch.as_tensor(chunk["a_id"].to_numpy(), device=device)
        yield part, inputs, positions, a_ctx, a_id


@torch.no_grad()
def reference_means(model, bundle, view, window_numbers, sites, batch_size, forward_function):
    """Mean of each site over reference windows, per position ([T, ...]); used for mean ablation."""
    totals = {}
    count = 0
    for part in batch_slices(len(window_numbers), batch_size):
        inputs, _ = window_inputs(bundle, view, window_numbers[part])
        _, cache = forward_function(model, inputs, capture=sites)
        for site in sites:
            summed = cache[site].sum(dim=0)
            totals[site] = summed if site not in totals else totals[site] + summed
        count += len(window_numbers[part])
    means = {}
    for site in sites:
        means[site] = totals[site] / count
    return means
