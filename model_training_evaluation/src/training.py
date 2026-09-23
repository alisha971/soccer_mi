"""
src/training.py — one controlled training procedure for every neural model (notebook 04).
"""
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import src.data as data
import src.batching as batching
import src.metrics as metrics
import src.model as model_module
import src.mlp_baseline as mlp_baseline

DEFAULT_TRAINING = {
    "batch_size": 256,
    "max_epochs": 30,
    "warmup_fraction": 0.05,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "patience": 5,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_determinism():
    """Ask PyTorch for deterministic kernels (warn instead of fail if one is unavailable)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment_info():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def learning_rate_at_step(step, total_steps, peak_lr, warmup_steps, min_lr_ratio):
    """Linear warm-up to peak_lr, then cosine decay to min_lr_ratio * peak_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def make_optimizer(model, training):
    """AdamW. Weight decay only on 2-D weight matrices outside the input embedding module."""
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        is_input_encoding = name.startswith("embedding.")
        if parameter.ndim == 2 and not is_input_encoding:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    groups = [
        {"params": decay_parameters, "weight_decay": training["weight_decay"]},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=training["learning_rate"],
                             betas=(training["beta1"], training["beta2"]), eps=training["eps"])

_DATA_CACHE = {}


def prepare_data(events, artifacts_dir, encoding_name, possession_features, context_length, stride,
                 device, run_mode="full"):
    """Encode inputs and build windows/evaluation views for one (encoding, context length)."""
    key = (encoding_name, possession_features, context_length, stride, str(device), run_mode)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    encoding = data.load_encoding(Path(artifacts_dir) / "encodings" / encoding_name)
    assert encoding["n_events"] == len(events), "encoding was built for a different event table"
    features, vocabulary_sizes = data.encode_features(events, encoding, possession_features)
    targets = events["action_id"].to_numpy().astype("int64")
    encoded = batching.EncodedData(features, targets, device)

    train_windows = data.build_windows(events, context_length, stride, split="train")
    views = {}
    for split in ["train", "val"]:
        views[split] = data.build_evaluation_view(events, split, context_length, stride)

    if run_mode == "smoke":
        # small, fixed subsets: code check only
        train_windows = train_windows[:512]
        # the subset is defined by EVENTS (first two validation match-periods), so that every
        # context length scores exactly the same events
        val_view = views["val"]
        period_key = (events["match_id"].to_numpy()[val_view["event_rows"]] * 10
                      + events["period"].to_numpy()[val_view["event_rows"]])
        first_two = np.unique(period_key)[:2]
        keep_events = np.isin(period_key, first_two)
        keep_windows = np.unique(val_view["window_index"][keep_events])
        new_window_index = np.searchsorted(keep_windows, val_view["window_index"][keep_events])
        small_view = {"windows": val_view["windows"][keep_windows],
                      "event_rows": val_view["event_rows"][keep_events],
                      "window_index": new_window_index,
                      "position": val_view["position"][keep_events]}
        views["val"] = small_view
        views["train"] = small_view
        views["test"] = small_view          # never real test data in smoke mode

    bundle = {
        "encoding_name": encoding_name,
        "possession_features": possession_features,
        "context_length": context_length,
        "stride": stride,
        "run_mode": run_mode,
        "vocabulary_sizes": vocabulary_sizes,
        "encoded": encoded,
        "targets": targets,
        "match_ids": events["match_id"].to_numpy(),
        "train_windows": torch.as_tensor(train_windows, device=device),
        "views": views,
        "events": events,
        "artifacts_dir": str(artifacts_dir),
        "test_unlocked": run_mode == "smoke",
        "device": device,
    }
    _DATA_CACHE[key] = bundle
    return bundle


def unlock_test_split(bundle, output_dir, events):
    """Allow test evaluation ONLY after the configuration is frozen. Logs every unlock."""
    frozen_path = Path(output_dir) / "config" / "frozen_config.json"
    if not frozen_path.exists():
        raise RuntimeError("Test split is locked: freeze the configuration (notebook 05) first.")
    if bundle["run_mode"] == "full" and "test" not in bundle["views"]:
        bundle["views"]["test"] = data.build_evaluation_view(events, "test", bundle["context_length"], bundle["stride"])
    bundle["test_unlocked"] = True
    log_path = Path(output_dir) / "logs" / "test_access_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                 "frozen_config": str(frozen_path),
                                 "encoding": bundle["encoding_name"]}) + "\n")


@torch.no_grad()
def predict_view(model, bundle, split, batch_size=512):
    """Logits for every event of a split, each event scored once. Returns (logits, targets, event_rows)."""
    if split == "test" and not bundle["test_unlocked"]:
        raise RuntimeError("Test split is locked.")
    view = bundle["views"][split]
    windows = torch.as_tensor(view["windows"], device=bundle["device"])
    model.eval()
    window_logits = []
    for start in range(0, len(windows), batch_size):
        inputs, _ = bundle["encoded"].batch(windows[start:start + batch_size])
        window_logits.append(model(inputs).float().cpu())
    window_logits = torch.cat(window_logits, dim=0)                      # [n_windows, T, C]
    window_index = torch.as_tensor(view["window_index"])
    position = torch.as_tensor(view["position"])
    event_logits = window_logits[window_index, position].numpy()         # [n_events, C]
    event_rows = view["event_rows"]
    return event_logits, bundle["targets"][event_rows], event_rows

def build_model(run_config, vocabulary_sizes):
    model_settings = dict(run_config["model"])
    model_settings["context_length"] = run_config["context_length"]
    if run_config["model_type"] == "transformer":
        config = model_module.ModelConfig(vocabulary_sizes=vocabulary_sizes, **model_settings)
        return model_module.EventTransformer(config)
    if run_config["model_type"] == "mlp":
        hidden = model_settings.pop("hidden", 512)
        n_previous = model_settings.pop("n_previous", 3)
        config = model_module.ModelConfig(vocabulary_sizes=vocabulary_sizes, **model_settings)
        return mlp_baseline.PreviousEventsMLP(config, n_previous=n_previous, hidden=hidden)
    raise ValueError(run_config["model_type"])


def run_name_for(run_config):
    """A readable name that encodes every setting that differs between runs."""
    model = run_config["model"]
    training = run_config["training"]
    groups = model.get("use_groups", model_module.ALL_GROUPS)
    if list(groups) == list(model_module.ALL_GROUPS):
        group_tag = "all"
    else:
        group_tag = "+".join(groups)
    name = (f"{run_config['model_type']}_{run_config['encoding']}_{run_config['possession_features']}"
            f"_L{model.get('n_layers', 4)}_H{model.get('n_heads', 4)}_D{model.get('d_model', 128)}"
            f"_F{model.get('d_ff', 512)}_K{run_config['context_length']}_g-{group_tag}"
            f"_lr{training['learning_rate']:g}_wd{training['weight_decay']:g}_s{run_config['seed']}")
    return name


def save_json(path, content):
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2, default=str)


def train_run(run_config, bundle, runs_dir, manifest_path, keep_every_epoch=False, verbose=True):
    """Train one model (or load it if already finished). Returns final_results dict."""
    run_name = run_name_for(run_config)
    run_dir = Path(runs_dir) / run_name
    final_path = run_dir / "final_results.json"
    if final_path.exists():
        with open(final_path) as handle:
            return json.load(handle)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_config.json", run_config)

    training = dict(DEFAULT_TRAINING)
    training.update(run_config["training"])
    if bundle["run_mode"] == "smoke":
        training["max_epochs"] = min(training["max_epochs"], 2)
    device = bundle["device"]
    seed = run_config["seed"]

    configure_determinism()
    set_seed(seed)
    model = build_model(run_config, bundle["vocabulary_sizes"]).to(device)
    optimizer = make_optimizer(model, training)

    train_windows = bundle["train_windows"]
    n_windows = len(train_windows)
    steps_per_epoch = math.ceil(n_windows / training["batch_size"])
    total_steps = steps_per_epoch * training["max_epochs"]
    warmup_steps = max(1, round(training["warmup_fraction"] * total_steps))

    state = {"epoch": 0, "step": 0, "best_val_nll": float("inf"), "best_epoch": -1,
             "epochs_without_improvement": 0, "diverged": False, "elapsed_seconds": 0.0}
    last_path = run_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["state"]
        if verbose:
            print(f"[{run_name}] resuming after epoch {state['epoch']}")

    history_path = run_dir / "history.csv"
    while (state["epoch"] < training["max_epochs"]
           and state["epochs_without_improvement"] < training["patience"]
           and not state["diverged"]):
        epoch_start = time.time()
        generator = torch.Generator().manual_seed(seed * 1000 + state["epoch"])
        order = torch.randperm(n_windows, generator=generator).to(device)
        model.train()
        loss_sum = 0.0
        gradient_norm_sum = 0.0
        clipped_steps = 0
        batches = 0
        learning_rate = 0.0
        for batch_start in range(0, n_windows, training["batch_size"]):
            batch_ids = order[batch_start:batch_start + training["batch_size"]]
            inputs, targets = bundle["encoded"].batch(train_windows[batch_ids])

            learning_rate = learning_rate_at_step(state["step"], total_steps, training["learning_rate"],
                                                  warmup_steps, training["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            logits = model(inputs)
            loss = model_module.compute_loss(logits, targets)
            if not torch.isfinite(loss):
                state["diverged"] = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"]).item()
            optimizer.step()

            state["step"] += 1
            loss_sum += loss.item()
            gradient_norm_sum += gradient_norm
            if gradient_norm > training["grad_clip"]:
                clipped_steps += 1
            batches += 1
        if state["diverged"]:
            break

        val_logits, val_targets, _ = predict_view(model, bundle, "val")
        val_metrics = metrics.compute_metrics(val_logits, val_targets)
        epoch_seconds = time.time() - epoch_start
        state["elapsed_seconds"] += epoch_seconds
        state["epoch"] += 1

        row = {"epoch": state["epoch"], "step": state["step"], "learning_rate": learning_rate,
               "train_loss": loss_sum / batches, "val_nll": val_metrics["nll"],
               "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
               "mean_gradient_norm": gradient_norm_sum / batches,
               "clipped_fraction": clipped_steps / batches, "epoch_seconds": epoch_seconds}
        pd.DataFrame([row]).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)

        if val_metrics["nll"] < state["best_val_nll"]:
            state["best_val_nll"] = val_metrics["nll"]
            state["best_epoch"] = state["epoch"]
            state["epochs_without_improvement"] = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            state["epochs_without_improvement"] += 1
        if keep_every_epoch:
            torch.save(model.state_dict(), run_dir / f"epoch_{state['epoch']:03d}.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, last_path)
        if verbose:
            print(f"[{run_name}] epoch {state['epoch']:2d} | train loss {row['train_loss']:.4f} | "
                  f"val NLL {row['val_nll']:.4f} | val acc {row['val_accuracy']:.4f} | "
                  f"val macro-F1 {row['val_macro_f1']:.4f} | {epoch_seconds:.0f}s")

    return finalise_run(run_config, run_name, run_dir, model, bundle, training, state, manifest_path)


def finalise_run(run_config, run_name, run_dir, model, bundle, training, state, manifest_path):
    if (run_dir / "best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=bundle["device"]))
    val_logits, val_targets, val_rows = predict_view(model, bundle, "val")
    np.savez_compressed(run_dir / "val_predictions.npz", logits=val_logits.astype("float32"), event_rows=val_rows)
    train_logits, train_targets, train_rows = predict_view(model, bundle, "train")
    train_nll = metrics.per_event_nll(train_logits, train_targets)
    np.savez_compressed(run_dir / "train_event_nll.npz", nll=train_nll.astype("float32"), event_rows=train_rows)

    if run_config["model_type"] == "transformer":
        parameter_counts = model_module.count_parameters(model)
    else:
        parameter_counts = {"total": sum(p.numel() for p in model.parameters())}
    final_results = {
        "run_name": run_name,
        "run_config": run_config,
        "training_settings": training,
        "run_mode": bundle["run_mode"],
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "stopped_early": state["epochs_without_improvement"] >= training["patience"],
        "diverged": state["diverged"],
        "runtime_seconds": state["elapsed_seconds"],
        "val_metrics": metrics.compute_metrics(val_logits, val_targets),
        "train_nll": float(train_nll.mean()),
        "parameter_counts": parameter_counts,
        "environment": environment_info(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_dir / "final_results.json", final_results)
    manifest_entry = {"run_name": run_name, "run_dir": str(run_dir), "seed": run_config["seed"],
                      "encoding": run_config["encoding"], "model_type": run_config["model_type"],
                      "model": run_config["model"], "context_length": run_config["context_length"],
                      "training": training, "best_epoch": state["best_epoch"],
                      "val_nll": final_results["val_metrics"]["nll"],
                      "parameters": parameter_counts.get("total"), "run_mode": bundle["run_mode"],
                      "finished_at": final_results["finished_at"]}
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as handle:
        handle.write(json.dumps(manifest_entry, default=str) + "\n")
    return final_results


def load_trained_model(run_dir, bundle):
    with open(Path(run_dir) / "run_config.json") as handle:
        run_config = json.load(handle)
    model = build_model(run_config, bundle["vocabulary_sizes"]).to(bundle["device"])
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=bundle["device"]))
    model.eval()
    return model
