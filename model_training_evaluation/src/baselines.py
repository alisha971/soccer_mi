"""
src/baselines.py — non-neural baselines B1-B3 (written by notebook 02).
Every function fits on training rows only and returns logits (log-probabilities) for
the rows it is asked to predict, so all models share one evaluation path.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

NUM_CLASSES = 15


def class_prior(train_targets):
    counts = np.bincount(train_targets, minlength=NUM_CLASSES).astype(float)
    return counts / counts.sum()


def predict_class_prior(train_targets, n_rows):
    """B1a: every event gets the overall training class distribution."""
    prior = class_prior(train_targets)
    return np.tile(np.log(prior), (n_rows, 1))


def smoothed_conditional(train_keys, train_targets, n_keys, fallback, alpha):
    """P(action | key) with additive smoothing towards `fallback` [n_keys, 15] probabilities."""
    counts = np.zeros((n_keys, NUM_CLASSES))
    np.add.at(counts, (train_keys, train_targets), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return (counts + alpha * fallback) / (totals + alpha)


def zone_prior_table(train_zone, train_targets, alpha):
    prior = class_prior(train_targets)
    fallback = np.tile(prior, (96, 1))
    return smoothed_conditional(train_zone, train_targets, 96, fallback, alpha)


def predict_zone_prior(train_zone, train_targets, zone, alpha):
    """B1b: P(action | start zone)."""
    table = zone_prior_table(train_zone, train_targets, alpha)
    return np.log(table[zone])


def predict_markov(train_prev_action, train_zone, train_targets, prev_action, zone, alpha):
    """B2: P(action | previous action, zone), smoothed towards P(action | zone)."""
    zone_table = zone_prior_table(train_zone, train_targets, alpha)
    n_keys = 16 * 96
    train_keys = train_prev_action * 96 + train_zone
    fallback = np.zeros((n_keys, NUM_CLASSES))
    for key in range(n_keys):
        fallback[key] = zone_table[key % 96]
    table = smoothed_conditional(train_keys, train_targets, n_keys, fallback, alpha)
    keys = prev_action * 96 + zone
    return np.log(table[keys])


def logistic_regression_design(features, categorical_names, continuous_names, rows, encoder=None, scaler=None):
    """One-hot categorical columns + standardised continuous columns (fitted on train only)."""
    categorical = np.stack([features[name][rows] for name in categorical_names], axis=1)
    continuous = np.stack([features[name][rows] for name in continuous_names], axis=1).astype(float)
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(categorical)
        scaler = StandardScaler()
        scaler.fit(continuous)
    design = sparse.hstack([encoder.transform(categorical), sparse.csr_matrix(scaler.transform(continuous))]).tocsr()
    return design, encoder, scaler


def fit_logistic_regression(features, targets, categorical_names, continuous_names, train_rows,
                            max_iter=1000, C=1.0, chunk_iterations=200, checkpoint_path=None):
    """B3: multinomial logistic regression on the current event's inputs.

    Fitted with L-BFGS in chunks of `chunk_iterations` iterations. Each chunk starts from the
    coefficients of the previous one (warm start). After every chunk the model is saved to
    `checkpoint_path`, so an interrupted fit (e.g. a Colab disconnection) resumes where it
    stopped. Fitting stops when a chunk converges before its limit, or after `max_iter` in total.
    """
    design, encoder, scaler = logistic_regression_design(features, categorical_names, continuous_names, train_rows)
    train_targets = targets[train_rows]
    model = LogisticRegression(C=C, max_iter=chunk_iterations, warm_start=True)
    state = {"iterations_done": 0, "converged": False, "n_train_rows": int(len(train_rows))}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        saved = joblib.load(checkpoint_path)
        if saved["state"]["n_train_rows"] == len(train_rows):
            model = saved["model"]
            state = saved["state"]
    while (not state["converged"]) and state["iterations_done"] < max_iter:
        model.fit(design, train_targets)
        used = int(model.n_iter_[0])
        state["iterations_done"] += used
        if used < chunk_iterations:
            state["converged"] = True
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "state": state}, checkpoint_path)
    return {"model": model, "encoder": encoder, "scaler": scaler,
            "categorical_names": categorical_names, "continuous_names": continuous_names,
            "iterations": state["iterations_done"], "converged": state["converged"]}


def predict_logistic_regression(fitted, features, rows):
    design, _, _ = logistic_regression_design(features, fitted["categorical_names"], fitted["continuous_names"],
                                              rows, fitted["encoder"], fitted["scaler"])
    log_probabilities = fitted["model"].predict_log_proba(design)
    # predict_log_proba only returns columns for classes seen in training; all 15 are present.
    return log_probabilities
