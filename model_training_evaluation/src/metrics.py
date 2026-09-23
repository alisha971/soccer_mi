"""
src/metrics.py — evaluation metrics shared by every Track A model.
Written by notebook 02. All functions take numpy arrays.
"""
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

NUM_CLASSES = 15


def log_softmax(logits):
    """Convert logits [n, 15] into log-probabilities, in a numerically stable way."""
    largest = logits.max(axis=1, keepdims=True)
    shifted = logits - largest
    log_normaliser = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - log_normaliser


def per_event_nll(logits, targets):
    """-log p(true class) for every event."""
    log_probabilities = log_softmax(logits)
    event_numbers = np.arange(len(targets))
    return -log_probabilities[event_numbers, targets]


def top_k_accuracy(logits, targets, k):
    """Share of events whose true class is among the k highest logits."""
    order = np.argsort(-logits, axis=1)
    top_k = order[:, :k]
    hits = (top_k == targets[:, None]).any(axis=1)
    return float(hits.mean())


def expected_calibration_error(probabilities, targets, n_bins=15):
    """Top-label ECE with equal-width confidence bins."""
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == targets).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_number in range(n_bins):
        low = bin_edges[bin_number]
        high = bin_edges[bin_number + 1]
        if bin_number == 0:
            in_bin = (confidence >= low) & (confidence <= high)
        else:
            in_bin = (confidence > low) & (confidence <= high)
        if in_bin.sum() == 0:
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += (in_bin.sum() / len(targets)) * gap
    return float(ece)


def reliability_table(probabilities, targets, n_bins=15):
    """Per-bin mean confidence, accuracy and count (used for reliability diagrams)."""
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == targets).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for bin_number in range(n_bins):
        low = bin_edges[bin_number]
        high = bin_edges[bin_number + 1]
        in_bin = (confidence > low) & (confidence <= high)
        if bin_number == 0:
            in_bin = (confidence >= low) & (confidence <= high)
        count = int(in_bin.sum())
        if count == 0:
            rows.append({"bin_low": low, "bin_high": high, "count": 0,
                         "mean_confidence": np.nan, "accuracy": np.nan})
        else:
            rows.append({"bin_low": low, "bin_high": high, "count": count,
                         "mean_confidence": float(confidence[in_bin].mean()),
                         "accuracy": float(correct[in_bin].mean())})
    return rows


def compute_metrics(logits, targets):
    """All headline metrics for one set of predictions."""
    log_probabilities = log_softmax(logits)
    probabilities = np.exp(log_probabilities)
    predicted = logits.argmax(axis=1)
    nll_values = -log_probabilities[np.arange(len(targets)), targets]
    per_class = f1_score(targets, predicted, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {
        "n_events": int(len(targets)),
        "nll": float(nll_values.mean()),
        "accuracy": float((predicted == targets).mean()),
        "top3_accuracy": top_k_accuracy(logits, targets, 3),
        "top5_accuracy": top_k_accuracy(logits, targets, 5),
        "macro_f1": float(per_class.mean()),
        "per_class_f1": [float(value) for value in per_class],
        "ece": expected_calibration_error(probabilities, targets),
    }


def confusion(logits, targets):
    """Confusion matrix: rows = true class, columns = predicted class."""
    return confusion_matrix(targets, logits.argmax(axis=1), labels=list(range(NUM_CLASSES)))


def match_bootstrap_mean(values, match_ids, n_resamples=2000, seed=0):
    """Mean of per-event `values` with a 95% CI from resampling whole matches.

    For each resample, matches are drawn with replacement; the statistic is
    (sum of values in drawn matches) / (number of events in drawn matches).
    """
    unique_matches, match_position = np.unique(match_ids, return_inverse=True)
    n_matches = len(unique_matches)
    sum_per_match = np.bincount(match_position, weights=values, minlength=n_matches)
    count_per_match = np.bincount(match_position, minlength=n_matches).astype(float)
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples)
    for resample in range(n_resamples):
        drawn = rng.integers(0, n_matches, size=n_matches)
        resampled_means[resample] = sum_per_match[drawn].sum() / count_per_match[drawn].sum()
    point = float(values.mean())
    return {"mean": point,
            "ci_low": float(np.percentile(resampled_means, 2.5)),
            "ci_high": float(np.percentile(resampled_means, 97.5))}


def match_bootstrap_macro_f1(logits, targets, match_ids, n_resamples=500, seed=0):
    """Macro-F1 with a 95% CI from resampling whole matches."""
    predicted = logits.argmax(axis=1)
    unique_matches, match_position = np.unique(match_ids, return_inverse=True)
    rows_per_match = []
    for match_number in range(len(unique_matches)):
        rows_per_match.append(np.where(match_position == match_number)[0])
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for resample in range(n_resamples):
        drawn = rng.integers(0, len(unique_matches), size=len(unique_matches))
        rows = np.concatenate([rows_per_match[m] for m in drawn])
        values[resample] = f1_score(targets[rows], predicted[rows], labels=list(range(NUM_CLASSES)),
                                    average="macro", zero_division=0)
    point = f1_score(targets, predicted, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    return {"mean": float(point),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5))}
