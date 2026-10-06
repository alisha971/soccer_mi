"""
mi_analysis/src/probing.py — residual-stream collection and linear probes with control tasks (MB05).

A probe tests whether a variable is linearly DECODABLE from activations. It does not show that
the model USES that variable; causal notebooks (MB06-MB11) are needed for that.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROBE_POINTS = ["resid_pre_0", "resid_post_0", "resid_post_1", "resid_post_2", "resid_post_3", "final_normalised"]


@torch.no_grad()
def collect(model, bundle, view, windows, positions, points, batch_size, forward_function):
    """Residual vectors at each event's scoring position: dict point -> array [N, D]."""
    collected = {point: [] for point in points}
    device = bundle["device"]
    for start in range(0, len(windows), batch_size):
        rows = torch.as_tensor(view["windows"][windows[start:start + batch_size]], device=device)
        inputs, _ = bundle["encoded"].batch(rows)
        pos = torch.as_tensor(positions[start:start + batch_size], device=device)
        _, cache = forward_function(model, inputs, capture=points)
        index = torch.arange(len(pos), device=device)
        for point in points:
            collected[point].append(cache[point][index, pos].float().cpu().numpy())
    return {point: np.concatenate(parts) for point, parts in collected.items()}


def control_labels(labels, seed):
    """Control task (Hewitt & Liang 2019): each label value is re-assigned to a random class drawn
    from the empirical class distribution, so the control has the same number of types but no
    relation to what the activations encode. Re-drawn (next seed) if every type lands in one class."""
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        mapping = {}
        for value in values:
            mapping[value] = rng.choice(values, p=probabilities)
        control = np.array([mapping[value] for value in labels])
        if len(np.unique(control)) > 1:
            return control
    return control


def fit_probe(features, labels, groups, c_values, max_iter=300):
    """Standardised multinomial logistic regression; C chosen by 3-fold CV grouped by match."""
    best_c = None
    best_score = -1.0
    n_folds = min(3, len(np.unique(groups)))
    for c in c_values:
        scores = []
        for train_index, test_index in GroupKFold(n_splits=n_folds).split(features, labels, groups):
            if len(np.unique(labels[train_index])) < 2:
                continue
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=max_iter))
            probe.fit(features[train_index], labels[train_index])
            scores.append(probe.score(features[test_index], labels[test_index]))
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score:
            best_score = score
            best_c = c
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=max_iter))
    probe.fit(features, labels)
    return probe, best_c, best_score
