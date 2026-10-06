"""
mi_analysis/src/mi_metrics.py — the arbitration metric and the statistics used by every notebook (MB02).

Uncertainty always comes from resampling whole MATCHES (events in one match are not independent),
exactly as src/metrics.match_bootstrap_mean does in model_training_evaluation.

REPAIR 2026-10-04 (F10, see REPAIR_REPORT.md): added probability_calibration, aggregate_log_odds and
reliability_bins, which compare predicted probabilities with realised frequencies at the SAME level of
aggregation. mean(log(p_ctx / p_id)) (the mean of Delta) is not comparable with log(n_ctx / n_id): even a
perfectly calibrated model makes them differ whenever probabilities vary across events.
Existing functions are unchanged.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

import src.metrics as track_a_metrics

REPAIR_VERSION = "2026-10-04"


def delta(logits_at_target, a_ctx, a_id):
    """Delta = logit(a_ctx) - logit(a_id) = log P(a_ctx) - log P(a_id) (softmax denominator cancels)."""
    if isinstance(logits_at_target, torch.Tensor):
        rows = torch.arange(logits_at_target.shape[0], device=logits_at_target.device)
        return (logits_at_target[rows, a_ctx] - logits_at_target[rows, a_id]).detach().cpu().numpy()
    rows = np.arange(len(logits_at_target))
    return logits_at_target[rows, a_ctx] - logits_at_target[rows, a_id]


def at_target(logits, positions):
    """Logits [B, T, C] -> [B, C] read at one position per row."""
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, positions]


def log_softmax(logits):
    return track_a_metrics.log_softmax(np.asarray(logits, dtype=np.float64))


def kl_divergence(logits_p, logits_q):
    """KL(p || q) per row, in nats, for two sets of logits over the same 15 actions."""
    log_p = log_softmax(logits_p)
    log_q = log_softmax(logits_q)
    return (np.exp(log_p) * (log_p - log_q)).sum(axis=1)


def resampled_means(values, match_ids, n_resamples, seed=0):
    """Point mean and the match-bootstrap distribution of the mean (same scheme as model_training_evaluation)."""
    values = np.asarray(values, dtype=np.float64)
    unique_matches, match_position = np.unique(match_ids, return_inverse=True)
    n_matches = len(unique_matches)
    sums = np.bincount(match_position, weights=values, minlength=n_matches)
    counts = np.bincount(match_position, minlength=n_matches).astype(float)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, n_matches, size=(n_resamples, n_matches))
    draws = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return float(values.mean()), draws


def summarise(values, match_ids, n_resamples, seed=0):
    """n, mean, SD and 95% match-bootstrap interval of per-event values."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_matches": 0}
    point, draws = resampled_means(values, match_ids, n_resamples, seed)
    if len(np.unique(match_ids)) < 2:
        draws = np.full_like(draws, np.nan)
    return {"n": int(len(values)), "n_matches": int(len(np.unique(match_ids))), "mean": point,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5))}


def bootstrap_pvalue(values, match_ids, n_resamples, seed=0, alternative="two-sided"):
    """Share of match-bootstrap means on the other side of 0 (two-sided: doubled, capped at 1).

    With few resamples the smallest attainable value is 1 / (n_resamples + 1).
    """
    if len(np.unique(match_ids)) < 2:
        return float("nan")
    _, draws = resampled_means(values, match_ids, n_resamples, seed)
    below = (np.sum(draws <= 0) + 1) / (n_resamples + 1)
    above = (np.sum(draws >= 0) + 1) / (n_resamples + 1)
    if alternative == "greater":
        return float(below)
    if alternative == "less":
        return float(above)
    return float(min(1.0, 2 * min(below, above)))


def aligned(numerator, denominator):
    """numerator * sign(denominator): a change counted as positive when it points the same way as
    the event's own clean-minus-corrupted difference."""
    return np.asarray(numerator, dtype=np.float64) * np.sign(np.asarray(denominator, dtype=np.float64))


def pooled_ratio(numerator, denominator):
    """Point value of the sign-aligned pooled ratio  sum(sign(d_i) n_i) / sum(|d_i|)."""
    numerator = aligned(numerator, denominator)
    bottom = np.abs(np.asarray(denominator, dtype=np.float64)).sum()
    return float(numerator.sum() / bottom) if bottom > 0 else np.nan


def pooled_recovery(numerator, denominator, match_ids, n_resamples, seed=0):
    """Sign-aligned ratio of sums  R = sum(sign(d_i) n_i) / sum(|d_i|)  with a match-bootstrap interval.

    Used for recovery (n = m_patched - m_corrupted, d = m_clean - m_corrupted) and for shares such as
    faithfulness. Aligning each event by the sign of its own d keeps the denominator away from 0
    when the clean-minus-corrupted difference points in different directions on different events
    (a plain sum of d can then cancel to nearly 0 and make the ratio unstable). R = 1 when every
    event's difference is fully restored, 0 when none is. No per-event cut-off is needed.
    """
    numerator = aligned(numerator, denominator)
    denominator = np.abs(np.asarray(denominator, dtype=np.float64))
    unique_matches, match_position = np.unique(match_ids, return_inverse=True)
    n_matches = len(unique_matches)
    if n_matches < 2:
        point = numerator.sum() / denominator.sum() if denominator.sum() else np.nan
        return {"recovery": float(point), "ci_low": np.nan, "ci_high": np.nan}
    top = np.bincount(match_position, weights=numerator, minlength=n_matches)
    bottom = np.bincount(match_position, weights=denominator, minlength=n_matches)
    point = numerator.sum() / denominator.sum() if denominator.sum() != 0 else np.nan
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, n_matches, size=(n_resamples, n_matches))
    top_sums, bottom_sums = top[chosen].sum(axis=1), bottom[chosen].sum(axis=1)
    draws = top_sums[bottom_sums != 0] / bottom_sums[bottom_sums != 0]
    return {"recovery": float(point),
            "ci_low": float(np.percentile(draws, 2.5)) if len(draws) else np.nan,
            "ci_high": float(np.percentile(draws, 97.5)) if len(draws) else np.nan}


def benjamini_hochberg(pvalues, q):
    """Benjamini-Hochberg: (reject flags, adjusted q-values) at false-discovery level q."""
    pvalues = np.asarray(pvalues, dtype=np.float64)
    valid = np.isfinite(pvalues)
    pvalues = np.where(valid, pvalues, 1.0)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted <= q, adjusted


def spearman(a, b):
    return float(spearmanr(a, b).correlation)

def _match_sums(values, match_ids):
    unique_matches, match_position = np.unique(match_ids, return_inverse=True)
    return np.bincount(match_position, weights=np.asarray(values, dtype=np.float64), minlength=len(unique_matches))


def probability_calibration(p, realised, match_ids, n_resamples, seed=0):
    """Mean predicted probability of an action vs the share of events on which it was realised.

    p         predicted probability of the action on each event
    realised  1 if that action happened, else 0
    The difference mean(p) - mean(realised) is a per-event paired mean, so its match bootstrap is valid.
    A calibrated model has difference ~0 (calibration-in-the-large for that action on that population).
    """
    p = np.asarray(p, dtype=np.float64)
    realised = np.asarray(realised, dtype=np.float64)
    difference = summarise(p - realised, match_ids, n_resamples, seed)
    return {"n": int(len(p)), "mean predicted": float(p.mean()), "realised share": float(realised.mean()),
            "predicted minus realised": difference["mean"], "CI low": difference["ci_low"], "CI high": difference["ci_high"]}


def aggregate_log_odds(p_ctx, p_id, realised_ctx, realised_id, match_ids, n_resamples, seed=0):
    """Model and data log-odds of a_ctx over a_id at the SAME aggregation level.

    model  log(sum p_ctx / sum p_id)          (expected counts under the model)
    data   log(n(realised = a_ctx) / n(realised = a_id))
    Both are ratios of sums over the same events, so they are directly comparable; the difference gets a
    match bootstrap (ratio of resampled sums). No 0.5 smoothing: draws with a zero count are dropped and
    the number dropped is reported.
    """
    sums = {name: _match_sums(values, match_ids) for name, values in
            [("p_ctx", p_ctx), ("p_id", p_id), ("n_ctx", realised_ctx), ("n_id", realised_id)]}
    rng = np.random.default_rng(seed)
    n_matches = len(sums["p_ctx"])
    chosen = rng.integers(0, n_matches, size=(n_resamples, n_matches))
    draws = {name: values[chosen].sum(axis=1) for name, values in sums.items()}
    valid = (draws["n_ctx"] > 0) & (draws["n_id"] > 0)
    model = np.log(draws["p_ctx"] / draws["p_id"])
    data = np.where(valid, np.log(np.where(valid, draws["n_ctx"], 1) / np.where(valid, draws["n_id"], 1)), np.nan)
    point_model = float(np.log(sums["p_ctx"].sum() / sums["p_id"].sum()))
    point_data = float(np.log(sums["n_ctx"].sum() / sums["n_id"].sum())) if sums["n_ctx"].sum() > 0 and sums["n_id"].sum() > 0 else np.nan
    difference = (model - data)[valid]
    return {"model aggregate log-odds": point_model, "data log-odds": point_data,
            "model minus data": point_model - point_data,
            "CI low": float(np.percentile(difference, 2.5)) if len(difference) else np.nan,
            "CI high": float(np.percentile(difference, 97.5)) if len(difference) else np.nan,
            "bootstrap draws dropped (zero count)": int((~valid).sum())}


def reliability_bins(p, realised, n_bins=10):
    """Equal-count bins of predicted probability: mean predicted vs realised share per bin."""
    import pandas as pd
    p = np.asarray(p, dtype=np.float64)
    realised = np.asarray(realised, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for index, members in enumerate(bins):
        if len(members) == 0:
            continue
        rows.append({"bin": index, "n": int(len(members)), "mean predicted": float(p[members].mean()),
                     "realised share": float(realised[members].mean()),
                     "p range": f"{p[members].min():.3f}-{p[members].max():.3f}"})
    return pd.DataFrame(rows)
