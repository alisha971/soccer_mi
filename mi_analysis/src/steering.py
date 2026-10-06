"""
mi_analysis/src/steering.py — batch-invariant steering and pre-specified calibration.
REPAIR 2026-10-04 for review findings F05 and F06 (see REPAIR_REPORT.md). Used by the corrected
sections of MB11c (and referenced by MB12 / MB16 annotations).

Scale (F06)
-----------
The original steer() multiplied beta by the mean target-residual norm OF THE CURRENT BATCH, so the push
an event received depended on which other events shared its batch. Here the scale is a single number,
`reference_scale`, computed once on a stated reference table (the discovery half, which also supplies
the direction) and then frozen. Every event receives exactly beta * scale * unit, whatever its batch.

Why this scale: it keeps the original unit ("beta = multiples of the mean final-residual norm at the
target"), so corrected and original numbers differ only by the batching defect; and it makes the
intervention one fixed vector, which is what a claim about a direction is about. A per-event scale
would make the push depend on each event's own state. MB16 section 9 offers a second fixed unit
(the mean norm of a real player-swap change); `convert_units` converts between them.

Calibration (F05)
-----------------
* beta = 0 is always on the grid; its change is exactly 0 by construction and is checked.
* The local regime is measured, not assumed: the secant slope y(beta)/beta is compared with the
  slope at 0 estimated from the smallest symmetric pair; the linear regime is the set of betas whose
  secant slope lies within LINEARITY_TOLERANCE (relative) of it. The tolerance is a module constant,
  fixed before any result.
* Candidate models are fitted on the DISCOVERY-half mean curve only: a line through the origin fitted
  inside the linear regime, a quadratic through the origin, and piecewise-linear interpolation through
  the measured points (including (0, 0)). The model used for prediction is chosen on discovery data by
  leave-one-beta-out error, before the estimation half is read.
* PASS criterion (fixed here, before any result): on the estimation half, at every evaluated beta,
  the prediction lies inside the 95% match-bootstrap interval of the measured mean change. Anything
  else is reported as FAILED calibration.
"""
import numpy as np
import torch

import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics
import mi_src.mi_setup as mi_setup

REPAIR_VERSION = "2026-10-04"
LINEARITY_TOLERANCE = 0.10
DEFAULT_BETAS = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
PASS_RULE = ("on the estimation half, at every evaluated beta, the model's prediction (fitted on the discovery "
             "half only) lies inside the 95% match-bootstrap interval of the measured mean change in Delta")


@torch.no_grad()
def reference_scale(model, bundle, view, table, site, batch_size):
    """Mean norm of `site` at each event's target over `table`. Computed once and frozen."""
    norms = []
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        _, cache = hooks.forward(model, inputs, capture=[site])
        rows = torch.arange(len(positions), device=positions.device)
        norms.append(cache[site][rows, positions].norm(dim=-1).cpu().numpy())
    return float(np.concatenate(norms).mean())


@torch.no_grad()
def steer_changes(model, bundle, view, table, direction, betas, scale, site, batch_size):
    """{beta: per-event change in Delta} when beta * scale * unit(direction) is added at the target.

    `scale` is a fixed float, so the push on an event does not depend on its batch (F06).
    """
    device = bundle["device"]
    unit = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=torch.float32, device=device)
    unit = unit / unit.norm()
    out = {float(beta): np.empty(len(table)) for beta in betas}
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        rows = torch.arange(len(positions), device=device)
        base_logits, cache = hooks.forward(model, inputs, capture=[site])
        base = mi_metrics.delta(mi_metrics.at_target(base_logits, positions), a_ctx, a_id)
        for beta in betas:
            shift = torch.zeros_like(cache[site])
            shift[rows, positions] = float(beta) * float(scale) * unit.to(shift.dtype)
            logits, _ = hooks.forward(model, inputs, replace={site: hooks.add_at(shift)})
            out[float(beta)][part] = mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id) - base
    return out


@torch.no_grad()
def steer_changes_batch_mean_legacy(model, bundle, view, table, direction, betas, site, batch_size):
    """DEFECTIVE ORIGINAL scale rule (F06), kept only to MEASURE how much batching changed the result.
    The scale is the mean target-residual norm of the current batch."""
    device = bundle["device"]
    unit = torch.as_tensor(np.asarray(direction, dtype=np.float64), dtype=torch.float32, device=device)
    unit = unit / unit.norm()
    out = {float(beta): np.empty(len(table)) for beta in betas}
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        rows = torch.arange(len(positions), device=device)
        base_logits, cache = hooks.forward(model, inputs, capture=[site])
        base = mi_metrics.delta(mi_metrics.at_target(base_logits, positions), a_ctx, a_id)
        scale = cache[site][rows, positions].norm(dim=-1, keepdim=True).mean()
        for beta in betas:
            shift = torch.zeros_like(cache[site])
            shift[rows, positions] = float(beta) * scale * unit.to(shift.dtype)
            logits, _ = hooks.forward(model, inputs, replace={site: hooks.add_at(shift)})
            out[float(beta)][part] = mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id) - base
    return out


def curve(changes, match_ids, n_resamples):
    """Sorted arrays (betas, mean, ci_low, ci_high) from steer_changes output."""
    betas = np.array(sorted(changes))
    stats = [mi_metrics.summarise(changes[b], match_ids, n_resamples) for b in betas]
    return (betas, np.array([s["mean"] for s in stats]), np.array([s["ci_low"] for s in stats]),
            np.array([s["ci_high"] for s in stats]))


def local_linearity(betas, means, tolerance=LINEARITY_TOLERANCE):
    """Slope at 0 from the smallest symmetric pair, each beta's secant slope y/beta, and whether it lies
    within `tolerance` (relative) of the slope at 0. Returns (slope0, table rows, widest linear |beta|)."""
    betas, means = np.asarray(betas, dtype=float), np.asarray(means, dtype=float)
    positive = sorted(b for b in betas if b > 0 and -b in betas)
    if not positive:
        raise ValueError("local_linearity needs at least one symmetric pair of betas")
    smallest = positive[0]
    y = dict(zip(betas, means))
    slope0 = (y[smallest] - y[-smallest]) / (2 * smallest)
    rows, widest = [], 0.0
    for b in betas:
        if b == 0:
            continue
        secant = y[b] / b
        relative = abs(secant - slope0) / abs(slope0) if slope0 != 0 else np.inf
        rows.append({"beta": float(b), "secant slope y/beta": float(secant), "relative difference from slope at 0": float(relative),
                     "within tolerance": bool(relative <= tolerance)})
    for magnitude in sorted({abs(b) for b in betas if b != 0}):
        inside = all(r["within tolerance"] for r in rows if abs(r["beta"]) <= magnitude)
        if inside:
            widest = magnitude
        else:
            break
    return float(slope0), rows, float(widest)


def fit_models(betas, means, linear_range):
    """Candidate calibration models fitted on ONE curve. Every model passes through (0, 0)."""
    betas, means = np.asarray(betas, dtype=float), np.asarray(means, dtype=float)
    models = {}
    near = (np.abs(betas) <= linear_range) & (betas != 0)
    if near.any():
        slope = float(betas[near] @ means[near] / (betas[near] @ betas[near]))
        models["line through 0 (linear regime)"] = lambda b, s=slope: s * np.asarray(b, dtype=float)
    design = np.stack([betas, betas ** 2], axis=1)
    coefficients = np.linalg.lstsq(design, means, rcond=None)[0]
    models["quadratic through 0"] = lambda b, c=coefficients: c[0] * np.asarray(b, dtype=float) + c[1] * np.asarray(b, dtype=float) ** 2
    xs, ys = betas.copy(), means.copy()
    if 0.0 not in xs:
        xs, ys = np.append(xs, 0.0), np.append(ys, 0.0)
    order = np.argsort(xs)
    models["piecewise-linear through measured points"] = lambda b, x=xs[order], y=ys[order]: np.interp(np.asarray(b, dtype=float), x, y)
    return models


def leave_one_beta_out_error(betas, means, linear_range):
    """Discovery-only model selection: RMSE over held-out betas (beta = 0 is never held out)."""
    betas, means = np.asarray(betas, dtype=float), np.asarray(means, dtype=float)
    errors = {}
    for index, held in enumerate(betas):
        if held == 0:
            continue
        keep = np.arange(len(betas)) != index
        for name, model in fit_models(betas[keep], means[keep], linear_range).items():
            errors.setdefault(name, []).append(float(model(held) - means[index]))
    return {name: float(np.sqrt(np.mean(np.square(values)))) for name, values in errors.items()}


def evaluate(model, betas, means, ci_low, ci_high):
    predicted = model(betas)
    inside = (ci_low <= predicted) & (predicted <= ci_high)
    return {"predicted": predicted, "inside": inside, "rmse": float(np.sqrt(np.mean((predicted - means) ** 2))),
            "largest error": float(np.max(np.abs(predicted - means))), "share inside": float(inside.mean()),
            "PASS (pre-specified rule)": bool(inside.all())}


def convert_units(beta, from_scale, to_scale):
    """beta in units of from_scale -> the same push in units of to_scale."""
    return float(beta) * float(from_scale) / float(to_scale)
