"""
mi_analysis/src/context_edits.py — input edits for the MB13 context survey, with explicit semantics.
REPAIR 2026-10-04 for review findings F14-F18 (see REPAIR_REPORT.md).

* resample_other (F14, F15): replace a field at the target by a value drawn from the population's
  values of that field EXCLUDING the event's own value. The draw for an event depends only on
  (seed, repeat, event_row), so it is identical whatever batch the event is in, and different events
  do not share a random stream. Rows for which no other value exists are left unchanged and counted
  (realised no-op rate). These are draws from the conditional marginal p(v | v != own); the edited
  event can still be an unseen joint combination, so they are NOT called "on-distribution".
* ordinal_step / grid_step (F16): an "adjacent" move is defined per field. Only fields declared
  ordinal (with their ordered codes) or spatial (with a declared grid layout) get one; special codes
  (OTHER, END / no previous event, padding) are never edited or produced; both directions are
  reported separately and rows at a boundary are skipped, never flipped. Unordered categories get no
  adjacent edit at all.
* factor_mean_ablation (F17): a genuine activation-level ablation of a factor group: the group's
  factor vector at the target is replaced by its mean over a reference table. Setting category
  indices to 0 is NOT removal for context features (their index 0 is a learned, real category); that
  operation is named set_to_code_zero and labelled literally.
* joint_donor_values / signed_nonadditivity (F18): single and joint edits use the SAME endpoint (the
  values of one donor event), and the per-event non-additivity is the signed contrast
  joint - sum(singles); for a pair it is the factorial interaction f(AB) - f(A) - f(B) + f(0).
"""
import numpy as np
import torch

REPAIR_VERSION = "2026-10-04"


def event_uniform(seed, repeat, event_rows):
    """One uniform number in [0, 1) per event, determined only by (seed, repeat, event_row)."""
    return np.array([np.random.default_rng([int(seed), int(repeat), int(row)]).random() for row in np.asarray(event_rows)])


def resample_other(pool_values, own_values, event_rows, seed, repeat):
    """Draw, for each event, a value from pool_values that differs from its own value.

    Returns (new_values, unchanged_mask). unchanged_mask is True where no different value exists.
    """
    pool_values = np.asarray(pool_values)
    own_values = np.asarray(own_values)
    uniform = event_uniform(seed, repeat, event_rows)
    new = own_values.copy()
    unchanged = np.zeros(len(own_values), dtype=bool)
    candidates_for = {}
    for i, own in enumerate(own_values):
        key = own.item() if hasattr(own, "item") else own
        if key not in candidates_for:
            candidates_for[key] = pool_values[pool_values != own]
        candidates = candidates_for[key]
        if len(candidates) == 0:
            unchanged[i] = True
            continue
        new[i] = candidates[min(int(uniform[i] * len(candidates)), len(candidates) - 1)]
    return new, unchanged


def two_draw_noop_rate(pool_values, own_values):
    """Expected share of rows left unchanged by the ORIGINAL MB13 rule (draw; if equal, draw once more):
    mean over rows of p(own)^2, p = marginal share in the pool (F15)."""
    pool_values = np.asarray(pool_values)
    values, counts = np.unique(pool_values, return_counts=True)
    share = dict(zip(values.tolist(), (counts / counts.sum()).tolist()))
    return float(np.mean([share.get(v, 0.0) ** 2 for v in np.asarray(own_values).tolist()]))


def ordinal_step(own_values, ordered_codes, direction):
    """Move each value one step along `ordered_codes` (direction +1 or -1).

    Values not in ordered_codes (special codes) and values at the boundary in that direction are not
    moved; returns (new_values, moved_mask).
    """
    assert direction in (1, -1)
    rank = {code: i for i, code in enumerate(ordered_codes)}
    own_values = np.asarray(own_values)
    new = own_values.copy()
    moved = np.zeros(len(own_values), dtype=bool)
    for i, value in enumerate(own_values.tolist()):
        if value not in rank:
            continue
        j = rank[value] + direction
        if 0 <= j < len(ordered_codes):
            new[i] = ordered_codes[j]
            moved[i] = True
    return new, moved


def grid_step(own_values, n_rows, n_cols, axis, direction, code_of_cell, cell_of_code):
    """Move a spatial bin to its neighbour along one grid axis.

    code_of_cell(row, col) -> code and cell_of_code(code) -> (row, col) or None must come from the
    encoding that built the field (Track A data.py); they are NOT guessed here. Codes with no cell
    (special codes) and boundary cells are not moved. Returns (new_values, moved_mask).
    """
    assert axis in ("row", "col") and direction in (1, -1)
    own_values = np.asarray(own_values)
    new = own_values.copy()
    moved = np.zeros(len(own_values), dtype=bool)
    for i, value in enumerate(own_values.tolist()):
        cell = cell_of_code(value)
        if cell is None:
            continue
        row, col = cell
        row, col = (row + direction, col) if axis == "row" else (row, col + direction)
        if 0 <= row < n_rows and 0 <= col < n_cols:
            new[i] = code_of_cell(row, col)
            moved[i] = True
    return new, moved


def set_to_code_zero(inputs, names, positions, continuous_names=()):
    """Set categorical `names` to code 0 and continuous `continuous_names` to 0.0 at the target.

    LITERAL label: for context features code 0 is a learned category, so this is an edit to a chosen
    input configuration, not removal of the features (F17).
    """
    modified = {name: tensor.clone() for name, tensor in inputs.items()}
    rows = torch.arange(len(positions), device=positions.device)
    for name in names:
        modified[name][rows, positions] = 0
    for name in continuous_names:
        modified[name][rows, positions] = 0.0
    return modified


def factor_mean_ablation(mean_vector, positions):
    """Edit for site factor_<group>: replace the group's factor vector at the target by `mean_vector`
    ([D], the mean of that site at the target over a reference table). A genuine activation-level
    ablation of the whole group (F17)."""
    def edit(value):
        new = value.clone()
        rows = torch.arange(len(positions), device=value.device)
        new[rows, positions] = mean_vector.to(value.device, value.dtype)
        return new
    return edit


def joint_donor_values(features, donor_rows, names):
    """{name: donor value per event} for a set of fields, all taken from the SAME donor event."""
    return {name: np.asarray(features[name])[np.asarray(donor_rows)] for name in names}


def apply_values(inputs, positions, values, device=None):
    """Set several fields at the target to given per-event values (dict name -> array)."""
    modified = {name: tensor.clone() for name, tensor in inputs.items()}
    rows = torch.arange(len(positions), device=positions.device)
    for name, array in values.items():
        modified[name][rows, positions] = torch.as_tensor(np.asarray(array), device=positions.device).to(modified[name].dtype)
    return modified


def signed_nonadditivity(joint_change, single_changes):
    """Per-event joint - sum(singles), all SIGNED changes in Delta from the same baseline and endpoint.
    Exactly 0 for every event when the response is additive in these fields (F18)."""
    return np.asarray(joint_change, dtype=float) - np.sum([np.asarray(c, dtype=float) for c in single_changes], axis=0)


def pairwise_interaction(f_ab, f_a, f_b, f_0):
    """Factorial interaction contrast f(AB) - f(A) - f(B) + f(0), per event."""
    return np.asarray(f_ab, float) - np.asarray(f_a, float) - np.asarray(f_b, float) + np.asarray(f_0, float)


def donor_rows_for(event_rows, pool_rows, seed):
    """One donor event per event (another event of the pool), determined only by (seed, event_row)."""
    pool_rows = np.asarray(pool_rows)
    uniform = event_uniform(seed, 0, event_rows)
    donors = np.empty(len(event_rows), dtype=np.int64)
    for i, (row, u) in enumerate(zip(np.asarray(event_rows), uniform)):
        others = pool_rows[pool_rows != row]
        donors[i] = others[min(int(u * len(others)), len(others) - 1)]
    return donors
