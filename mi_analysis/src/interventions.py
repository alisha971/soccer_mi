"""
mi_analysis/src/interventions.py — input interventions with scope checks (MB02).

An intervention changes model INPUTS (integer indices) at stated positions and nothing else.
Every function returns a modified copy; `changed_mask` and `assert_scope` prove afterwards that
only the declared feature at the declared positions differs from the original.
"""
import numpy as np
import torch

IDENTITY_FEATURES = ["player", "team", "role", "competition"]


def clone_inputs(inputs):
    copy = {}
    for name, tensor in inputs.items():
        copy[name] = tensor.clone()
    return copy


def set_feature(inputs, feature, positions, new_values):
    """Set inputs[feature][row, positions[row]] = new_values[row] for every batch row."""
    modified = clone_inputs(inputs)
    rows = torch.arange(len(positions), device=positions.device)
    modified[feature][rows, positions] = new_values.to(modified[feature].dtype)
    return modified


def remove_feature(inputs, feature, positions):
    """Index 0 = the model's own 'no dedicated identity' vector (a fixed zero vector)."""
    zeros = torch.zeros(len(positions), dtype=torch.long, device=positions.device)
    return set_feature(inputs, feature, positions, zeros)


def remove_feature_all_occurrences(inputs, feature, positions):
    """Set the feature to 0 at every position of the window where the target's value appears."""
    modified = clone_inputs(inputs)
    rows = torch.arange(len(positions), device=positions.device)
    target_values = inputs[feature][rows, positions]
    same = (inputs[feature] == target_values.unsqueeze(1)) & (target_values.unsqueeze(1) > 0)
    modified[feature][same] = 0
    return modified


def write_identity(target_inputs, source_inputs, positions, features):
    """Copy the given identity features at the target position from one window into another."""
    modified = clone_inputs(target_inputs)
    rows = torch.arange(len(positions), device=positions.device)
    for feature in features:
        modified[feature][rows, positions] = source_inputs[feature][rows, positions]
    return modified


def changed_mask(original, modified):
    """{feature: bool tensor [B, T]} of the entries that differ."""
    masks = {}
    for name in original:
        masks[name] = original[name] != modified[name]
    return masks


def assert_scope(original, modified, allowed_features, positions=None):
    """Raise unless every changed entry is in an allowed feature (and at the stated positions)."""
    masks = changed_mask(original, modified)
    for name, mask in masks.items():
        if not mask.any():
            continue
        if name not in allowed_features:
            raise AssertionError(f"intervention changed a feature it must not change: {name}")
        if positions is not None:
            rows = torch.arange(mask.shape[0], device=mask.device)
            allowed = torch.zeros_like(mask)
            allowed[rows, positions] = True
            if (mask & ~allowed).any():
                raise AssertionError(f"intervention changed {name} outside the target position")


def norm_matched_vector(factor_player, positions, generator):
    """Edit function for site factor_player: at the target, a random direction with the same norm.

    Rows whose player vector is zero (index 0) stay zero, so the control is a no-op there too.
    """
    def edit(value):
        new = value.clone()
        rows = torch.arange(len(positions), device=value.device)
        original = value[rows, positions]
        norms = original.norm(dim=-1, keepdim=True)
        direction = torch.randn(original.shape, generator=generator).to(value.device)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        new[rows, positions] = direction * norms
        return new
    return edit


def build_partner_table(events, features, train_mask):
    """Matched swap partner for every dedicated (T300) player.

    Same role (most frequent model_training_evaluation role index in training), same competition when such a
    player exists, otherwise same role in any competition; among candidates the one with the
    nearest log training count; ties broken by the lower player index.
    """
    import pandas as pd
    table = pd.DataFrame({"player": np.asarray(features["player"])[train_mask],
                          "role": np.asarray(features["role"])[train_mask],
                          "competition": np.asarray(features["competition"])[train_mask]})
    table = table[table["player"] > 0]
    summary = table.groupby("player").agg(
        role=("role", lambda s: s.value_counts().index[0]),
        competition=("competition", lambda s: s.value_counts().index[0]),
        train_events=("role", "size")).reset_index()
    partners = []
    for _, row in summary.iterrows():
        candidates = summary[(summary["role"] == row["role"]) & (summary["player"] != row["player"])]
        same_competition = candidates[candidates["competition"] == row["competition"]]
        rule = "same role and competition"
        if len(same_competition) == 0:
            same_competition = candidates
            rule = "same role, other competition"
        if len(same_competition) == 0:
            partners.append({"player": int(row["player"]), "partner": -1, "rule": "no partner in role",
                             "role": int(row["role"]), "competition": int(row["competition"]),
                             "train_events": int(row["train_events"]), "partner_train_events": -1})
            continue
        gap = np.abs(np.log(same_competition["train_events"].to_numpy()) - np.log(row["train_events"]))
        order = np.lexsort((same_competition["player"].to_numpy(), gap))
        chosen = same_competition.iloc[order[0]]
        partners.append({"player": int(row["player"]), "partner": int(chosen["player"]), "rule": rule,
                         "role": int(row["role"]), "competition": int(row["competition"]),
                         "train_events": int(row["train_events"]),
                         "partner_train_events": int(chosen["train_events"])})
    return pd.DataFrame(partners)
