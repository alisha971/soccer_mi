"""
mi_analysis/src/attention.py — attention-pattern statistics (MB07).

Attention weights describe where a head looks; they are not evidence that the head matters.
Ablation and patching decide that.
"""
import torch

RELATIONS = ["self", "previous event", "same player (earlier)", "same team (earlier)", "after a team change (earlier)",
             "first position"]


def query_rows(pattern, positions):
    """Attention probabilities of the target query: [B, H, T] from a [B, H, T, T] pattern."""
    rows = torch.arange(pattern.shape[0], device=pattern.device)
    return pattern[rows, :, positions, :]


def relation_masks(inputs, positions):
    """{relation: bool [B, T]} marking the key positions that stand in each relation to the target."""
    batch_size, n_positions = inputs["player"].shape
    device = inputs["player"].device
    key = torch.arange(n_positions, device=device).unsqueeze(0).expand(batch_size, -1)
    target = positions.unsqueeze(1)
    earlier = key < target
    rows = torch.arange(batch_size, device=device)
    player = inputs["player"][rows, positions].unsqueeze(1)
    team = inputs["team"][rows, positions].unsqueeze(1)
    team_change = torch.zeros_like(earlier)
    team_change[:, 1:] = inputs["team"][:, 1:] != inputs["team"][:, :-1]
    return {
        "self": key == target,
        "previous event": key == target - 1,
        "same player (earlier)": earlier & (inputs["player"] == player) & (player > 0),
        "same team (earlier)": earlier & (inputs["team"] == team) & (team > 0),
        "after a team change (earlier)": earlier & team_change,
        "first position": (key == 0) & (target > 0),
    }


def relation_shares(rows, masks, positions):
    """Attention share per relation and head, and the share uniform attention would give.

    rows [B, H, T]; returns {relation: (attended [B, H], uniform [B])}.
    """
    allowed = (positions + 1).float()
    shares = {}
    for relation, mask in masks.items():
        attended = (rows * mask.unsqueeze(1).float()).sum(dim=-1)
        uniform = mask.float().sum(dim=-1) / allowed
        shares[relation] = (attended, uniform)
    return shares


def offset_profile(rows, positions, n_positions):
    """Attention to the event k steps back (k = 0 .. T-1): [B, H, T]; NaN where k > target."""
    batch_size, n_heads, _ = rows.shape
    profile = torch.full((batch_size, n_heads, n_positions), float("nan"), device=rows.device)
    for k in range(n_positions):
        key = positions - k
        valid = key >= 0
        index = torch.nonzero(valid).flatten()
        profile[index, :, k] = rows[index, :, key[valid]]
    return profile


def row_kl(rows_p, rows_q, eps=1e-12):
    """KL(p || q) between two attention rows, per event and head."""
    return (rows_p * (torch.log(rows_p + eps) - torch.log(rows_q + eps))).sum(dim=-1)
