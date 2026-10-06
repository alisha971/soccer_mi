"""
mi_analysis/src/twotoken.py — the two-token event Transformer with in-context player identity (MB22/MB23).

Written by MB22. Three variants share one code path:

  single          one token per event, every SEL-4L input (identity and history fields).
                  The control: SEL-4L's design, trained with this code.
  two_token_id    two tokens per event; the history fields are removed from the context token and
                  carried by the previous event's action token; the player keeps a global embedding.
  two_token_anon  as two_token_id, but the player has NO global embedding. Each window gives its players
                  random anonymous labels ("slots"), so the model can tell which events belong to the same
                  player, but not which player it is. Identity can only come from that player's earlier
                  events in the window, through attention. This is the main model.

Sequence layout of the two-token variants (K events -> 2K tokens):
  c_1, a_1, c_2, a_2, ..., c_K, a_K
  c_t (context token): pre-action fields of event t (team, role, competition, space, time, state) + player.
  a_t (action token):  what event t turned out to be: its action, outcome and end zone + player + team.
                       These are read from event t+1's previous-event fields, i.e. exactly the information
                       SEL-4L gives to token t+1. The prediction of event t's action is made at c_t, which
                       (causal mask) cannot see a_t.

The instrumented forward(model, inputs, replace, capture) mirrors mi_src.hooks.forward, with the same site
names, so the MI code of MB06-MB19 carries over.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import FACTOR_GROUPS, CONTINUOUS_GROUPS

HISTORY_FEATURES = ["prev_action", "prev_outcome", "prev_end_zone", "same_player_as_prev"]
ACTION_FIELDS = ["prev_action", "prev_outcome", "prev_end_zone"]        # read from event t+1
VARIANTS = ["single", "two_token_id", "two_token_anon"]
CONTEXT_GROUPS = ["team", "role", "competition", "space", "time", "state"]


class Config:
    def __init__(self, variant, sizes, n_events=32, n_layers=4, n_heads=2, d_model=128, d_ff=512, n_slots=None):
        assert variant in VARIANTS, variant
        self.variant, self.sizes = variant, dict(sizes)
        self.n_events, self.n_layers, self.n_heads, self.d_model, self.d_ff = n_events, n_layers, n_heads, d_model, d_ff
        self.n_slots = n_slots if n_slots is not None else n_events + 8
        self.two_token = variant != "single"
        self.n_tokens = 2 * n_events if self.two_token else n_events

    def to_dict(self):
        return {k: getattr(self, k) for k in ["variant", "sizes", "n_events", "n_layers", "n_heads", "d_model", "d_ff", "n_slots"]}

    @staticmethod
    def from_dict(d):
        return Config(d["variant"], d["sizes"], d["n_events"], d["n_layers"], d["n_heads"], d["d_model"], d["d_ff"], d["n_slots"])


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.d_head = c.n_heads, c.d_model // c.n_heads
        self.W_Q, self.W_K, self.W_V, self.W_O = (nn.Linear(c.d_model, c.d_model) for _ in range(4))
        self.register_buffer("causal_mask", torch.tril(torch.ones(c.n_tokens, c.n_tokens, dtype=torch.bool)))

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_attention, self.attention = nn.LayerNorm(c.d_model), Attention(c)
        self.ln_mlp = nn.LayerNorm(c.d_model)
        self.mlp = nn.Module()
        self.mlp.input_layer, self.mlp.output_layer = nn.Linear(c.d_model, c.d_ff), nn.Linear(c.d_ff, c.d_model)


class TwoTokenModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        D = c.d_model
        self.embeddings = nn.ModuleDict()
        groups = (["player"] if c.variant != "two_token_anon" else []) + CONTEXT_GROUPS + (["history"] if c.variant == "single" else [])
        self.context_groups = groups
        for group in groups:
            for feature in FACTOR_GROUPS[group]:
                self.embeddings[feature] = nn.Embedding(c.sizes[feature], D)
        self.continuous = nn.ModuleDict({g: nn.Linear(len(v), D, bias=False) for g, v in CONTINUOUS_GROUPS.items() if g in groups})
        if c.variant == "two_token_anon":
            self.slot_embedding = nn.Embedding(c.n_slots, D)
        if c.two_token:
            self.action_embeddings = nn.ModuleDict({f: nn.Embedding(c.sizes[f], D) for f in ACTION_FIELDS})
            self.type_embedding = nn.Embedding(2, D)
        self.position_embedding = nn.Embedding(c.n_events, D)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_final = nn.LayerNorm(D)
        self.unembed = nn.Linear(D, 15)
        with torch.no_grad():
            if "player" in self.embeddings:
                self.embeddings["player"].weight[0].zero_()
            if c.variant == "two_token_anon":
                self.slot_embedding.weight[0].zero_()

    def context_positions(self):
        c = self.config
        return torch.arange(c.n_events) * (2 if c.two_token else 1)


def _group_vector(model, inputs, group):
    vector = None
    for feature in FACTOR_GROUPS[group]:
        v = model.embeddings[feature](inputs[feature])
        vector = v if vector is None else vector + v
    if group in model.continuous:
        values = torch.stack([inputs[name] for name in CONTINUOUS_GROUPS[group]], dim=-1)
        vector = vector + model.continuous[group](values)
    return vector


def embed_factors(model, inputs):
    """{site name: [B, T, D]} — one vector per input group, zero at positions where the group is absent."""
    c = model.config
    B, K = inputs["zone"].shape
    D = c.d_model
    out = {}

    def place(event_vectors, kind):
        if not c.two_token:
            return event_vectors
        full = event_vectors.new_zeros(B, 2 * K, D)
        full[:, (0 if kind == "context" else 1)::2] = event_vectors
        return full

    identity_name = "slot" if c.variant == "two_token_anon" else "player"
    for group in model.context_groups:
        name = identity_name if group == "player" else group
        out[f"factor_{name}"] = place(_group_vector(model, inputs, group), "context")
    if c.variant == "two_token_anon":
        slot_vec = model.slot_embedding(inputs["slot"])
        out["factor_slot"] = place(slot_vec, "context")
    if c.two_token:
        act = None
        for f in ACTION_FIELDS:
            v = model.action_embeddings[f](inputs[f"act_{f}"])
            act = v if act is None else act + v
        out["factor_act_action"] = place(act, "action")
        ident = model.slot_embedding(inputs["slot"]) if c.variant == "two_token_anon" else model.embeddings["player"](inputs["player"])
        out[f"factor_act_{identity_name}"] = place(ident, "action")
        out["factor_act_team"] = place(model.embeddings["team"](inputs["team"]), "action")
        types = torch.arange(2 * K, device=inputs["zone"].device) % 2
        out["factor_type"] = model.type_embedding(types).unsqueeze(0).expand(B, -1, -1)
        positions = torch.arange(2 * K, device=inputs["zone"].device) // 2
    else:
        positions = torch.arange(K, device=inputs["zone"].device)
    out["factor_position"] = model.position_embedding(positions).unsqueeze(0).expand(B, -1, -1)
    return out


def forward(model, inputs, replace=None, capture=None):
    """(logits [B, T, 15], cache). Same site names as mi_src.hooks.forward."""
    replace = replace or {}
    cache = {}

    def site(name, value):
        if name in replace:
            value = replace[name](value)
        if capture is not None and (capture == "all" or name in capture):
            cache[name] = value.detach()
        return value

    residual = None
    for name, vector in embed_factors(model, inputs).items():
        vector = site(name, vector)
        residual = vector if residual is None else residual + vector
    residual = site("resid_pre_0", residual)
    B, T, _ = residual.shape
    for layer, block in enumerate(model.blocks):
        if layer > 0:
            residual = site(f"resid_pre_{layer}", residual)
        attention = block.attention
        shared = block.ln_attention(residual)
        q, k, v = (attention.split_heads(p(shared)) for p in (attention.W_Q, attention.W_K, attention.W_V))
        for head in range(attention.n_heads):
            for kind, projection, tensor in [("q", attention.W_Q, q), ("k", attention.W_K, k), ("v", attention.W_V, v)]:
                name = f"attn_in_{kind}_{layer}.{head}"
                if name in replace or (capture is not None and capture != "all" and name in capture):
                    receiver_input = site(name, residual)
                    if name in replace:
                        tensor[:, head] = attention.split_heads(projection(block.ln_attention(receiver_input)))[:, head]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attention.d_head)
        scores = scores.masked_fill(~attention.causal_mask[:T, :T], float("-inf"))
        scores = site(f"attn_scores_{layer}", scores)
        pattern = site(f"attn_pattern_{layer}", torch.softmax(scores, dim=-1))
        z = site(f"head_z_{layer}", torch.matmul(pattern, v).transpose(1, 2))
        attn_out = site(f"attn_out_{layer}", attention.W_O(z.reshape(B, T, -1)))
        residual = site(f"resid_mid_{layer}", residual + attn_out)
        mlp_in = site(f"mlp_in_{layer}", residual)
        hidden = site(f"mlp_hidden_{layer}", F.gelu(block.mlp.input_layer(block.ln_mlp(mlp_in))))
        mlp_out = site(f"mlp_out_{layer}", block.mlp.output_layer(hidden))
        residual = site(f"resid_post_{layer}", residual + mlp_out)
    final = site("final_resid", residual)
    normalised = site("final_normalised", model.ln_final(final))
    return model.unembed(normalised), cache


# ----------------------------------------------------------------------------------------------- data

class EventData:
    """Per-event arrays (model's own T300 encoding) and the windows of each split.

    events: events_prepared (row order = time order within a match); encoded: {feature: tensor over all events}.
    """

    def __init__(self, events, encoded, targets, n_events, stride=None):
        self.n_events = n_events
        self.stride = stride or n_events // 2
        self.features = {name: tensor.detach().cpu().numpy() for name, tensor in encoded.items()}
        self.targets = np.asarray(targets).astype(np.int64)
        match = events["match_id"].to_numpy()
        n = len(events)
        same_next = np.zeros(n, dtype=bool)
        same_next[:-1] = match[1:] == match[:-1]
        for f in ACTION_FIELDS:
            nxt = np.zeros(n, dtype=self.features[f].dtype)
            nxt[:-1] = self.features[f][1:]
            self.features[f"act_{f}"] = np.where(same_next, nxt, 0)
        self.match = match
        self.split = events["split"].to_numpy()
        self.windows, self.scored = {}, {}
        for split in np.unique(self.split):
            self._build(split)

    def _build(self, split):
        K, S = self.n_events, self.stride
        rows_all = np.where(self.split == split)[0]
        windows, scored_rows, scored_window, scored_pos = [], [], [], []
        for m in np.unique(self.match[rows_all]):
            rows = rows_all[self.match[rows_all] == m]
            if len(rows) < K:
                continue
            starts = list(range(0, len(rows) - K + 1, S))
            if starts[-1] != len(rows) - K:
                starts.append(len(rows) - K)
            first = len(windows)
            for s in starts:
                windows.append(rows[s:s + K])
            starts = np.array(starts)
            for i, r in enumerate(rows):
                ok = np.where((starts <= i) & (i < starts + K))[0]
                best = ok[np.argmax(i - starts[ok])]
                scored_rows.append(r); scored_window.append(first + best); scored_pos.append(i - starts[best])
        self.windows[split] = np.array(windows)
        self.scored[split] = {"event_row": np.array(scored_rows), "window": np.array(scored_window), "position": np.array(scored_pos)}

    def batch(self, split, window_ids, device, slot_rng=None, n_slots=None):
        rows = self.windows[split][np.asarray(window_ids)]
        out = {}
        for name, values in self.features.items():
            dtype = torch.float32 if values.dtype.kind == "f" else torch.long
            out[name] = torch.as_tensor(values[rows], dtype=dtype, device=device)
        players = self.features["player"][rows]
        slots = np.zeros_like(players)
        rng = slot_rng if slot_rng is not None else np.random.default_rng(0)
        for b in range(len(rows)):
            unique = [p for p in dict.fromkeys(players[b].tolist()) if p > 0]
            labels = rng.permutation(np.arange(1, n_slots))[:len(unique)]
            lookup = dict(zip(unique, labels))
            slots[b] = [lookup.get(p, 0) for p in players[b]]
        out["slot"] = torch.as_tensor(slots, dtype=torch.long, device=device)
        return out, torch.as_tensor(self.targets[rows], dtype=torch.long, device=device)

    def eval_slot_rng(self, split, window_id):
        """Deterministic anonymous labels for evaluation: one fixed permutation per window."""
        return np.random.default_rng(1_000_003 * {"train": 1, "val": 2, "test": 3}.get(str(split), 0) + int(window_id))


def eval_batch(data, split, window_ids, device, n_slots):
    """Evaluation inputs with fixed slots per window (same labels every time a window is used)."""
    parts, targets = [], []
    for w in np.asarray(window_ids):
        inputs, t = data.batch(split, [w], device, slot_rng=data.eval_slot_rng(split, w), n_slots=n_slots)
        parts.append(inputs); targets.append(t)
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}, torch.cat(targets)


def context_index(model, event_positions):
    """Token index of the context token of each event position."""
    return event_positions * (2 if model.config.two_token else 1)


def logits_at_events(logits, model):
    """[B, K, 15] logits read at every event's context token."""
    return logits[:, ::2] if model.config.two_token else logits
