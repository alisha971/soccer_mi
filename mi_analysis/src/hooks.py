"""
mi_analysis/src/hooks.py — an explicit, instrumented forward pass of the model_training_evaluation model (MB02).

`forward` recomputes EventTransformer.forward (src/model.py) step by step with the model's own
modules and parameters. At every named SITE it can record the tensor (capture) and/or replace it
(replace = {site: function(tensor) -> tensor}). MB02 proves that, without replacements, it gives
exactly the logits of model(inputs).

Sites (B = batch, T = 16 positions, D = 128, H = 2 heads, E = 64, F = 512):
  factor_<group>        [B, T, D]  player, team, role, competition, space, time, state, history, position
  resid_pre_0           [B, T, D]  the embedding (sum of factor vectors)
  resid_pre_l (l > 0)   [B, T, D]  residual stream entering block l
  attn_in_q_l.h / attn_in_k_l.h / attn_in_v_l.h   [B, T, D]  residual read by head h's query / key / value
  attn_scores_l         [B, H, T, T]  pre-softmax scores (future entries are -inf)
  attn_pattern_l        [B, H, T, T]  attention probabilities
  head_z_l              [B, T, H, E]  head outputs before W_O
  attn_out_l            [B, T, D]  attention output after W_O (added to the residual)
  resid_mid_l           [B, T, D]  residual after attention
  mlp_in_l              [B, T, D]  residual read by the MLP (before its LayerNorm)
  mlp_hidden_l          [B, T, F]  MLP hidden units after GELU
  mlp_out_l             [B, T, D]  MLP output (added to the residual)
  resid_post_l          [B, T, D]  residual leaving block l
  final_resid           [B, T, D]  residual entering the final LayerNorm
  final_normalised      [B, T, D]  output of the final LayerNorm
"""
import math

import torch
import torch.nn.functional as F

from src.model import FACTOR_GROUPS, CONTINUOUS_GROUPS


def forward(model, inputs, replace=None, capture=None, final_scale=None):
    """Return (logits [B, T, 15], cache dict).

    replace      dict site -> function(tensor) -> tensor, applied where the site is computed
    capture      None, "all", or a collection of site names to record
    final_scale  optional [B, T, 1] tensor: use this 1/sqrt(var + eps) in the final LayerNorm
                 instead of recomputing it (freezes the LayerNorm scale for decomposition)
    """
    if replace is None:
        replace = {}
    cache = {}

    def site(name, value):
        if name in replace:
            value = replace[name](value)
        if capture is not None and (capture == "all" or name in capture):
            cache[name] = value.detach()
        return value

    embedding = model.embedding
    factor_vectors = {}
    for group in embedding.groups:
        group_vector = None
        for feature in FACTOR_GROUPS[group]:
            vector = embedding.embeddings[feature](inputs[feature])
            group_vector = vector if group_vector is None else group_vector + vector
        if group in embedding.continuous_projections:
            values = torch.stack([inputs[name] for name in CONTINUOUS_GROUPS[group]], dim=-1)
            group_vector = group_vector + embedding.continuous_projections[group](values)
        factor_vectors[group] = site("factor_" + group, group_vector)
    any_input = inputs["zone"]
    batch_size, n_positions = any_input.shape
    if embedding.config.use_position:
        positions = torch.arange(n_positions, device=any_input.device)
        position_vectors = embedding.position_embedding(positions).unsqueeze(0).expand(batch_size, n_positions, -1)
        factor_vectors["position"] = site("factor_position", position_vectors)
    residual = None
    for name in factor_vectors:
        residual = factor_vectors[name] if residual is None else residual + factor_vectors[name]
    residual = site("resid_pre_0", residual)

    for layer, block in enumerate(model.blocks):
        if layer > 0:
            residual = site(f"resid_pre_{layer}", residual)
        attention = block.attention
        n_heads = attention.n_heads
        d_head = attention.d_head

        shared_input = block.ln_attention(residual)
        queries = attention.split_heads(attention.W_Q(shared_input))
        keys = attention.split_heads(attention.W_K(shared_input))
        values = attention.split_heads(attention.W_V(shared_input))
        for head in range(n_heads):
            for kind, projection, tensor in [("q", attention.W_Q, queries), ("k", attention.W_K, keys),
                                             ("v", attention.W_V, values)]:
                name = f"attn_in_{kind}_{layer}.{head}"
                if name in replace or (capture is not None and capture != "all" and name in capture):
                    receiver_input = site(name, residual)
                    if name in replace:
                        own = attention.split_heads(projection(block.ln_attention(receiver_input)))
                        tensor[:, head] = own[:, head]

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_head)
        mask = attention.causal_mask[:n_positions, :n_positions]
        scores = scores.masked_fill(~mask, float("-inf"))
        scores = site(f"attn_scores_{layer}", scores)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = site(f"attn_pattern_{layer}", probabilities)
        head_outputs = torch.matmul(probabilities, values)
        head_outputs = head_outputs.transpose(1, 2)
        head_outputs = site(f"head_z_{layer}", head_outputs)
        concatenated = head_outputs.reshape(batch_size, n_positions, n_heads * d_head)
        attention_output = site(f"attn_out_{layer}", attention.W_O(concatenated))
        residual = site(f"resid_mid_{layer}", residual + attention_output)

        mlp_input = site(f"mlp_in_{layer}", residual)
        hidden = F.gelu(block.mlp.input_layer(block.ln_mlp(mlp_input)))
        hidden = site(f"mlp_hidden_{layer}", hidden)
        mlp_output = site(f"mlp_out_{layer}", block.mlp.output_layer(hidden))
        residual = site(f"resid_post_{layer}", residual + mlp_output)

    final = site("final_resid", residual)
    layer_norm = model.ln_final
    if final_scale is None:
        normalised = layer_norm(final)
    else:
        centred = final - final.mean(dim=-1, keepdim=True)
        normalised = centred * final_scale * layer_norm.weight + layer_norm.bias
    normalised = site("final_normalised", normalised)
    logits = model.unembed(normalised)
    return logits, cache


def final_layernorm_scale(model, final_resid):
    """1 / sqrt(var + eps) of the final LayerNorm, per position ([B, T, 1])."""
    variance = final_resid.var(dim=-1, unbiased=False, keepdim=True)
    return 1.0 / torch.sqrt(variance + model.ln_final.eps)


def rows_and_positions(positions):
    rows = torch.arange(len(positions), device=positions.device)
    return rows, positions


def copy_at(source, positions, head=None):
    """Edit function: copy `source` into the tensor at one position per batch row.

    source     tensor with the same shape as the site ([B, T, ...] or head_z [B, T, H, E])
    positions  LongTensor [B]; rows with position < 0 or >= T are left unchanged
    head       for head_z sites: only this head's slice is copied
    """
    def edit(value):
        new = value.clone()
        valid = (positions >= 0) & (positions < value.shape[1])
        rows = torch.nonzero(valid).flatten()
        cols = positions[valid]
        if head is None:
            new[rows, cols] = source[rows, cols]
        else:
            new[rows, cols, head] = source[rows, cols, head]
        return new
    return edit


def add_at(delta, positions=None):
    """Edit function: add `delta` ([B, T, D]) everywhere, or only at one position per row."""
    def edit(value):
        if positions is None:
            return value + delta
        new = value.clone()
        rows = torch.arange(value.shape[0], device=value.device)
        new[rows, positions] = new[rows, positions] + delta[rows, positions]
        return new
    return edit


def set_mean(mean, positions=None, head=None):
    """Edit function for ablation: replace by a reference mean ([T, ...] per position).

    positions None = every position; otherwise one position per row.
    head      for head_z sites: only this head's slice.
    """
    def edit(value):
        new = value.clone()
        if positions is None:
            if head is None:
                new[:] = mean.unsqueeze(0)
            else:
                new[:, :, head] = mean[:, head].unsqueeze(0)
        else:
            rows = torch.arange(value.shape[0], device=value.device)
            if head is None:
                new[rows, positions] = mean[positions]
            else:
                new[rows, positions, head] = mean[positions, head]
        return new
    return edit


def set_values(source, positions=None, head=None):
    """Edit function for resample ablation: take values from another run (same shape)."""
    def edit(value):
        new = value.clone()
        if positions is None:
            if head is None:
                new[:] = source
            else:
                new[:, :, head] = source[:, :, head]
        else:
            rows = torch.arange(value.shape[0], device=value.device)
            if head is None:
                new[rows, positions] = source[rows, positions]
            else:
                new[rows, positions, head] = source[rows, positions, head]
        return new
    return edit


def combine(*functions):
    """Apply several edit functions to one site, in order."""
    def edit(value):
        for function in functions:
            value = function(value)
        return value
    return edit
