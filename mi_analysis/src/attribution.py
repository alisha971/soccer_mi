"""
mi_analysis/src/attribution.py — direct logit attribution and direct vs indirect routes (MB09).

With the final LayerNorm scale held at its value in the actual run, the final residual stream is
an exact sum of component vectors, and Delta splits into one term per component:
    Delta = sum_i  u . (gamma * (c_i - mean(c_i))) * s  +  u . beta  +  (b[a_ctx] - b[a_id])
with u = W_U[a_ctx] - W_U[a_id], s = 1 / sqrt(var + eps). The split depends on freezing s and is
not unique: a component that forwards information gets the credit for writing it.
"""
import torch

import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics

FACTORS = ["player", "team", "role", "competition", "space", "time", "state", "history", "position"]


def component_vectors(model, cache, positions):
    """{component name: [B, D]} vectors written into the residual stream, read at the target."""
    rows = torch.arange(len(positions), device=positions.device)
    vectors = {}
    for factor in FACTORS:
        key = "factor_" + factor
        if key in cache:
            vectors[factor] = cache[key][rows, positions]
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        z = cache[f"head_z_{layer}"][rows, positions]
        weight = attention.W_O.weight
        for head in range(attention.n_heads):
            columns = slice(head * attention.d_head, (head + 1) * attention.d_head)
            vectors[f"L{layer} head {head}"] = z[:, head] @ weight[:, columns].T
        vectors[f"L{layer} attn bias"] = attention.W_O.bias.unsqueeze(0).expand(len(positions), -1)
        vectors[f"L{layer} mlp"] = cache[f"mlp_out_{layer}"][rows, positions]
    return vectors


def decompose(model, cache, positions, a_ctx, a_id):
    """Per-component contribution to Delta at the target, the constant, and the reconstruction."""
    rows = torch.arange(len(positions), device=positions.device)
    final = cache["final_resid"][rows, positions]
    scale = 1.0 / torch.sqrt(final.var(dim=-1, unbiased=False, keepdim=True) + model.ln_final.eps)
    gamma = model.ln_final.weight
    beta = model.ln_final.bias
    unembed = model.unembed.weight
    u = unembed[a_ctx] - unembed[a_id]
    contributions = {}
    for name, vector in component_vectors(model, cache, positions).items():
        centred = vector - vector.mean(dim=-1, keepdim=True)
        contributions[name] = ((u * (gamma * centred)).sum(dim=-1) * scale.squeeze(-1)).cpu().numpy()
    constant = ((u * beta).sum(dim=-1) + model.unembed.bias[a_ctx] - model.unembed.bias[a_id]).cpu().numpy()
    return contributions, constant


@torch.no_grad()
def direct_indirect(model, clean_inputs, corrupt_inputs, positions, a_ctx, a_id):
    """Change in Delta through the direct route (embedding -> logits) and through later layers."""
    n_layers = len(model.blocks)
    sites = [f"attn_out_{l}" for l in range(n_layers)] + [f"mlp_out_{l}" for l in range(n_layers)]
    clean_logits, clean_cache = hooks.forward(model, clean_inputs, capture=sites + ["resid_pre_0"])
    corrupt_logits, corrupt_cache = hooks.forward(model, corrupt_inputs, capture=sites + ["resid_pre_0", "final_resid"])
    m_clean = mi_metrics.delta(mi_metrics.at_target(clean_logits, positions), a_ctx, a_id)
    m_corrupt = mi_metrics.delta(mi_metrics.at_target(corrupt_logits, positions), a_ctx, a_id)

    frozen = {}
    for site in sites:
        clean_value = clean_cache[site]
        frozen[site] = (lambda value, clean_value=clean_value: clean_value)
    direct_logits, _ = hooks.forward(model, corrupt_inputs, replace=frozen)
    m_direct = mi_metrics.delta(mi_metrics.at_target(direct_logits, positions), a_ctx, a_id)

    difference = clean_cache["resid_pre_0"] - corrupt_cache["resid_pre_0"]
    indirect_logits, _ = hooks.forward(model, corrupt_inputs, replace={"final_resid": hooks.add_at(difference)})
    m_indirect = mi_metrics.delta(mi_metrics.at_target(indirect_logits, positions), a_ctx, a_id)
    return {"full": m_clean - m_corrupt, "direct": m_clean - m_direct, "indirect": m_clean - m_indirect}
