"""
mi_analysis/src/patching.py — activation patching over sites and positions (MB06).

Denoising: run the CORRUPTED input, copy one CLEAN activation in; does the clean behaviour return?
Noising:   run the CLEAN input, copy one CORRUPTED activation in; does the behaviour break?
The metric is always Delta = logit(a_ctx) - logit(a_id) of the clean event, read at its target.
"""
import torch

import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics


def sweep_sites(n_layers, n_heads):
    """(label, site, head) for the 21 patched sites: embedding, then per layer attn_out, each
    head, mlp_out, resid_post."""
    sites = [("embedding", "resid_pre_0", None)]
    for layer in range(n_layers):
        sites.append((f"L{layer} attn_out", f"attn_out_{layer}", None))
        for head in range(n_heads):
            sites.append((f"L{layer} head {head}", f"head_z_{layer}", head))
        sites.append((f"L{layer} mlp_out", f"mlp_out_{layer}", None))
        sites.append((f"L{layer} resid_post", f"resid_post_{layer}", None))
    return sites


@torch.no_grad()
def patch_batch(model, clean_inputs, corrupt_inputs, positions, a_ctx, a_id, sites, offsets, directions):
    """Return m_clean, m_corrupt and {(direction, label, offset): m_patched} for one batch."""
    site_names = sorted(set(site for _, site, _ in sites))
    clean_logits, clean_cache = hooks.forward(model, clean_inputs, capture=site_names)
    corrupt_logits, corrupt_cache = hooks.forward(model, corrupt_inputs, capture=site_names)
    m_clean = mi_metrics.delta(mi_metrics.at_target(clean_logits, positions), a_ctx, a_id)
    m_corrupt = mi_metrics.delta(mi_metrics.at_target(corrupt_logits, positions), a_ctx, a_id)
    patched = {}
    for direction in directions:
        if direction == "denoise":
            run_inputs, source_cache = corrupt_inputs, clean_cache
        else:
            run_inputs, source_cache = clean_inputs, corrupt_cache
        for label, site, head in sites:
            for offset in offsets:
                patch_positions = positions + offset
                edit = hooks.copy_at(source_cache[site], patch_positions, head=head)
                logits, _ = hooks.forward(model, run_inputs, replace={site: edit})
                patched[(direction, label, offset)] = mi_metrics.delta(
                    mi_metrics.at_target(logits, positions), a_ctx, a_id)
    return m_clean, m_corrupt, patched
