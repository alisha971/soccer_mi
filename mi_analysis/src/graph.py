"""
mi_analysis/src/graph.py — faithfulness of an EDGE graph (not of a component set).
REPAIR 2026-10-04 for review finding F09 (see REPAIR_REPORT.md). Used by the corrected section of MB11.

Why a separate test
-------------------
MB11's F mean-ablates every component outside the circuit and keeps every component inside it, with all
of their connections. That validates a COMPONENT SET. MB10 also selected an EDGE graph (sender ->
receiver edges that each passed an exact single-edge path patch). A retained component can still
communicate with another retained component through an edge that was not selected, so component-set
faithfulness says nothing about whether the selected edges carry the effect jointly.

The graph run (denoising, recursive)
------------------------------------
Clean input x, corrupted input x' (the identity contrast: player -> 0 at the target). The two runs
differ only at the target position t (the model is causal), and the graph run is computed there.
Each receiver r (a head's q / k / v input, an MLP input, or the logits' final residual) reads

    in_r = R'_r + sum over kept senders s of r of (g_s - c_s)

where R'_r is what r reads in the corrupted run, c_s is sender s's output in the corrupted run, and g_s
is sender s's output IN THE GRAPH RUN ITSELF (so a sender only carries a clean signal if its own kept
inputs bring one). The embedding sender's g is the clean embedding. With every edge kept the graph run
equals the clean run exactly; with no edge kept it equals the corrupted run exactly (both are tests).

Senders are resolved to atoms: "embedding" and "factor_player" both mean the embedding difference (for
the identity contrast they are the same vector; this is asserted), "L{l}ATTN" means both heads of
layer l, a head is itself, an MLP is itself. A receiver keeps an atom if any kept edge names it.
Head contributions exclude the W_O bias, which is identical in both runs.

Only offset-0 edges of the identity contrast are handled. The context contrast changes every position
of the window, so its edge graph is NOT tested here (stated wherever this is used).
"""
import numpy as np
import torch

import mi_src.circuits as circuits
import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics

REPAIR_VERSION = "2026-10-04"


def sender_atoms(sender, n_heads):
    if sender in ("embedding", "factor_player"):
        return ["embedding"]
    kind, layer, head = circuits.parse(sender)
    if kind == "attn":
        return [f"L{layer}H{h}" for h in range(n_heads)]
    return [sender]


def atom_layer(atom):
    return -1 if atom == "embedding" else circuits.parse(atom)[1]


def receiver_layer(receiver, n_layers):
    if receiver == "logits":
        return n_layers
    return circuits.parse(receiver.split(".")[0])[1]


def can_feed(atom, receiver, n_layers):
    """True if the atom is computed before the receiver reads the residual stream."""
    a_layer, r_layer = atom_layer(atom), receiver_layer(receiver, n_layers)
    if receiver == "logits":
        return True
    if receiver.endswith("MLP"):
        if atom == "embedding":
            return True
        kind = circuits.parse(atom)[0]
        return a_layer < r_layer or (a_layer == r_layer and kind == "head")
    return a_layer < r_layer


def kept_atoms(edges, n_layers, n_heads):
    """{receiver: set(atoms)} from a list of (sender, receiver) edges; raises on an impossible edge."""
    kept = {r: set() for r in circuits.all_receivers(n_layers, n_heads)}
    for sender, receiver in edges:
        for atom in sender_atoms(sender, n_heads):
            if not can_feed(atom, receiver, n_layers):
                raise ValueError(f"edge {sender} -> {receiver} cannot exist (sender is not upstream)")
            kept[receiver].add(atom)
    return kept


def all_edges(n_layers, n_heads):
    """Every identity-contrast edge at offset 0, in MB10's naming."""
    out = []
    for sender in circuits.all_senders(n_layers, n_heads):
        for receiver in circuits.receivers_after(sender, n_layers, n_heads):
            out.append((sender, receiver))
    return out


def component_induced_edges(components, n_layers, n_heads):
    """Every edge whose sender is the embedding / player factor or a circuit component and whose
    receiver is a circuit component's input or the logits: the component set expressed as a graph."""
    members = set(components)
    out = []
    for sender, receiver in all_edges(n_layers, n_heads):
        sender_ok = sender in ("embedding", "factor_player") or sender in members or (
            sender.endswith("ATTN") and all(a in members for a in sender_atoms(sender, n_heads)))
        receiver_ok = receiver == "logits" or receiver.split(".")[0] in members
        if sender_ok and receiver_ok:
            out.append((sender, receiver))
    return out


def _corrupt_reference_site(receiver):
    if receiver == "logits":
        return "final_resid"
    if receiver.endswith("MLP"):
        return receiver_site_name(receiver)
    layer = circuits.parse(receiver.split(".")[0])[1]
    return "resid_pre_0" if layer == 0 else f"resid_pre_{layer}"


def receiver_site_name(receiver):
    return circuits.receiver_site(receiver)


def _head_contribution(model, layer, head, z_at_target):
    attention = model.blocks[layer].attention
    columns = slice(head * attention.d_head, (head + 1) * attention.d_head)
    return z_at_target[:, head] @ attention.W_O.weight[:, columns].T


@torch.no_grad()
def graph_delta(model, clean_inputs, corrupt_inputs, positions, a_ctx, a_id, kept):
    """Delta at the target in the recursive graph run (see module docstring)."""
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attention.n_heads
    rows = torch.arange(len(positions), device=positions.device)
    sites = (["resid_pre_0", "final_resid"] + [f"resid_pre_{l}" for l in range(1, n_layers)]
             + [f"mlp_in_{l}" for l in range(n_layers)] + [f"head_z_{l}" for l in range(n_layers)]
             + [f"mlp_out_{l}" for l in range(n_layers)])
    _, clean = hooks.forward(model, clean_inputs, capture=["resid_pre_0"])
    _, corrupt = hooks.forward(model, corrupt_inputs, capture=sites)
    embedding_difference = clean["resid_pre_0"] - corrupt["resid_pre_0"]
    off_target = embedding_difference.clone()
    off_target[rows, positions] = 0
    if off_target.abs().max().item() > 0:
        raise ValueError("graph_delta handles contrasts that change the embedding at the target only")
    c, g = {}, {}
    c["embedding"] = corrupt["resid_pre_0"][rows, positions]
    g["embedding"] = clean["resid_pre_0"][rows, positions]
    for layer in range(n_layers):
        z = corrupt[f"head_z_{layer}"][rows, positions]
        for head in range(n_heads):
            c[f"L{layer}H{head}"] = _head_contribution(model, layer, head, z)
        c[f"L{layer}MLP"] = corrupt[f"mlp_out_{layer}"][rows, positions]

    replace = {}
    for layer in range(n_layers):
        def record_heads(value, layer=layer):
            z = value[rows, positions]
            for head in range(n_heads):
                g[f"L{layer}H{head}"] = _head_contribution(model, layer, head, z)
            return value

        def record_mlp(value, layer=layer):
            g[f"L{layer}MLP"] = value[rows, positions]
            return value
        replace[f"head_z_{layer}"] = record_heads
        replace[f"mlp_out_{layer}"] = record_mlp
    for receiver, atoms in kept.items():
        reference = corrupt[_corrupt_reference_site(receiver)][rows, positions]

        def feed(value, atoms=tuple(sorted(atoms)), reference=reference):
            new = value.clone()
            total = reference.clone()
            for atom in atoms:
                total = total + (g[atom] - c[atom])
            new[rows, positions] = total
            return new
        replace[receiver_site_name(receiver)] = feed
    logits, _ = hooks.forward(model, corrupt_inputs, replace=replace)
    return mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id)


def canonical_edges(edges, n_layers, n_heads):
    """Expand aliases and deduplicate actual atom-to-receiver connections."""
    kept = kept_atoms(edges, n_layers, n_heads)
    return sorted((atom, receiver) for receiver, atoms in kept.items() for atom in atoms)


def random_edge_sets(universe, size, n_sets, seed):
    """n_sets random edge sets of `size` edges drawn without replacement from `universe`."""
    rng = np.random.default_rng(seed)
    universe = list(universe)
    return [[universe[i] for i in rng.choice(len(universe), size=size, replace=False)] for _ in range(n_sets)]
