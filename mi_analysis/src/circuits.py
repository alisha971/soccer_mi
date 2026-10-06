"""
mi_analysis/src/circuits.py — component ablation, path patching and faithfulness (written in MB07;
used by MB07, MB08, MB10, MB11, MB12).

A component is one attention head ("L2H1") or one MLP ("L3MLP"). Ablation replaces its output by
the reference mean at the same position (mean ablation) or by its value in another window
(resample ablation); zero is never used because the model never produces it.

REPAIR 2026-10-04 (see REPAIR_REPORT.md):
  * F01  receiver_gradients no longer detaches every receiver input. The old function is kept,
         unchanged, as receiver_gradients_detached_legacy so that MB10's historical screen can be
         reproduced; it must not be used for new analyses.
  * F08  random_circuit_space / enumerate_size_matched_circuits report and enumerate the distinct
         size-matched alternatives; random_circuits keeps its exact draws (same RNG calls) but warns
         when more draws are requested than distinct sets exist.
"""
import itertools
import math
import warnings

import numpy as np
import torch

import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics
import mi_src.mi_setup as mi_setup

REPAIR_VERSION = "2026-10-04"


def all_components(n_layers, n_heads):
    names = []
    for layer in range(n_layers):
        for head in range(n_heads):
            names.append(f"L{layer}H{head}")
        names.append(f"L{layer}MLP")
    return names


def parse(component):
    """'L2H1' -> ('head', 2, 1); 'L3MLP' -> ('mlp', 3, None); 'L1ATTN' -> ('attn', 1, None)."""
    if component.endswith("MLP"):
        return "mlp", int(component[1:-3]), None
    if component.endswith("ATTN"):
        return "attn", int(component[1:-4]), None
    layer_text, head_text = component[1:].split("H")
    return "head", int(layer_text), int(head_text)


def ablation_edits(components, means, positions=None, resample_cache=None):
    """replace-dict that ablates the given components.

    means           {site: [T, ...]} reference means (mean ablation), used when resample_cache is None
    resample_cache  {site: tensor [B, T, ...]} values from other windows (resample ablation)
    positions       None = every position, else a LongTensor [B] (one position per row)
    """
    per_site = {}
    for component in components:
        kind, layer, head = parse(component)
        site = f"head_z_{layer}" if kind == "head" else f"mlp_out_{layer}"
        if resample_cache is None:
            edit = hooks.set_mean(means[site], positions=positions, head=head)
        else:
            edit = hooks.set_values(resample_cache[site], positions=positions, head=head)
        per_site.setdefault(site, []).append(edit)
    replace = {}
    for site, edits in per_site.items():
        replace[site] = hooks.combine(*edits)
    return replace


def sender_sites(sender):
    """Cache sites needed to compute a sender's output."""
    if sender.startswith("factor_"):
        return [sender]
    if sender == "embedding":
        return ["resid_pre_0"]
    kind, layer, _ = parse(sender)
    return {"mlp": [f"mlp_out_{layer}"], "attn": [f"attn_out_{layer}"], "head": [f"head_z_{layer}"]}[kind]


def sender_difference(model, clean_cache, corrupt_cache, sender):
    """Residual-stream vector [B, T, D] by which the sender's clean output differs from its corrupted one."""
    if sender.startswith("factor_"):
        return clean_cache[sender] - corrupt_cache[sender]
    if sender == "embedding":
        return clean_cache["resid_pre_0"] - corrupt_cache["resid_pre_0"]
    kind, layer, head = parse(sender)
    if kind == "mlp":
        return clean_cache[f"mlp_out_{layer}"] - corrupt_cache[f"mlp_out_{layer}"]
    if kind == "attn":
        return clean_cache[f"attn_out_{layer}"] - corrupt_cache[f"attn_out_{layer}"]
    attention = model.blocks[layer].attention
    columns = slice(head * attention.d_head, (head + 1) * attention.d_head)
    dz = clean_cache[f"head_z_{layer}"][:, :, head] - corrupt_cache[f"head_z_{layer}"][:, :, head]
    return dz @ attention.W_O.weight[:, columns].T


def sender_layer(sender):
    if sender.startswith("factor_") or sender == "embedding":
        return -1
    return parse(sender)[1]


def only_at(difference, positions):
    """Keep a [B, T, D] difference at one position per row (zero elsewhere)."""
    kept = torch.zeros_like(difference)
    rows = torch.arange(difference.shape[0], device=difference.device)
    valid = (positions >= 0) & (positions < difference.shape[1])
    kept[rows[valid], positions[valid]] = difference[rows[valid], positions[valid]]
    return kept


def receivers_after(sender, n_layers, n_heads):
    """Every receiver that can read the sender: later heads' q/k/v, later (or same-layer, for a head)
    MLPs, and the logits."""
    layer = sender_layer(sender)
    kind = "factor" if layer < 0 else parse(sender)[0]
    receivers = []
    for later in range(n_layers):
        if later > layer:
            for head in range(n_heads):
                for side in ["q", "k", "v"]:
                    receivers.append(f"L{later}H{head}.{side}")
        if later > layer or (later == layer and kind in ("head", "attn")):
            receivers.append(f"L{later}MLP")
    receivers.append("logits")
    return receivers


def receiver_site(receiver):
    if receiver == "logits":
        return "final_resid"
    if receiver.endswith("MLP"):
        return f"mlp_in_{int(receiver[1:receiver.index('M')])}"
    component, side = receiver.split(".")
    _, layer, head = parse(component)
    return f"attn_in_{side}_{layer}.{head}"


@torch.no_grad()
def path_patch(model, corrupt_inputs, difference, receiver, positions, a_ctx, a_id):
    """Denoising path patch: the corrupted run, with the sender's clean-minus-corrupted difference
    added ONLY to the input of one receiver."""
    logits, _ = hooks.forward(model, corrupt_inputs, replace={receiver_site(receiver): hooks.add_at(difference)})
    return mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id)


@torch.no_grad()
def delta_under(model, inputs, positions, a_ctx, a_id, replace=None):
    logits, _ = hooks.forward(model, inputs, replace=replace)
    return mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id)


def random_circuit_space(components, circuit):
    """Number of DISTINCT component sets with the circuit's numbers of heads and MLPs (F08).

    Returns a dict: heads available / chosen, MLPs available / chosen, and the count
    C(n_heads_available, n_heads) * C(n_mlps_available, n_mlps). With 8 heads and 4 MLPs and a
    circuit of 7 heads + 4 MLPs this is C(8,7) * C(4,4) = 8.
    """
    heads = [c for c in components if not c.endswith("MLP")]
    mlps = [c for c in components if c.endswith("MLP")]
    n_heads = sum(1 for c in circuit if not c.endswith("MLP"))
    n_mlps = sum(1 for c in circuit if c.endswith("MLP"))
    return {"heads_available": len(heads), "heads_chosen": n_heads, "mlps_available": len(mlps),
            "mlps_chosen": n_mlps, "distinct_sets": math.comb(len(heads), n_heads) * math.comb(len(mlps), n_mlps)}


def enumerate_size_matched_circuits(components, circuit):
    """Every distinct component set with the circuit's numbers of heads and MLPs, each exactly once,
    in a fixed order (F08). Use this instead of sampling when the space is small."""
    heads = [c for c in components if not c.endswith("MLP")]
    mlps = [c for c in components if c.endswith("MLP")]
    n_heads = sum(1 for c in circuit if not c.endswith("MLP"))
    n_mlps = sum(1 for c in circuit if c.endswith("MLP"))
    out = []
    for chosen_heads in itertools.combinations(heads, n_heads):
        for chosen_mlps in itertools.combinations(mlps, n_mlps):
            out.append(sorted(str(c) for c in chosen_heads + chosen_mlps))
    return out


def random_circuits(components, circuit, n_circuits, seed, warn=True):
    """Random sets with the same number of heads and MLPs as the circuit.

    Draws are WITH replacement across draws (the same set can recur). The RNG calls are unchanged
    from the original, so historical draws are reproduced exactly. When n_circuits exceeds the
    number of distinct sets, a warning says so: "n random circuits" then does not mean n different
    architectures (F08). Use random_circuit_space() to report the count.
    """
    if warn:
        space = random_circuit_space(components, circuit)
        if n_circuits > space["distinct_sets"]:
            warnings.warn(f"random_circuits: {n_circuits} draws requested but only {space['distinct_sets']} distinct "
                          "size-matched sets exist; draws repeat (sampling with replacement).", stacklevel=2)
    rng = np.random.default_rng(seed)
    heads = [c for c in components if not c.endswith("MLP")]
    mlps = [c for c in components if c.endswith("MLP")]
    n_heads = sum(1 for c in circuit if not c.endswith("MLP"))
    n_mlps = sum(1 for c in circuit if c.endswith("MLP"))
    circuits = []
    for _ in range(n_circuits):
        chosen = list(rng.choice(heads, size=n_heads, replace=False)) + list(rng.choice(mlps, size=n_mlps, replace=False))
        circuits.append(sorted(str(c) for c in chosen))
    return circuits


def all_senders(n_layers, n_heads):
    """Every component whose output can differ between a clean and a corrupted run."""
    names = ["factor_player", "embedding"]
    for layer in range(n_layers):
        names.append(f"L{layer}ATTN")
        for head in range(n_heads):
            names.append(f"L{layer}H{head}")
        names.append(f"L{layer}MLP")
    return names


def all_receivers(n_layers, n_heads):
    """Every receiver that reads the residual stream: head q/k/v inputs, MLP inputs, the logits."""
    receivers = []
    for layer in range(n_layers):
        for head in range(n_heads):
            for side in ["q", "k", "v"]:
                receivers.append(f"L{layer}H{head}.{side}")
        receivers.append(f"L{layer}MLP")
    receivers.append("logits")
    return receivers


def receiver_gradients(model, inputs, positions, a_ctx, a_id, receivers):
    """d(Delta) / d(input of each receiver), from one backward pass (edge attribution patching).

    REPAIRED (F01). The gradient wanted for an edge score is the derivative of Delta with respect to
    the copy of the residual stream that ONE receiver reads, with everything downstream of that
    receiver computed normally. hooks.forward gives every receiver its own read of the residual
    (mlp_in_l feeds only MLP l; attn_in_{q,k,v}_l.h feeds only that head's projection; final_resid
    feeds only the final LayerNorm), so each read is replaced by `value.clone()` with retain_grad():
    a new node that is still connected to everything upstream. The forward values are unchanged.

    The only detached leaf is the embedding (resid_pre_0), which is upstream of every receiver; it
    exists because the inputs are integer indices and the parameters are frozen, so without it
    nothing in the graph would require a gradient.

    The previous version replaced every receiver read by value.detach().requires_grad_(True). That
    cut final_resid from the computation above it, so every internal receiver had no path to the
    output and its gradient was silently returned as zeros (MB10: all 4,012 internal-receiver edges
    scored exactly 0). It is kept as receiver_gradients_detached_legacy for provenance only.

    Raises if any receiver receives no gradient, instead of substituting zeros.
    """
    copies = {}
    replace = {}

    def root(value):
        return value.detach().requires_grad_(True)

    replace["resid_pre_0"] = root
    for receiver in receivers:
        site = receiver_site(receiver)
        if site in replace:
            continue

        def keep(value, site=site):
            copy = value.clone()
            copy.retain_grad()
            copies[site] = copy
            return copy

        replace[site] = keep
    with torch.enable_grad():
        logits, _ = hooks.forward(model, inputs, replace=replace)
        target = mi_metrics.at_target(logits, positions)
        rows = torch.arange(len(positions), device=target.device)
        (target[rows, a_ctx] - target[rows, a_id]).sum().backward()
    missing = [site for site, copy in copies.items() if copy.grad is None]
    if missing:
        raise RuntimeError(f"receiver_gradients: no gradient reached {missing}; the receiver is disconnected from Delta.")
    return {site: copy.grad.detach() for site, copy in copies.items()}


def receiver_gradients_detached_legacy(model, inputs, positions, a_ctx, a_id, receivers):
    """DEFECTIVE ORIGINAL (F01) — kept only to reproduce MB10's historical screen. Do not use.

    Every receiver read is replaced by an independent detached leaf, so only the final residual is
    connected to the output; every internal receiver gets a zero gradient.
    """
    leaves = {}
    replace = {}
    for receiver in receivers:
        site = receiver_site(receiver)
        if site in replace:
            continue

        def keep(value, site=site):
            leaf = value.detach().requires_grad_(True)
            leaves[site] = leaf
            return leaf

        replace[site] = keep
    with torch.enable_grad():
        logits, _ = hooks.forward(model, inputs, replace=replace)
        target = mi_metrics.at_target(logits, positions)
        rows = torch.arange(len(positions), device=target.device)
        (target[rows, a_ctx] - target[rows, a_id]).sum().backward()
    return {site: (leaf.grad.detach() if leaf.grad is not None else torch.zeros_like(leaf))
            for site, leaf in leaves.items()}


def copy_from(source, source_positions, target_positions, head=None):
    """Edit: value[row, target_positions[row]] = source[row, source_positions[row]]."""
    def edit(value):
        new = value.clone()
        rows = torch.arange(value.shape[0], device=value.device)
        if head is None:
            new[rows, target_positions] = source[rows, source_positions]
        else:
            new[rows, target_positions, head] = source[rows, source_positions, head]
        return new
    return edit


def scrub_edits(components, donor_cache, donor_positions, positions, at_target=True):
    """replace-dict overwriting each component's output with a donor event's output."""
    per_site = {}
    for component in components:
        kind, layer, head = parse(component)
        site = f"head_z_{layer}" if kind == "head" else f"mlp_out_{layer}"
        if at_target:
            edit = copy_from(donor_cache[site], donor_positions, positions, head=head)
        else:
            edit = hooks.set_values(donor_cache[site], positions=None, head=head)
        per_site.setdefault(site, []).append(edit)
    return {site: hooks.combine(*edits) for site, edits in per_site.items()}


def class_donors(class_values, rng):
    """For every row, another row of the same class (itself only when the class is a singleton)."""
    donors = np.arange(len(class_values))
    order = {}
    for position, value in enumerate(class_values):
        order.setdefault(value, []).append(position)
    for value, members in order.items():
        members = np.asarray(members)
        if len(members) == 1:
            continue
        shift = rng.integers(1, len(members), size=len(members))
        donors[members] = members[(np.arange(len(members)) + shift) % len(members)]
    return donors


@torch.no_grad()
def deltas(model, bundle, view, table, batch_size, replace_builder=None):
    """Delta at each row's target; replace_builder(inputs, positions) -> replace dict (or None)."""
    out = np.empty(len(table))
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        replace = replace_builder(inputs, positions) if replace_builder is not None else None
        out[part] = delta_under(model, inputs, positions, a_ctx, a_id, replace)
    return out


@torch.no_grad()
def identity_effects(model, bundle, view, table, batch_size, corrupt, replace_builder=None):
    """(Delta clean, Delta corrupted) with the same ablation applied to both runs.

    corrupt(inputs, positions, chunk) -> corrupted inputs (for example player -> 0 at the target).
    """
    clean = np.empty(len(table))
    corrupted = np.empty(len(table))
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        replace = replace_builder(inputs, positions) if replace_builder is not None else None
        other = corrupt(inputs, positions, table.iloc[part])
        clean[part] = delta_under(model, inputs, positions, a_ctx, a_id, replace)
        corrupted[part] = delta_under(model, other, positions, a_ctx, a_id, replace)
    return clean, corrupted


def scored_events(view, targets, match_ids, limit=None, seed=0):
    """Table of every scored event of an evaluation view (window, position, true action, match)."""
    import pandas as pd
    table = pd.DataFrame({"event_row": np.asarray(view["event_rows"]), "window": np.asarray(view["window_index"]),
                          "position": np.asarray(view["position"])})
    table["action"] = np.asarray(targets)[table["event_row"]]
    table["match_id"] = np.asarray(match_ids)[table["event_row"]]
    table["a_ctx"] = table["action"]
    table["a_id"] = table["action"]
    if limit is not None and len(table) > limit:
        table = table.sample(limit, random_state=seed).sort_values("event_row")
    return table.reset_index(drop=True)


@torch.no_grad()
def event_nll(model, bundle, view, table, batch_size, replace_builder=None):
    """Negative log-likelihood of the true action at each scored event (nats)."""
    out = np.empty(len(table))
    for part, inputs, positions, a_ctx, _ in mi_setup.event_batches(bundle, view, table, batch_size):
        replace = replace_builder(inputs, positions) if replace_builder is not None else None
        logits, _ = hooks.forward(model, inputs, replace=replace)
        log_probs = torch.log_softmax(mi_metrics.at_target(logits, positions), dim=-1)
        rows = torch.arange(len(positions), device=positions.device)
        out[part] = (-log_probs[rows, a_ctx]).cpu().numpy()
    return out


def component_sites(components):
    sites = set()
    for component in components:
        kind, layer, _ = parse(component)
        sites.add(f"head_z_{layer}" if kind == "head" else f"mlp_out_{layer}")
    return sorted(sites)


def mean_ablation(components, means, at_target):
    """replace_builder for mean ablation of components at the target position or at every position."""
    if not components:
        return None
    def builder(inputs, positions):
        return ablation_edits(components, means, positions if at_target else None)
    return builder


def resample_ablation(model, bundle, view, pool_windows, components, at_target, generator):
    """replace_builder for resample ablation: values from random other windows (same positions)."""
    if not components:
        return None
    sites = component_sites(components)
    def builder(inputs, positions):
        size = len(positions)
        chosen = pool_windows[torch.randint(0, len(pool_windows), (size,), generator=generator).numpy()]
        source_inputs, _ = mi_setup.window_inputs(bundle, view, chosen)
        _, cache = hooks.forward(model, source_inputs, capture=sites)
        return ablation_edits(components, None, positions if at_target else None, resample_cache=cache)
    return builder


class AblationPanel:
    """Holds the intact Delta of every table and the intact NLL, and measures changes under ablation."""

    def __init__(self, model, bundle, view, tables, nll_table, batch_size):
        self.model, self.bundle, self.view = model, bundle, view
        self.tables, self.nll_table, self.batch_size = tables, nll_table, batch_size
        self.intact = {name: deltas(model, bundle, view, table, batch_size) for name, table in tables.items()}
        self.intact_nll = event_nll(model, bundle, view, nll_table, batch_size)

    def changes(self, builder_factory, n_repeats=1):
        """Mean over repeats of (ablated - intact); builder_factory() -> replace_builder."""
        changes = {name: np.zeros(len(table)) for name, table in self.tables.items()}
        nll_change = np.zeros(len(self.nll_table))
        for _ in range(n_repeats):
            builder = builder_factory()
            for name, table in self.tables.items():
                if len(table):
                    ablated = deltas(self.model, self.bundle, self.view, table, self.batch_size, builder)
                    changes[name] += (ablated - self.intact[name]) / n_repeats
            ablated_nll = event_nll(self.model, self.bundle, self.view, self.nll_table, self.batch_size, builder)
            nll_change += (ablated_nll - self.intact_nll) / n_repeats
        return changes, nll_change

    def rows(self, unit, method, scope, changes, nll_change):
        import pandas as pd
        frames = []
        for name, table in self.tables.items():
            if len(table):
                frames.append(pd.DataFrame({"unit": unit, "method": method, "scope": scope, "population": name,
                                            "event_row": table["event_row"].to_numpy(), "match_id": table["match_id"].to_numpy(),
                                            "half": table["half"].to_numpy(), "change": changes[name]}))
        frames.append(pd.DataFrame({"unit": unit, "method": method, "scope": scope, "population": "NLL (scored events)",
                                    "event_row": self.nll_table["event_row"].to_numpy(),
                                    "match_id": self.nll_table["match_id"].to_numpy(), "half": "all", "change": nll_change}))
        return pd.concat(frames, ignore_index=True)


def summarise_ablation(frame, controls, n_resamples, q=0.05):
    """Per unit x method x scope (estimation half): mean change per population with CI, p-value on
    P-player, specificity |change conflict| - |change matched control| per pair, and BH q-values
    within each method x scope."""
    import pandas as pd
    out = []
    rows = frame[frame.half.isin(["estimation", "all"])]
    for (unit, method, scope), group in rows.groupby(["unit", "method", "scope"], sort=False):
        row = {"unit": unit, "method": method, "scope": scope}
        for population, part in group.groupby("population"):
            summary = mi_metrics.summarise(part["change"], part["match_id"], n_resamples)
            row[f"{population}: mean change"] = summary["mean"]
            row[f"{population}: CI low"] = summary["ci_low"]
            row[f"{population}: CI high"] = summary["ci_high"]
            if population == "P-player":
                row["P-player: p"] = mi_metrics.bootstrap_pvalue(part["change"], part["match_id"], n_resamples)
        conflict = group[group.population == "P-player"][["event_row", "match_id", "change"]]
        control = group[group.population == "matched controls"][["event_row", "change"]]
        if len(control) and len(conflict):
            paired = controls[["event_row", "partner_event_row"]].merge(control, on="event_row").merge(
                conflict, left_on="partner_event_row", right_on="event_row", suffixes=("_control", "_conflict"))
            if len(paired):
                specificity = np.abs(paired["change_conflict"]) - np.abs(paired["change_control"])
                summary = mi_metrics.summarise(specificity, paired["match_id"], n_resamples)
                row.update({"specificity: mean": summary["mean"], "specificity: CI low": summary["ci_low"],
                            "specificity: CI high": summary["ci_high"], "specificity: n pairs": summary["n"],
                            "specificity: p (greater)": mi_metrics.bootstrap_pvalue(
                                specificity, paired["match_id"], n_resamples, alternative="greater")})
        out.append(row)
    table = pd.DataFrame(out)
    for _, group in table.groupby(["method", "scope"]):
        for column, target in [("P-player: p", "P-player: q"), ("specificity: p (greater)", "specificity: q")]:
            if column in group:
                values = group[column].fillna(1.0).to_numpy()
                table.loc[group.index, target] = mi_metrics.benjamini_hochberg(values, q)[1]
    return table
