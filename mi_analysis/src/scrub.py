"""
mi_analysis/src/scrub.py — causal scrubbing of a paired (clean, counterfactual) effect.
REPAIR 2026-10-04 for review findings F02, F03 and F04 (see REPAIR_REPORT.md). Used by the corrected
sections of MB11b and MB16.

What is being tested
--------------------
The behaviour is a paired effect, e.g. the identity effect  E(x) = Delta(x) - Delta(phi(x)),  where
phi removes the player at the event's target position. A hypothesis H assigns each component c a set of
class keys k_c and claims  out_c(x) = g_c(k_c(x)): the component's output at the target depends on the
input only through those event fields. Components outside the hypothesis ("free", keys None) are
claimed to carry nothing about the behaviour.

Donor construction (the decision this module implements)
-------------------------------------------------------
For every recipient event x a donor d is drawn from the same class (k_c(d) = k_c(x), d != x when the
class has another member). Then, for the counterfactual run phi(x):

* if the counterfactual variable is one of c's class keys, H lets c respond to it, and the class of
  phi(x) is {e : k_c(e) = k_c(phi(x))}. phi(d) is a member of that class because phi acts on the key
  tuple the same way for d and x. The counterfactual run therefore receives c's output on phi(d):
  the SAME donor event with the SAME intervention applied at the donor's own target ("paired").
* if the counterfactual variable is not among c's keys (including free components), H asserts c is
  invariant to it, so out_c(phi(x)) = out_c(x) = out_c(d). The counterfactual run receives c's output
  on d, the same activation as the clean run ("same").

Both choices are exact samples from H's equivalence classes. Using one paired donor for both runs (rather
than independent donors per run) keeps donor-to-donor variation out of the per-event effect: the
effect is ~0.17 nats, while Delta varies by several nats across donors, so independent per-run donors
make the per-event effect mostly donor noise (see recovery_noise_check and finding F25 on MB16).

MB11b's original scrub wrote the clean donor activation into both runs for EVERY component, i.e. it
used "same" even where H says the component responds to the player. That blocks the effect by
construction (F02). It is reproducible here with branch_rule="all_same".

Self-check (F03): with every donor equal to the event itself and branch_rule="all_paired", both runs
must reproduce the intact model exactly, and so must their difference. All three are checked.

Wrong-class control (F04): the scoring table (a_ctx, a_id, positions) is never modified. One row
permutation is drawn per draw and applied to each component's COMPLETE class-label tuple, so the number
of classes, every class size and every joint tuple frequency are preserved exactly; only the pairing
between events and labels is broken.
"""
import numpy as np
import pandas as pd
import torch

import mi_src.circuits as circuits
import mi_src.hooks as hooks
import mi_src.mi_metrics as mi_metrics
import mi_src.mi_setup as mi_setup

REPAIR_VERSION = "2026-10-04"
BRANCH_RULES = ("hypothesis", "all_paired", "all_same")


def component_site(component):
    kind, layer, head = circuits.parse(component)
    return (f"head_z_{layer}" if kind == "head" else f"mlp_out_{layer}"), head


def class_labels(table, keys):
    """One hashable label per row: the tuple of the key columns, or 0 for a free component."""
    if not keys:
        return [0] * len(table)
    return list(map(tuple, table[list(keys)].to_numpy()))


def branch_modes(spec, counterfactual_feature, branch_rule="hypothesis"):
    """{component: 'paired' or 'same'} — which donor activation the counterfactual run receives."""
    if branch_rule not in BRANCH_RULES:
        raise ValueError(f"branch_rule must be one of {BRANCH_RULES}")
    modes = {}
    for component, keys in spec.items():
        if branch_rule == "all_paired":
            modes[component] = "paired"
        elif branch_rule == "all_same":
            modes[component] = "same"
        else:
            modes[component] = "paired" if (keys and counterfactual_feature in keys) else "same"
    return modes


def assign_donors(spec, table, rng):
    """{component: donor row index} drawn within each component's class (circuits.class_donors)."""
    return {component: circuits.class_donors(class_labels(table, keys), rng) for component, keys in spec.items()}


def permuted_label_assignment(spec, table, rng):
    """Wrong-class control (F04). Returns (assignment, diagnostics).

    One permutation of the rows is applied to every keyed component's complete label tuple; free
    components are drawn as usual. Diagnostics record that class sizes are preserved and how often the
    donor happens to share the recipient's TRUE class anyway.
    """
    permutation = rng.permutation(len(table))
    assignment, diagnostics = {}, {}
    for component, keys in spec.items():
        labels = class_labels(table, keys)
        if not keys:
            assignment[component] = circuits.class_donors(labels, rng)
            continue
        wrong = [labels[i] for i in permutation]
        assert sorted(pd.Series(labels).value_counts().tolist()) == sorted(pd.Series(wrong).value_counts().tolist())
        donors = circuits.class_donors(wrong, rng)
        assignment[component] = donors
        diagnostics[component] = {
            "classes": int(len(set(labels))),
            "class sizes preserved": True,
            "share of donors that share the recipient's true class": float(np.mean([labels[d] == labels[i] for i, d in enumerate(donors)])),
            "share of rows that are their own donor (singleton)": float(np.mean(donors == np.arange(len(donors)))),
        }
    return assignment, diagnostics


def table_fingerprint(table, columns=("event_row", "window", "position", "a_ctx", "a_id")):
    """Order-sensitive hash of the scoring columns; used to prove they were not modified."""
    present = [c for c in columns if c in table.columns]
    return int(pd.util.hash_pandas_object(table[present], index=False).sum())


@torch.no_grad()
def scrub_pair(model, bundle, view, table, assignment, modes, counterfactual, batch_size, at_target=True):
    """(Delta on the recorded events, Delta under the counterfactual), each with the scrub applied.

    assignment     {component: donor row index into `table`}
    modes          {component: 'paired' | 'same'} from branch_modes()
    counterfactual function(inputs, positions) -> counterfactual inputs (e.g. player -> 0 at the target)
    """
    device = bundle["device"]
    clean_out = np.empty(len(table))
    counter_out = np.empty(len(table))
    components = list(assignment)
    sites = sorted({component_site(c)[0] for c in components})
    need_paired = any(modes[c] == "paired" for c in components)
    for part, inputs, positions, a_ctx, a_id in mi_setup.event_batches(bundle, view, table, batch_size):
        replace_clean, replace_counter = {}, {}
        if components:
            donor_index = {c: np.asarray(assignment[c][part]) for c in components}
            needed = sorted({int(i) for values in donor_index.values() for i in np.unique(values)})
            lookup = {row: position for position, row in enumerate(needed)}
            donor_table = table.iloc[needed]
            donor_inputs, _ = mi_setup.window_inputs(bundle, view, donor_table["window"].to_numpy())
            donor_positions = torch.as_tensor(donor_table["position"].to_numpy(), device=device)
            _, donor_clean = hooks.forward(model, donor_inputs, capture=sites)
            donor_counter = None
            if need_paired:
                _, donor_counter = hooks.forward(model, counterfactual(donor_inputs, donor_positions), capture=sites)
            per_clean, per_counter = {}, {}
            for component in components:
                site, head = component_site(component)
                take = torch.as_tensor([lookup[int(i)] for i in donor_index[component]], device=device)
                for branch, cache, per_site in [("clean", donor_clean, per_clean),
                                                ("counter", donor_counter if modes[component] == "paired" else donor_clean, per_counter)]:
                    source = cache[site][take]
                    if at_target:
                        edit = circuits.copy_from(source, donor_positions[take], positions, head=head)
                    else:
                        edit = hooks.set_values(source, positions=None, head=head)
                    per_site.setdefault(site, []).append(edit)
            replace_clean = {site: hooks.combine(*edits) for site, edits in per_clean.items()}
            replace_counter = {site: hooks.combine(*edits) for site, edits in per_counter.items()}
        clean_out[part] = circuits.delta_under(model, inputs, positions, a_ctx, a_id, replace_clean)
        counter_out[part] = circuits.delta_under(model, counterfactual(inputs, positions), positions, a_ctx, a_id,
                                                 replace_counter)
    return clean_out, counter_out


def averaged_scrub(model, bundle, view, table, spec, counterfactual, counterfactual_feature, draws, seed, batch_size,
                   at_target=True, branch_rule="hypothesis", wrong_class=False, assignment_plan=None, modes_override=None):
    """Mean over draws of (Delta clean, Delta counterfactual). One rng (seeded once) feeds every draw,
    so draws do not repeat. Returns (clean, counter, info)."""
    before = table_fingerprint(table)
    rng = np.random.default_rng(seed)
    modes = branch_modes(spec, counterfactual_feature, branch_rule) if modes_override is None else dict(modes_override)
    clean_total = np.zeros(len(table))
    counter_total = np.zeros(len(table))
    diagnostics, assignments, draw_clean, draw_counter = [], [], [], []
    nonself = np.ones(len(table), dtype=bool)
    for draw in range(draws):
        if assignment_plan is not None:
            assignment = {c: np.asarray(v).copy() for c,v in assignment_plan[draw].items()}
        elif wrong_class:
            assignment, diag = permuted_label_assignment(spec, table, rng)
            diagnostics.append(diag)
        else:
            assignment = assign_donors(spec, table, rng)
        assignments.append({c: np.asarray(v).copy() for c,v in assignment.items()})
        for c, keys in spec.items():
            if keys:
                nonself &= np.asarray(assignment[c]) != np.arange(len(table))
        clean, counter = scrub_pair(model, bundle, view, table, assignment, modes, counterfactual, batch_size, at_target)
        draw_clean.append(clean.copy()); draw_counter.append(counter.copy())
        clean_total += clean / draws
        counter_total += counter / draws
    assert table_fingerprint(table) == before, "the scoring table was modified during the scrub"
    return clean_total, counter_total, {"modes": modes, "wrong_class_diagnostics": diagnostics, "assignments": assignments, "nonself_every_keyed_node_all_draws": nonself, "draw_clean": np.asarray(draw_clean), "draw_counter": np.asarray(draw_counter)}


def self_scrub_check(model, bundle, view, table, components, counterfactual, batch_size, at_target=True, tolerance=1e-4):
    """F03: three invariants with every donor equal to the event itself (branch_rule all_paired).

    Returns a dict of the largest absolute errors and a pass flag; also reports what the original
    MB11b construction (all_same) gives on the counterfactual run, which should FAIL invariant 2
    whenever the components respond to the counterfactual.
    """
    identity = {c: np.arange(len(table)) for c in components}
    intact_clean, intact_counter = circuits.identity_effects(
        model, bundle, view, table, batch_size, lambda inputs, positions, chunk: counterfactual(inputs, positions))
    paired_clean, paired_counter = scrub_pair(model, bundle, view, table, identity,
                                              {c: "paired" for c in components}, counterfactual, batch_size, at_target)
    same_clean, same_counter = scrub_pair(model, bundle, view, table, identity,
                                          {c: "same" for c in components}, counterfactual, batch_size, at_target)
    errors = {"clean run vs intact clean": float(np.abs(paired_clean - intact_clean).max()),
              "counterfactual run vs intact counterfactual": float(np.abs(paired_counter - intact_counter).max()),
              "effect vs intact effect": float(np.abs((paired_clean - paired_counter) - (intact_clean - intact_counter)).max())}
    return {"errors": errors, "passed": all(value < tolerance for value in errors.values()),
            "original design (same donor in both runs): counterfactual-run error": float(np.abs(same_counter - intact_counter).max()),
            "original design: effect error": float(np.abs((same_clean - same_counter) - (intact_clean - intact_counter)).max())}


def recovered_share(value, full, floor, match_ids, n_resamples):
    """Pooled sign-aligned (value - floor) / (full - floor), as in MB11b."""
    return mi_metrics.pooled_recovery(np.asarray(value) - np.asarray(floor), np.asarray(full) - np.asarray(floor),
                                      match_ids, n_resamples)


def recovery_noise_check(floor_a, floor_b, full, match_ids, n_resamples):
    """Recovered share of a SECOND independent floor draw, scored against the first. It must be close
    to 0. A value near 1 means the per-event floor is mostly noise, so sign alignment by
    (full - floor) selects on that noise and every scrub looks like ~100% recovery (finding F25)."""
    share = recovered_share(floor_b, full, floor_a, match_ids, n_resamples)
    noise = float(np.std(np.asarray(floor_a) - np.asarray(floor_b)))
    return {"recovered share of floor draw 2 vs floor draw 1": share["recovery"], "CI low": share["ci_low"],
            "CI high": share["ci_high"], "SD of floor draw difference": noise,
            "mean |full - floor|": float(np.mean(np.abs(np.asarray(full) - np.asarray(floor_a))))}
