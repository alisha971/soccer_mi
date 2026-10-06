"""
mi_analysis/tests/test_repairs.py — regression tests for the 2026-10-04 repairs (see REPAIR_REPORT.md).

Same convention as tests/test_hooks.py: every test takes a `context` dict and returns
(passed, detail). run_all(context) runs them all.

context keys: model, bundle, view, table (scored events with event_row, window, position, a_ctx, a_id,
match_id, player, team_change), batch_size, features ({name: numpy array per event row}).
The same file runs on the real SEL-4L model (MB07R) and on the synthetic stand-in
(tests/run_synthetic.py). Tests marked "documents the defect" pass when the ORIGINAL code still shows
the defect the repair addresses, so the report can rely on a demonstrated failure mode.
"""
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import torch

import mi_src.circuits as circuits
import mi_src.context_edits as context_edits
import mi_src.graph as graph
import mi_src.hooks as hooks
import mi_src.interventions as interventions
import mi_src.mi_metrics as mi_metrics
import mi_src.mi_setup as mi_setup
import mi_src.patching as patching
import mi_src.scrub as scrub
import mi_src.steering as steering


def _first_batch(context, n=None):
    table = context["table"] if n is None else context["table"].iloc[:n].reset_index(drop=True)
    part, inputs, positions, a_ctx, a_id = next(mi_setup.event_batches(context["bundle"], context["view"], table, len(table)))
    return table, inputs, positions, a_ctx, a_id


def _double(model, inputs):
    model64 = copy.deepcopy(model).double()
    inputs64 = {k: (v.double() if v.is_floating_point() else v) for k, v in inputs.items()}
    return model64, inputs64


def _remove_player(inputs, positions):
    return interventions.remove_feature(inputs, "player", positions)


# ---------------------------------------------------------------- gradients (F01)
def test_receiver_gradients_match_finite_differences(context):
    """Every receiver type, including internal ones, gets the derivative a finite difference measures."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model64, inputs64 = _double(context["model"], inputs)
    n_layers, n_heads = len(model64.blocks), model64.blocks[0].attention.n_heads
    receivers = circuits.all_receivers(n_layers, n_heads)
    gradients = circuits.receiver_gradients(model64, inputs64, positions, a_ctx, a_id, receivers)
    generator = torch.Generator().manual_seed(0)
    worst, checked, nonzero_internal = 0.0, 0, 0
    for site, gradient in gradients.items():
        direction = torch.randn(gradient.shape, generator=generator, dtype=torch.float64).to(gradient.device)

        def total(eps):
            with torch.no_grad():
                logits, _ = hooks.forward(model64, inputs64, replace={site: hooks.add_at(eps * direction)})
            return float(mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id).sum())
        eps = 1e-5
        numeric = (total(eps) - total(-eps)) / (2 * eps)
        analytic = float((gradient * direction).sum())
        worst = max(worst, abs(numeric - analytic) / max(abs(numeric), 1e-6))
        checked += 1
        if site != "final_resid" and gradient.abs().sum() > 0:
            nonzero_internal += 1
    passed = worst < 1e-4 and nonzero_internal == checked - 1
    return passed, f"{checked} receiver sites; largest relative error vs finite difference {worst:.2e}; internal sites with non-zero gradient {nonzero_internal}/{checked - 1}"


def test_legacy_gradients_zero_internal(context):
    """Documents the defect: the original function gives exactly zero at every internal receiver."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model = context["model"]
    receivers = circuits.all_receivers(len(model.blocks), model.blocks[0].attention.n_heads)
    gradients = circuits.receiver_gradients_detached_legacy(model, inputs, positions, a_ctx, a_id, receivers)
    internal = {s: float(g.abs().sum()) for s, g in gradients.items() if s != "final_resid"}
    return all(v == 0.0 for v in internal.values()) and float(gradients["final_resid"].abs().sum()) > 0, \
        f"legacy: {len(internal)} internal receiver sites all exactly zero, final_resid non-zero"


def test_eap_predicts_small_exact_path_patch(context):
    """First-order edge score = limit of an exact path patch with a scaled-down sender difference."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model64, inputs64 = _double(context["model"], inputs)
    corrupted = _remove_player(inputs64, positions)
    sites = ["mlp_out_0", "head_z_0", "resid_pre_0"]
    _, clean_cache = hooks.forward(model64, inputs64, capture=sites)
    _, corrupt_cache = hooks.forward(model64, corrupted, capture=sites)
    n_layers, n_heads = len(model64.blocks), model64.blocks[0].attention.n_heads
    gradients = circuits.receiver_gradients(model64, corrupted, positions, a_ctx, a_id, circuits.all_receivers(n_layers, n_heads))
    worst = 0.0
    edges = [("L0MLP", f"L{n_layers - 1}MLP"), ("L0H0", f"L{n_layers - 1}H0.q"), ("embedding", "L0MLP"), ("L0MLP", "logits")]
    eps = 1e-4
    for sender, receiver in edges:
        difference = circuits.only_at(circuits.sender_difference(model64, clean_cache, corrupt_cache, sender), positions)
        base = circuits.path_patch(model64, corrupted, 0 * difference, receiver, positions, a_ctx, a_id)
        moved = circuits.path_patch(model64, corrupted, eps * difference, receiver, positions, a_ctx, a_id)
        exact = (moved - base).sum() / eps
        first_order = float((difference * gradients[circuits.receiver_site(receiver)]).sum())
        worst = max(worst, abs(exact - first_order) / max(abs(exact), 1e-6))
    return worst < 1e-3, f"{len(edges)} edges incl. internal receivers: largest relative gap between EAP and a small exact path patch {worst:.2e}"


# ---------------------------------------------------------------- hooks / patching
def test_no_state_left_after_interventions(context):
    """hooks are functional (no registered module hooks): after replace-runs and a gradient screen,
    a plain run equals model(inputs), parameters stay frozen and hold no gradient."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model = context["model"]
    with torch.no_grad():
        reference = model(inputs)
    hooks.forward(model, inputs, replace={"mlp_out_0": lambda v: v * 0})
    circuits.receiver_gradients(model, inputs, positions, a_ctx, a_id, ["L0MLP", "logits"])
    with torch.no_grad():
        after_model = model(inputs)
        after_hooks, _ = hooks.forward(model, inputs)
    registered = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks) for m in model.modules())
    frozen = all(not p.requires_grad and p.grad is None for p in model.parameters())
    passed = torch.equal(reference, after_model) and torch.equal(reference, after_hooks) and registered == 0 and frozen
    return passed, f"plain runs unchanged; registered module hooks {registered}; parameters frozen with no grad: {frozen}"


def test_patching_endpoints(context):
    """Denoising the last residual at the target restores the clean Delta; patching nothing leaves the corrupted one."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model = context["model"]
    last = f"resid_post_{len(model.blocks) - 1}"
    corrupted = _remove_player(inputs, positions)
    m_clean, m_corrupt, patched = patching.patch_batch(model, inputs, corrupted, positions, a_ctx, a_id,
                                                       [("last", last, None)], [0], ["denoise", "noise"])
    restored = np.abs(patched[("denoise", "last", 0)] - m_clean).max()
    broken = np.abs(patched[("noise", "last", 0)] - m_corrupt).max()
    nothing = circuits.path_patch(model, corrupted, torch.zeros_like(hooks.forward(model, corrupted, capture=["final_resid"])[1]["final_resid"]),
                                  "logits", positions, a_ctx, a_id)
    return max(restored, broken, np.abs(nothing - m_corrupt).max()) < 1e-5, \
        f"denoise -> clean {restored:.1e}; noise -> corrupt {broken:.1e}; zero patch -> corrupt {np.abs(nothing - m_corrupt).max():.1e}"


# ---------------------------------------------------------------- steering (F05, F06)
def test_steering_beta_zero_is_noop(context):
    table = context["table"]
    model = context["model"]
    site = f"resid_post_{len(model.blocks) - 1}"
    direction = np.random.default_rng(0).standard_normal(model.ln_final.weight.shape[0])
    changes = steering.steer_changes(model, context["bundle"], context["view"], table, direction, [0.0], 3.0, site, context["batch_size"])
    return float(np.abs(changes[0.0]).max()) == 0.0, f"largest |change| at beta = 0: {float(np.abs(changes[0.0]).max())}"


def test_steering_batch_invariant(context):
    """The same event receives the same push whatever its batch (repaired); the legacy rule does not."""
    table = context["table"]
    model = context["model"]
    site = f"resid_post_{len(model.blocks) - 1}"
    direction = np.random.default_rng(0).standard_normal(model.ln_final.weight.shape[0])
    scale = steering.reference_scale(model, context["bundle"], context["view"], table, site, context["batch_size"])
    shuffled = table.sample(frac=1.0, random_state=1)
    order = np.argsort(shuffled.index.to_numpy())
    a = steering.steer_changes(model, context["bundle"], context["view"], table, direction, [1.0], scale, site, 3)[1.0]
    b = steering.steer_changes(model, context["bundle"], context["view"], shuffled.reset_index(drop=True), direction, [1.0], scale, site, 7)[1.0][order]
    legacy_a = steering.steer_changes_batch_mean_legacy(model, context["bundle"], context["view"], table, direction, [1.0], site, 3)[1.0]
    legacy_b = steering.steer_changes_batch_mean_legacy(model, context["bundle"], context["view"], shuffled.reset_index(drop=True), direction, [1.0], site, 7)[1.0][order]
    repaired_gap, legacy_gap = float(np.abs(a - b).max()), float(np.abs(legacy_a - legacy_b).max())
    return repaired_gap < 1e-5 and legacy_gap > 1e-4, \
        f"regrouped + reordered: repaired max per-event difference {repaired_gap:.1e}; legacy batch-mean rule {legacy_gap:.3f}"


def test_calibration_models_pass_origin_and_rule(context):
    """Every calibration model predicts 0 at beta = 0; a linear truth is recovered; a curved truth
    fails the line but not the piecewise model."""
    betas = np.array(steering.DEFAULT_BETAS)
    ok = True
    for name, model in steering.fit_models(betas, 2.0 * betas, 0.5).items():
        ok &= abs(float(model(0.0))) < 1e-12
    linear = steering.fit_models(betas, 2.0 * betas, 0.5)["line through 0 (linear regime)"]
    ok &= abs(float(linear(1.0)) - 2.0) < 1e-9
    curved = np.tanh(2 * betas) * 3
    slope0, rows, widest = steering.local_linearity(betas, curved)
    line = steering.fit_models(betas, curved, widest)["line through 0 (linear regime)"]
    evaluation = steering.evaluate(line, betas, curved, curved - 0.05, curved + 0.05)
    piecewise = steering.evaluate(steering.fit_models(betas, curved, widest)["piecewise-linear through measured points"], betas, curved, curved - 0.05, curved + 0.05)
    ok &= (not evaluation["PASS (pre-specified rule)"]) and piecewise["PASS (pre-specified rule)"] and widest < 1.0
    return bool(ok), f"origin respected; tanh curve: linear regime |beta| <= {widest}, line PASS={evaluation['PASS (pre-specified rule)']}, piecewise PASS={piecewise['PASS (pre-specified rule)']}"


# ---------------------------------------------------------------- randomisation / resampling (F15)
def test_resample_other_never_noop_and_batch_free(context):
    features, table = context["features"], context["table"]
    name = "possession_flag" if "possession_flag" in features else "team_change"
    pool = features[name][table["event_row"].to_numpy()]
    own = pool.copy()
    new, unchanged = context_edits.resample_other(pool, own, table["event_row"].to_numpy(), seed=5, repeat=0)
    half = len(table) // 2
    new_part, _ = context_edits.resample_other(pool, own[half:], table["event_row"].to_numpy()[half:], seed=5, repeat=0)
    zone_pool = features["zone"][table["event_row"].to_numpy()] if "zone" in features else pool
    zone_0, _ = context_edits.resample_other(zone_pool, zone_pool, table["event_row"].to_numpy(), seed=5, repeat=0)
    zone_1, _ = context_edits.resample_other(zone_pool, zone_pool, table["event_row"].to_numpy(), seed=5, repeat=1)
    feasible = np.array([np.any(pool != v) for v in own])
    passed = (np.all(new[feasible] != own[feasible]) and np.array_equal(unchanged, ~feasible)
              and np.array_equal(new[half:], new_part))
    return bool(passed), (f"{name}: realised no-op rate {unchanged.mean():.3f} (only rows with no other value); same draws for a subset; "
                          f"zone draws for repeat 1 differ from repeat 0 on {np.mean(zone_0 != zone_1):.2f} of rows")


def test_two_draw_noop_formula(context):
    """Documents the defect: the original 'draw, redraw once' rule leaves sum(p^3) of rows unchanged."""
    rng = np.random.default_rng(0)
    pool = rng.choice([0, 1], size=2000, p=[0.75, 0.25])
    own = pool.copy()
    simulated = []
    for repeat in range(20):
        drawn = rng.choice(pool, size=len(own))
        drawn = np.where(drawn == own, rng.choice(pool, size=len(own)), drawn)
        simulated.append(np.mean(drawn == own))
    expected = context_edits.two_draw_noop_rate(pool, own)
    return abs(np.mean(simulated) - expected) < 0.01, f"expected unchanged share {expected:.3f}, simulated {np.mean(simulated):.3f} (0.75^3 + 0.25^3 = 0.438)"


# ---------------------------------------------------------------- category edits (F16, F17)
def test_ordinal_and_grid_steps_respect_special_codes(context):
    codes = [0, 1, 2, 3, 4]
    own = np.array([0, 2, 4, 14, 15])
    up, moved_up = context_edits.ordinal_step(own, codes, +1)
    down, moved_down = context_edits.ordinal_step(own, codes, -1)
    ok = list(up) == [1, 3, 4, 14, 15] and list(moved_up) == [True, True, False, False, False]
    ok &= list(down) == [0, 1, 3, 14, 15] and list(moved_down) == [False, True, True, False, False]
    cell_of = lambda code: (code // 4, code % 4) if 0 <= code < 12 else None
    code_of = lambda r, c: r * 4 + c
    right, moved = context_edits.grid_step(np.array([0, 3, 5, 99]), 3, 4, "col", +1, code_of, cell_of)
    ok &= list(right) == [1, 3, 6, 99] and list(moved) == [True, False, True, False]
    return bool(ok), "special codes never edited or produced; boundaries skipped, not flipped; grid neighbours follow the declared layout"


def test_code_zero_is_not_removal_for_context(context):
    """Documents F17: identity index 0 is a zero vector, a context index 0 is a learned vector."""
    model = context["model"]
    identity_zero = model.embedding.embeddings["player"].weight[0].abs().max().item()
    context_name = "zone" if "zone" in model.embedding.embeddings else next(n for n in model.embedding.embeddings if n not in interventions.IDENTITY_FEATURES)
    context_zero = model.embedding.embeddings[context_name].weight[0].abs().max().item()
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    with torch.no_grad():
        _, cache = hooks.forward(model, inputs, capture=["factor_space"])
    rows = torch.arange(len(positions), device=positions.device)
    mean_vector = cache["factor_space"][rows, positions].mean(0)
    edit = context_edits.factor_mean_ablation(mean_vector, positions)
    with torch.no_grad():
        _, ablated = hooks.forward(model, inputs, replace={"factor_space": edit}, capture=["factor_space"])
    replaced = (ablated["factor_space"][rows, positions] - mean_vector).abs().max().item()
    return identity_zero == 0.0 and context_zero > 0.0 and replaced < 1e-6, \
        f"player code 0 max |w| {identity_zero}; {context_name} code 0 max |w| {context_zero:.3f}; factor mean-ablation sets the group vector exactly (error {replaced:.1e})"


def test_signed_nonadditivity(context):
    """Documents F18 and checks the repair: an additive response has zero signed non-additivity, while
    |joint| - sum|singles| is -2 for single changes +1 and -1."""
    single = [np.array([1.0]), np.array([-1.0])]
    joint = np.array([0.0])
    old = np.abs(joint) - np.sum(np.abs(single), axis=0)
    new = context_edits.signed_nonadditivity(joint, single)
    interaction = context_edits.pairwise_interaction(np.array([3.0]), np.array([1.0]), np.array([1.5]), np.array([0.5]))
    return float(old[0]) == -2.0 and float(new[0]) == 0.0 and float(interaction[0]) == 1.0, \
        f"additive +1/-1: old statistic {old[0]}, signed non-additivity {new[0]}; factorial contrast example {interaction[0]}"


# ---------------------------------------------------------------- scrubbing (F02, F03, F04)
def test_self_scrub_three_invariants(context):
    model = context["model"]
    components = circuits.all_components(len(model.blocks), model.blocks[0].attention.n_heads)
    table = context["table"].iloc[:min(len(context["table"]), 40)].reset_index(drop=True)
    result = scrub.self_scrub_check(model, context["bundle"], context["view"], table, components, _remove_player, context["batch_size"])
    passed = result["passed"] and result["original design (same donor in both runs): counterfactual-run error"] > 1e-4
    return passed, (f"paired self-scrub errors {result['errors']}; original same-donor design counterfactual-run error "
                    f"{result['original design (same donor in both runs): counterfactual-run error']:.3f} (must be > 0)")


def test_paired_scrub_preserves_effect_under_true_abstraction(context):
    """Duplicate every event; class key = duplicate id (+ player). The abstraction 'each component depends
    only on (event, player)' is then exactly true, so the paired scrub must keep the identity effect
    exactly. The original same-donor design does not; a false abstraction (no player key) loses it."""
    model = context["model"]
    base = context["table"].iloc[:min(len(context["table"]), 20)].reset_index(drop=True)
    table = pd.concat([base.assign(dup=np.arange(len(base))), base.assign(dup=np.arange(len(base)))], ignore_index=True)
    components = circuits.all_components(len(model.blocks), model.blocks[0].attention.n_heads)
    clean, counter = circuits.identity_effects(model, context["bundle"], context["view"], table, context["batch_size"],
                                               lambda inputs, positions, chunk: _remove_player(inputs, positions))
    effect = clean - counter
    args = (model, context["bundle"], context["view"], table)
    true_spec = {c: ["dup", "player"] for c in components}
    false_spec = {c: ["dup"] for c in components}
    t_clean, t_counter, _ = scrub.averaged_scrub(*args, true_spec, _remove_player, "player", 2, 0, context["batch_size"])
    o_clean, o_counter, _ = scrub.averaged_scrub(*args, true_spec, _remove_player, "player", 2, 0, context["batch_size"], branch_rule="all_same")
    f_clean, f_counter, _ = scrub.averaged_scrub(*args, false_spec, _remove_player, "player", 2, 0, context["batch_size"])
    true_error = float(np.abs((t_clean - t_counter) - effect).max())
    old_error = float(np.abs((o_clean - o_counter) - effect).max())
    false_error = float(np.abs((f_clean - f_counter) - effect).max())
    return true_error < 1e-4 and old_error > 1e-3 and false_error > 1e-3, \
        f"max |effect error|: paired, true abstraction {true_error:.1e}; original same-donor design {old_error:.3f}; false abstraction {false_error:.3f}"


def test_wrong_class_control_preserves_classes_and_scoring(context):
    model = context["model"]
    table = context["table"].iloc[:min(len(context["table"]), 30)].reset_index(drop=True)
    spec = {"L0MLP": ["team_change", "player"], "L0H0": ["team_change"], "L0H1": None}
    before = table.copy()
    rng = np.random.default_rng(0)
    assignment, diagnostics = scrub.permuted_label_assignment(spec, table, rng)
    unchanged = before.equals(table)
    ok = unchanged and all(d["class sizes preserved"] for d in diagnostics.values())
    labels = scrub.class_labels(table, spec["L0MLP"])
    rng2 = np.random.default_rng(1)
    permutation = rng2.permutation(len(table))
    joint_before = pd.Series(labels).value_counts().sort_index()
    joint_after = pd.Series([labels[i] for i in permutation]).value_counts().sort_index()
    ok &= joint_before.equals(joint_after)
    clean, counter, info = scrub.averaged_scrub(model, context["bundle"], context["view"], table, spec, _remove_player, "player", 1, 3,
                                                context["batch_size"], wrong_class=True)
    ok &= before.equals(table)
    return bool(ok), f"scoring table unchanged; joint class-tuple counts preserved; diagnostics {diagnostics}"


def test_recovery_noise_check(context):
    """Documents F25: with a per-event floor that is mostly noise, sign-aligned recovery of a second
    floor draw is ~1; with a low-noise floor it is ~0."""
    rng = np.random.default_rng(0)
    n = 4000
    match = np.arange(n) % 40
    full = rng.normal(-0.17, 0.3, n)
    noisy_a, noisy_b = rng.normal(0, 3.0, n), rng.normal(0, 3.0, n)
    quiet_a, quiet_b = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    noisy = scrub.recovery_noise_check(noisy_a, noisy_b, full, match, 200)["recovered share of floor draw 2 vs floor draw 1"]
    quiet = scrub.recovery_noise_check(quiet_a, quiet_b, full, match, 200)["recovered share of floor draw 2 vs floor draw 1"]
    return noisy > 0.6 and abs(quiet) < 0.1, f"floor-vs-floor recovered share: noisy floor {noisy:.2f}; low-noise floor {quiet:.3f}"


# ---------------------------------------------------------------- circuits (F08) and graphs (F09)
def test_random_circuit_space(context):
    heads = [f"L{l}H{h}" for l in range(4) for h in range(2)]
    mlps = [f"L{l}MLP" for l in range(4)]
    components = heads + mlps
    circuit = [h for h in heads if h != "L3H0"] + mlps
    space = circuits.random_circuit_space(components, circuit)
    enumerated = circuits.enumerate_size_matched_circuits(components, circuit)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        draws = circuits.random_circuits(components, circuit, 100, seed=0)
    distinct = {tuple(d) for d in draws}
    ok = space["distinct_sets"] == 8 and len({tuple(e) for e in enumerated}) == 8 and len(distinct) <= 8 and len(caught) == 1
    return ok, f"distinct sets {space['distinct_sets']}; enumerated {len(enumerated)}; 100 draws contain {len(distinct)} distinct sets; warning raised: {len(caught) == 1}"


def test_graph_endpoints(context):
    """All edges kept -> the clean Delta; no edge kept -> the corrupted Delta; the component set of all
    components as a graph -> the clean Delta."""
    table, inputs, positions, a_ctx, a_id = _first_batch(context, 8)
    model = context["model"]
    n_layers, n_heads = len(model.blocks), model.blocks[0].attention.n_heads
    corrupted = _remove_player(inputs, positions)
    m_clean = circuits.delta_under(model, inputs, positions, a_ctx, a_id)
    m_corrupt = circuits.delta_under(model, corrupted, positions, a_ctx, a_id)
    every = graph.graph_delta(model, inputs, corrupted, positions, a_ctx, a_id, graph.kept_atoms(graph.all_edges(n_layers, n_heads), n_layers, n_heads))
    none = graph.graph_delta(model, inputs, corrupted, positions, a_ctx, a_id, graph.kept_atoms([], n_layers, n_heads))
    induced = graph.graph_delta(model, inputs, corrupted, positions, a_ctx, a_id,
                                graph.kept_atoms(graph.component_induced_edges(circuits.all_components(n_layers, n_heads), n_layers, n_heads), n_layers, n_heads))
    errors = (float(np.abs(every - m_clean).max()), float(np.abs(none - m_corrupt).max()), float(np.abs(induced - m_clean).max()))
    return max(errors) < 1e-4, f"all edges vs clean {errors[0]:.1e}; no edges vs corrupted {errors[1]:.1e}; all-component graph vs clean {errors[2]:.1e}"


# ---------------------------------------------------------------- statistics
def test_statistics(context):
    rng = np.random.default_rng(0)
    ok = True
    details = []
    adjusted = mi_metrics.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2]), 0.05)[1]
    ok &= np.allclose(adjusted, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)
    details.append(f"BH {np.round(adjusted, 4).tolist()}")
    values = rng.normal(1.0, 1.0, 600)
    matches = np.arange(600) % 30
    summary = mi_metrics.summarise(values, matches, 1000)
    ok &= summary["ci_low"] < 1.0 < summary["ci_high"] and abs(summary["mean"] - values.mean()) < 1e-12
    ok &= abs(mi_metrics.pooled_recovery(values, values, matches, 100)["recovery"] - 1.0) < 1e-12
    p_ctx = np.array([0.99, 0.50]); p_id = np.array([0.01, 0.50])
    mean_log_odds = float(np.mean(np.log(p_ctx / p_id)))
    aggregate = float(np.log(p_ctx.mean() / p_id.mean()))
    ok &= abs(mean_log_odds - 2.297560) < 1e-5 and abs(aggregate - 1.072121) < 1e-5
    details.append(f"review example: mean log-odds {mean_log_odds:.6f} vs aggregate {aggregate:.6f}")
    n = 20000
    p = rng.uniform(0.05, 0.95, n)
    outcome = rng.uniform(size=n) < p
    calibration = mi_metrics.probability_calibration(p, outcome, np.arange(n) % 50, 500)
    ok &= calibration["CI low"] < 0 < calibration["CI high"]
    details.append(f"calibrated simulation: predicted - realised {calibration['predicted minus realised']:.4f} [{calibration['CI low']:.4f}, {calibration['CI high']:.4f}]")
    p_c = rng.uniform(0.5, 0.95, n)
    p_i = (1 - p_c) * rng.uniform(0.1, 0.9, n)
    u = rng.uniform(size=n)
    realised_c, realised_i = u < p_c, (u >= p_c) & (u < p_c + p_i)
    agg = mi_metrics.aggregate_log_odds(p_c, p_i, realised_c, realised_i, np.arange(n) % 50, 500)
    ok &= agg["CI low"] < 0 < agg["CI high"]
    details.append(f"calibrated simulation aggregate log-odds model-data {agg['model minus data']:.3f} [{agg['CI low']:.3f}, {agg['CI high']:.3f}] "
                   f"while mean Delta - data = {float(np.mean(np.log(p_c / p_i)) - agg['data log-odds']):.3f}")
    always_positive = np.abs(rng.normal(0, 0.01, 300)) + 1e-6
    p_value = mi_metrics.bootstrap_pvalue(always_positive, np.arange(300) % 30, 2000, alternative="greater")
    ok &= abs(p_value - 1 / 2001) < 1e-12
    details.append(f"F20: a tiny positive magnitude gets the floor p = {p_value:.6f}")
    return bool(ok), "; ".join(details)


ALL_TESTS = [test_receiver_gradients_match_finite_differences, test_legacy_gradients_zero_internal,
             test_eap_predicts_small_exact_path_patch, test_no_state_left_after_interventions, test_patching_endpoints,
             test_steering_beta_zero_is_noop, test_steering_batch_invariant, test_calibration_models_pass_origin_and_rule,
             test_resample_other_never_noop_and_batch_free, test_two_draw_noop_formula,
             test_ordinal_and_grid_steps_respect_special_codes, test_code_zero_is_not_removal_for_context,
             test_signed_nonadditivity, test_self_scrub_three_invariants,
             test_paired_scrub_preserves_effect_under_true_abstraction, test_wrong_class_control_preserves_classes_and_scoring,
             test_recovery_noise_check, test_random_circuit_space, test_graph_endpoints, test_statistics]


def run_all(context):
    results = []
    for test in ALL_TESTS:
        try:
            passed, detail = test(context)
        except Exception as error:
            passed, detail = False, f"raised {type(error).__name__}: {error}"
        results.append({"test": test.__name__, "passed": bool(passed), "detail": detail})
    return results
