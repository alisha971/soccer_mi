"""
tests/test_model.py — mathematical correctness tests for src/model.py (notebook 03).

Every test takes a `context` dict with:
  make_model(seed, **overrides) -> EventTransformer on the right device
  make_batch(n_windows, first_window) -> (inputs, targets)
and returns (passed: bool, detail: str).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

import src.model as model_module


def test_shapes(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](8, 0)
    logits, internals = model(inputs, return_internals=True)
    batch_size, n_positions = targets.shape
    config = model.config
    expected = {
        "logits": (batch_size, n_positions, config.n_actions),
        "attention_probabilities": (batch_size, config.n_heads, n_positions, n_positions),
        "resid_post": (batch_size, n_positions, config.d_model),
    }
    observed = {
        "logits": tuple(logits.shape),
        "attention_probabilities": tuple(internals["blocks"][0]["attention_probabilities"].shape),
        "resid_post": tuple(internals["blocks"][-1]["resid_post"].shape),
    }
    passed = expected == observed and len(internals["blocks"]) == config.n_layers
    return passed, f"observed {observed}"


def test_causal_mask(context):
    model = context["make_model"](0)
    inputs, _ = context["make_batch"](8, 0)
    other_inputs, _ = context["make_batch"](8, 100)
    cut = 10
    changed = {}
    for name in inputs:
        tensor = inputs[name].clone()
        tensor[:, cut + 1:] = other_inputs[name][:, cut + 1:]      # replace every future event
        changed[name] = tensor
    with torch.no_grad():
        original = model(inputs)
        perturbed = model(changed)
    difference_before = (original[:, :cut + 1] - perturbed[:, :cut + 1]).abs().max().item()
    difference_after = (original[:, cut + 1:] - perturbed[:, cut + 1:]).abs().max().item()
    passed = difference_before == 0.0 and difference_after > 0.0
    return passed, f"max change at positions <= {cut}: {difference_before}; after: {difference_after:.3g}"


def test_attention_matches_reference(context):
    model = context["make_model"](0)
    attention = model.blocks[0].attention
    torch.manual_seed(1)
    x = torch.randn(4, 32, model.config.d_model, device=next(model.parameters()).device)
    with torch.no_grad():
        ours = attention(x)
        queries = attention.split_heads(attention.W_Q(x))
        keys = attention.split_heads(attention.W_K(x))
        values = attention.split_heads(attention.W_V(x))
        reference_heads = F.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        reference = attention.W_O(reference_heads.transpose(1, 2).reshape(4, 32, model.config.d_model))
    max_difference = (ours - reference).abs().max().item()
    return max_difference < 1e-5, f"max |ours - reference| = {max_difference:.2e}"


def test_attention_probabilities(context):
    model = context["make_model"](0)
    inputs, _ = context["make_batch"](8, 0)
    with torch.no_grad():
        _, internals = model(inputs, return_internals=True)
    worst_row_sum = 0.0
    worst_future = 0.0
    for block in internals["blocks"]:
        probabilities = block["attention_probabilities"]
        worst_row_sum = max(worst_row_sum, (probabilities.sum(-1) - 1).abs().max().item())
        future = torch.triu(torch.ones(32, 32, dtype=torch.bool, device=probabilities.device), diagonal=1)
        worst_future = max(worst_future, probabilities[..., future].abs().max().item())
    passed = worst_row_sum < 1e-5 and worst_future == 0.0
    return passed, f"max |row sum - 1| = {worst_row_sum:.2e}; max future weight = {worst_future}"


def test_residual_additions(context):
    model = context["make_model"](0)
    inputs, _ = context["make_batch"](8, 0)
    with torch.no_grad():
        _, internals = model(inputs, return_internals=True)
    errors = []
    factor_sum = sum(internals["factor_vectors"].values())
    errors.append((factor_sum - internals["blocks"][0]["resid_pre"]).abs().max().item())
    previous_post = None
    for block in internals["blocks"]:
        errors.append((block["resid_pre"] + block["attention_output"] - block["resid_mid"]).abs().max().item())
        errors.append((block["resid_mid"] + block["mlp_output"] - block["resid_post"]).abs().max().item())
        if previous_post is not None:
            errors.append((previous_post - block["resid_pre"]).abs().max().item())
        previous_post = block["resid_post"]
    worst = max(errors)
    return worst < 1e-5, f"max reconstruction error = {worst:.2e}"


def test_layernorm_placement(context):
    model = context["make_model"](0)
    inputs, _ = context["make_batch"](8, 0)
    with torch.no_grad():
        _, internals = model(inputs, return_internals=True)
    errors = []
    for layer_number, block in enumerate(internals["blocks"]):
        module_block = model.blocks[layer_number]
        errors.append((module_block.ln_attention(block["resid_pre"]) - block["attention_input"]).abs().max().item())
        errors.append((module_block.ln_mlp(block["resid_mid"]) - block["mlp_input"]).abs().max().item())
    # explicit LayerNorm formula: (x - mean) / sqrt(variance + eps) * scale + shift
    residual = internals["blocks"][0]["resid_pre"]
    layer_norm = model.blocks[0].ln_attention
    mean = residual.mean(dim=-1, keepdim=True)
    variance = residual.var(dim=-1, unbiased=False, keepdim=True)
    manual = (residual - mean) / torch.sqrt(variance + layer_norm.eps) * layer_norm.weight + layer_norm.bias
    formula_error = (manual - internals["blocks"][0]["attention_input"]).abs().max().item()
    final_error = (model.ln_final(internals["final_residual"]) - internals["final_normalised"]).abs().max().item()
    passed = max(errors) < 1e-6 and final_error < 1e-6 and formula_error < 1e-5
    return passed, (f"sub-layer inputs equal LN(residual): error {max(errors):.1e}; "
                    f"explicit formula (x-mean)/sqrt(var+eps)*scale+shift: error {formula_error:.1e}")


def test_positional_embeddings(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](8, 0)
    with torch.no_grad():
        original = model(inputs)
        saved = model.embedding.position_embedding.weight.clone()
        model.embedding.position_embedding.weight.zero_()
        without_positions = model(inputs)
        model.embedding.position_embedding.weight.copy_(saved)
    output_change = (original - without_positions).abs().max().item()
    loss = model_module.compute_loss(model(inputs), targets)
    loss.backward()
    gradient_norm = model.embedding.position_embedding.weight.grad.norm().item()
    passed = output_change > 0 and gradient_norm > 0
    return passed, f"output change when positions removed {output_change:.3g}; position gradient norm {gradient_norm:.3g}"


def test_output_dimension(context):
    model = context["make_model"](0)
    inputs, _ = context["make_batch"](2, 0)
    with torch.no_grad():
        logits = model(inputs)
    return logits.shape[-1] == 15, f"classes = {logits.shape[-1]} (15 actions; END is not a target)"


def test_determinism(context):
    model_a = context["make_model"](123)
    model_b = context["make_model"](123)
    parameters_identical = all(torch.equal(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))
    inputs, targets = context["make_batch"](8, 0)
    with torch.no_grad():
        outputs_identical = torch.equal(model_a(inputs), model_a(inputs))

    def short_run(seed):
        model = context["make_model"](seed)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for step in range(10):
            step_inputs, step_targets = context["make_batch"](8, step * 8)
            loss = model_module.compute_loss(model(step_inputs), step_targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        return losses

    run_a = short_run(7)
    run_b = short_run(7)
    passed = parameters_identical and outputs_identical and run_a == run_b
    return passed, f"params identical {parameters_identical}; outputs identical {outputs_identical}; 10-step losses identical {run_a == run_b}"


def test_gradient_flow(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](16, 0)
    loss = model_module.compute_loss(model(inputs), targets)
    loss.backward()
    problems = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            problems.append(f"{name}: no gradient")
            continue
        if not torch.isfinite(parameter.grad).all():
            problems.append(f"{name}: non-finite gradient")
            continue
        gradient = parameter.grad
        if name.endswith("weight") and "embeddings" in name:
            feature = name.split(".")[2]
            if feature in model_module.IDENTITY_FEATURES:
                if gradient[0].abs().max().item() != 0.0:
                    problems.append(f"{name}: index 0 received gradient")
                continue
        if gradient.abs().max().item() == 0.0 and "embeddings" not in name:
            problems.append(f"{name}: zero gradient")
    return len(problems) == 0, "all parameters receive finite gradient; index-0 rows receive none" if not problems else "; ".join(problems[:5])


def test_loss(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](8, 0)
    with torch.no_grad():
        logits = model(inputs)
        ours = model_module.compute_loss(logits, targets).item()
        log_probabilities = torch.log_softmax(logits, dim=-1)
        manual = -log_probabilities.gather(-1, targets.unsqueeze(-1)).mean().item()
        entropy = -(log_probabilities.exp() * log_probabilities).sum(-1).mean().item()
    # At initialisation the model must be uninformative: its predictive entropy should be
    # (almost) the maximum possible, ln 15. The loss itself can differ from ln 15, because
    # small random class preferences shared by all events may disfavour frequent classes.
    passed = abs(ours - manual) < 1e-5 and entropy > math.log(15) - 0.05
    return passed, (f"loss {ours:.5f} = manual {manual:.5f}; mean predictive entropy {entropy:.4f} "
                    f"(maximum ln 15 = {math.log(15):.4f})")


def test_valid_mask(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](8, 0)
    valid_mask = torch.ones_like(targets, dtype=torch.bool)
    valid_mask[:, :5] = False
    random_targets = targets.clone()
    generator = torch.Generator(device="cpu").manual_seed(0)
    random_targets[:, :5] = torch.randint(0, 15, (targets.shape[0], 5), generator=generator).to(targets.device)
    with torch.no_grad():
        logits = model(inputs)
        loss_original = model_module.compute_loss(logits, targets, valid_mask).item()
        loss_random = model_module.compute_loss(logits, random_targets, valid_mask).item()
    return loss_original == loss_random, f"loss with original targets {loss_original:.6f}; with randomised masked targets {loss_random:.6f}"


def test_memorise_one_batch(context):
    model = context["make_model"](0)
    inputs, targets = context["make_batch"](4, 0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    loss_value = None
    for step in range(400):
        loss = model_module.compute_loss(model(inputs), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_value = loss.item()
        if loss_value < 0.01:
            break
    return loss_value < 0.05, f"training loss on one batch of 4 windows after {step + 1} steps: {loss_value:.4f}"


def test_shuffled_labels(context):
    """Trained on permuted targets, the model must not beat the class prior on validation."""
    model = context["make_model"](0)
    train_batches = context["shuffled_label_batches"]
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for inputs, targets in train_batches:
        loss = model_module.compute_loss(model(inputs), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    inputs, targets = context["validation_batch"]
    with torch.no_grad():
        validation_nll = model_module.compute_loss(model(inputs), targets).item()
    prior_nll = context["prior_nll_on_validation_batch"]
    tolerance = context["shuffled_label_tolerance"]
    passed = validation_nll >= prior_nll - tolerance
    return passed, (f"validation NLL after training on shuffled labels {validation_nll:.4f}; "
                    f"class-prior NLL {prior_nll:.4f} (pass if not lower by more than {tolerance})")


def test_target_side_columns_cannot_change_inputs(context):
    events = context["events"]
    encoding = context["encoding"]
    encode = context["encode_features"]
    features_original, _ = encode(events, encoding)
    perturbed = events.copy()
    rng = np.random.default_rng(0)
    perturbed["action_id"] = rng.integers(0, 15, len(events))
    perturbed["end_zone"] = rng.integers(0, 96, len(events))
    perturbed["action"] = "SHOT"
    perturbed["type_name"] = "Shot"
    features_perturbed, _ = encode(perturbed, encoding)
    unchanged = all(np.array_equal(features_original[name], features_perturbed[name]) for name in features_original)
    return unchanged, "all 19 input arrays identical after randomising action_id, action, type_name, end_zone"


def test_index_zero_stays_zero(context):
    model = context["make_model"](0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    for step in range(5):
        inputs, targets = context["make_batch"](8, step * 8)
        loss = model_module.compute_loss(model(inputs), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    largest = 0.0
    for feature in model_module.IDENTITY_FEATURES:
        if feature in model.embedding.embeddings:
            largest = max(largest, model.embedding.embeddings[feature].weight[0].abs().max().item())
    return largest == 0.0, f"largest |value| in any index-0 identity row after 5 AdamW steps (weight decay 0.1): {largest}"


ALL_TESTS = [
    ("1 shapes", test_shapes),
    ("2 causal mask", test_causal_mask),
    ("3 attention vs reference", test_attention_matches_reference),
    ("4 attention probabilities", test_attention_probabilities),
    ("5 residual additions", test_residual_additions),
    ("6 LayerNorm placement", test_layernorm_placement),
    ("7 positional embeddings", test_positional_embeddings),
    ("8 output dimension", test_output_dimension),
    ("9 determinism", test_determinism),
    ("10 gradient flow", test_gradient_flow),
    ("11 loss", test_loss),
    ("12 valid-position mask", test_valid_mask),
    ("13 memorise one batch", test_memorise_one_batch),
    ("14 shuffled labels", test_shuffled_labels),
    ("15 target-side columns", test_target_side_columns_cannot_change_inputs),
    ("16 index 0 stays zero", test_index_zero_stays_zero),
]


def run_all(context):
    results = []
    for name, test in ALL_TESTS:
        try:
            passed, detail = test(context)
        except Exception as error:          # a crashing test is a failed test
            passed, detail = False, f"ERROR: {error!r}"
        results.append({"test": name, "passed": bool(passed), "detail": detail})
    return results
