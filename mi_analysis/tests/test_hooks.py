"""
mi_analysis/tests/test_hooks.py — correctness tests of the instrumented forward pass (MB02).

Every test takes a `context` dict with: model, inputs (a batch of real validation windows),
positions (one target position per row). Each returns (passed, detail).
"""
import torch

import mi_src.hooks as hooks
import mi_src.attribution as attribution
import mi_src.mi_metrics as mi_metrics


def test_no_op(context):
    model, inputs = context["model"], context["inputs"]
    with torch.no_grad():
        reference = model(inputs)
        ours, _ = hooks.forward(model, inputs)
    return torch.equal(reference, ours), f"max |difference| = {(reference - ours).abs().max().item():.2e} (must be exactly 0)"


def test_self_patch(context):
    model, inputs = context["model"], context["inputs"]
    with torch.no_grad():
        reference, cache = hooks.forward(model, inputs, capture="all")
        worst = 0.0
        for site, value in cache.items():
            logits, _ = hooks.forward(model, inputs, replace={site: (lambda v, value=value: value.clone())})
            worst = max(worst, (logits - reference).abs().max().item())
    return worst < 1e-6, f"replacing each of {len(cache)} sites by its own value: max change {worst:.2e}"


def test_residual_arithmetic(context):
    model, inputs = context["model"], context["inputs"]
    with torch.no_grad():
        _, cache = hooks.forward(model, inputs, capture="all")
    errors = []
    factor_sum = sum(value for name, value in cache.items() if name.startswith("factor_"))
    errors.append((factor_sum - cache["resid_pre_0"]).abs().max().item())
    for layer in range(len(model.blocks)):
        pre = cache["resid_pre_0"] if layer == 0 else cache[f"resid_pre_{layer}"]
        errors.append((pre + cache[f"attn_out_{layer}"] + cache[f"mlp_out_{layer}"] - cache[f"resid_post_{layer}"]).abs().max().item())
    return max(errors) < 1e-5, f"max reconstruction error {max(errors):.2e}"


def test_head_split(context):
    model, inputs = context["model"], context["inputs"]
    with torch.no_grad():
        _, cache = hooks.forward(model, inputs, capture="all")
    worst = 0.0
    for layer, block in enumerate(model.blocks):
        attention = block.attention
        total = attention.W_O.bias.clone()
        for head in range(attention.n_heads):
            columns = slice(head * attention.d_head, (head + 1) * attention.d_head)
            total = total + cache[f"head_z_{layer}"][:, :, head] @ attention.W_O.weight[:, columns].T
        worst = max(worst, (total - cache[f"attn_out_{layer}"]).abs().max().item())
    return worst < 1e-5, f"sum of head contributions + bias vs attention output: max error {worst:.2e}"


def test_causal_locality(context):
    model, inputs = context["model"], context["inputs"]
    position = 9
    with torch.no_grad():
        reference, cache = hooks.forward(model, inputs, capture=["head_z_1"])
        noise = torch.randn_like(cache["head_z_1"])
        positions = torch.full((inputs["zone"].shape[0],), position, device=noise.device)
        patched, _ = hooks.forward(model, inputs, replace={"head_z_1": hooks.copy_at(noise, positions)})
    before = (patched[:, :position] - reference[:, :position]).abs().max().item()
    after = (patched[:, position:] - reference[:, position:]).abs().max().item()
    return before == 0.0 and after > 0.0, f"change at positions < {position}: {before}; at >= {position}: {after:.3g}"


def test_frozen_layernorm_decomposition(context):
    model, inputs, positions = context["model"], context["inputs"], context["positions"]
    a_ctx = torch.zeros_like(positions) + 3
    a_id = torch.zeros_like(positions)
    with torch.no_grad():
        logits, cache = hooks.forward(model, inputs, capture="all")
        contributions, constant = attribution.decompose(model, cache, positions, a_ctx, a_id)
    total = sum(contributions.values()) + constant
    actual = mi_metrics.delta(mi_metrics.at_target(logits, positions), a_ctx, a_id)
    error = float(abs(total - actual).max())
    return error < 1e-4, f"sum of {len(contributions)} component terms + constant vs Delta: max error {error:.2e}"


def test_device_agreement(context):
    model, inputs = context["model"], context["inputs"]
    if not torch.cuda.is_available():
        return True, "skipped: no GPU in this session (recorded as not applicable)"
    import copy
    with torch.no_grad():
        gpu = model.to("cuda")(({k: v.to("cuda") for k, v in inputs.items()})).cpu()
        cpu_model = copy.deepcopy(model).to("cpu")
        cpu = cpu_model({k: v.to("cpu") for k, v in inputs.items()})
    difference = (gpu - cpu).abs().max().item()
    return difference < 1e-4, f"CPU vs GPU logits: max |difference| {difference:.2e}"


ALL_TESTS = [
    ("1 no-op: instrumented forward equals model(inputs)", test_no_op),
    ("2 self-patch leaves logits unchanged", test_self_patch),
    ("3 residual arithmetic", test_residual_arithmetic),
    ("4 head split of the attention output", test_head_split),
    ("5 causal locality of a patch", test_causal_locality),
    ("6 frozen-LayerNorm decomposition sums to Delta", test_frozen_layernorm_decomposition),
    ("9 CPU and GPU agree", test_device_agreement),
]


def run_all(context):
    results = []
    for name, test in ALL_TESTS:
        try:
            passed, detail = test(context)
        except Exception as error:
            passed, detail = False, f"ERROR: {error!r}"
        results.append({"test": name, "passed": bool(passed), "detail": detail})
    return results
