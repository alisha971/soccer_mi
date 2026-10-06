"""
mi_analysis/tests/test_interventions.py — scope tests of the input interventions (MB02).
"""
import torch

import mi_src.hooks as hooks
import mi_src.interventions as interventions


def test_scope(context):
    inputs, positions = context["inputs"], context["positions"]
    removed = interventions.remove_feature(inputs, "player", positions)
    interventions.assert_scope(inputs, removed, ["player"], positions)
    partner = interventions.set_feature(inputs, "player", positions, torch.full_like(positions, 7))
    interventions.assert_scope(inputs, partner, ["player"], positions)
    caught = False
    wrong = interventions.set_feature(inputs, "team", positions, torch.full_like(positions, 3))
    try:
        interventions.assert_scope(inputs, wrong, ["player"], positions)
    except AssertionError:
        caught = True
    return caught, "player removal and swap change only the player at the target; a team change is caught"


def test_index0_swap_is_no_op(context):
    model, inputs, positions = context["model"], context["inputs"], context["positions"]
    rows = torch.arange(len(positions), device=positions.device)
    zero_rows = inputs["player"][rows, positions] == 0
    if zero_rows.sum() == 0:
        return True, "no index-0 target in this batch (not applicable)"
    subset = {name: tensor[zero_rows] for name, tensor in inputs.items()}
    sub_positions = positions[zero_rows]
    removed = interventions.remove_feature(subset, "player", sub_positions)
    with torch.no_grad():
        a, _ = hooks.forward(model, subset)
        b, _ = hooks.forward(model, removed)
    return torch.equal(a, b), f"{int(zero_rows.sum())} index-0 targets: removal changes nothing (exactly)"


def test_norm_matched_vector(context):
    model, inputs, positions = context["model"], context["inputs"], context["positions"]
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        _, cache = hooks.forward(model, inputs, capture=["factor_player"])
        edit = interventions.norm_matched_vector(cache["factor_player"], positions, generator)
        replaced = edit(cache["factor_player"])
    rows = torch.arange(len(positions), device=positions.device)
    original_norm = cache["factor_player"][rows, positions].norm(dim=-1)
    new_norm = replaced[rows, positions].norm(dim=-1)
    same_norm = torch.allclose(original_norm, new_norm, atol=1e-5)
    other = replaced.clone()
    other[rows, positions] = cache["factor_player"][rows, positions]
    untouched = torch.equal(other, cache["factor_player"])
    return same_norm and untouched, "random vector has the original norm at the target and nothing else changes"


ALL_TESTS = [
    ("7 intervention scope", test_scope),
    ("8 index-0 player removal is a no-op", test_index0_swap_is_no_op),
    ("7b norm-matched random vector", test_norm_matched_vector),
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
