# Reference predictors and conflict set

## What we did

Trained two reference predictors on `research_events.parquet` (658,905 × 40, splits 525,893 / 68,003 / 65,009 over 280 / 35 / 35 matches).
Ran on a T4, torch 2.11: `python train_refs.py`.
Identity models: summed 32-dim embeddings → 15-logit head. Context model: summed 64-dim embeddings → MLP (64→64→15).
All: unweighted cross-entropy, Adam 3e-3, batch 8192, 25 epochs, best val NLL kept.
Three variants: identity-full (player+team+role+comp), identity-player-only, context (11 categorical + 4 continuous).

Loss is unweighted by design. These are priors. They must reflect base rates, not balanced accuracy.

## What it shows

| model | val acc / top-3 / NLL | test acc / top-3 / NLL | T | ECE before → after |
|---|---|---|---|---|
| identity-full | 0.336 / 0.739 / 1.880 | 0.320 / 0.721 / 1.925 | 0.970 | 0.017 → 0.016 / 0.025 → 0.025 |
| identity-player-only | 0.323 / 0.736 / 1.956 | 0.301 / 0.717 / 2.000 | 0.975 | 0.014 → 0.009 / 0.007 → 0.013 |
| context | 0.613 / 0.943 / 0.931 | 0.613 / 0.938 / 0.936 | 1.011 | 0.012 → 0.011 / 0.011 → 0.015 |

Per-class recall (val). Identity-full: PASS_SHORT 0.83, PASS_LONG 0.28, GK_ACTION 0.21, CARRY 0.06, rest ~0.
Player-only: PASS_SHORT 0.95, PASS_LONG 0.09, rest 0. Team/role/comp add +1.3pp accuracy and −0.075 NLL.
Context: CARRY 0.90, DUEL 0.87, RECOVERY 0.80, INTERCEPTION 0.71, GK_ACTION 0.90, FOUL_WON 1.00, SHOT 0.53, PASS_SHORT 0.55, PASS_LONG 0.38, CROSS 0.38, FOUL_COMMITTED 0.57, CLEARANCE 0.62, DRIBBLE 0.01, LOSS 0.00, OTHER 0.02.
Loop history: linear 32-dim context scored 0.572 val. MLP head → 0.594. 25 epochs → 0.603. Width 64 → 0.613. Test matches val (0.613). No overfit gap.

Calibration: temperatures 0.97–1.01, NLL moves ≤ 0.001, ECE stays 0.01–0.02. Models were already calibrated. JSD comparison is valid.

Leakage: perturbing identity columns leaves Model 2 output byte-identical (True). Identity never enters its input constructor. Per-player error std 0.34. Positioning carries weak identity signal. Reported, not acted on.

Conflict set. Val argmax-disagree rate 0.647. τ = 0.3492 (75th percentile of disagree-JSD on val, true divergence in nats, 0–ln2).
22,472 events = 16.9% of val+test (10,990 val + 11,482 test). 650 rows touch untrained `comp_ix` 5. Flagged, kept.
Top pairs: (PASS_SHORT, CARRY) 4,270; (PASS_SHORT, DUEL) 3,904; (PASS_SHORT, INTERCEPTION) 2,542; (PASS_SHORT, FOUL_WON) 1,545; (PASS_LONG, GK_ACTION) 1,264; (PASS_SHORT, CLEARANCE) 1,257. Top pair holds 19% at τ. No single-pair dominance.
Coverage: 96/96 zones, 21/24 minute bins, 9/9 play patterns, 70/70 val+test matches.
Actual-action mix in conflict: CARRY 0.24, DUEL 0.20, INTERCEPTION 0.13, FOUL_WON 0.08, CLEARANCE 0.07. Rare classes surface here. PASS_SHORT is 0.03.
Players: 180 distinct. `player_ix` 0 (shared UNK, 68.8% base rate) holds 83.1% of conflict rows. Dedicated: 179 players, top-5 hold 12%. Not dominated.
Teams: 83 distinct, top-5 hold 27%. Not dominated.

## What it means

Identity alone predicts habit. PASS_SHORT and little else. Context carries the signal: 0.336 → 0.613 accuracy, test matching val.
The conflict set is broad across zones, phases, players, and teams. It is fit for transformer arbitration.
Join on `match_id, event_id, index, split`. No re-deriving. Read Δt off the transformer's logits later.

## Flags from §3

1. EDA `DATA_PATH` pointed at a housing CSV with `read_csv`. Left the EDA notebook untouched. New work lives in `train_refs.py` + `reference_predictors_and_conflict_set.ipynb`, loads `research_events.parquet` with `read_parquet`.
2. Output is 15-wide. `prev_action_id` embedding has 16 rows (sentinel 15, 706 period-opening rows). Verified: `action_id` 0–14, `prev_action_id` 0–15.
3. `comp_ix` 5 has zero train rows. Hits 2,127 val + 1,788 test rows, 650 in conflict. Flagged in `has_untrained_comp`. Kept per team decision.
4. Vocabularies reused as-is (351/214/25/21). Splits reused as-is. Nothing rebuilt.

## Decisions the team asked for

- Identity scope: tried both. Default conflict set uses identity-full. Player-only predictions ship alongside as `P_id_po`.
- Class imbalance: unweighted CE + per-class reporting. Weighting would distort the prior.
- Calibration: independent temperature scaling per model on val. Negligible shift. Kept for the JSD guarantee.
- τ: threshold-based (p75, val only). Fixed top-N kept as sensitivity readout in `tau_selection.json`.
- Backoff: shared UNK index + additive role/team. Sufficient. Explicit player↔role interpolation is future work.

## Open questions back to the team

- Should the primary `P_id` stay identity-full, or switch to player-only for a stricter habit prior?
- Keep flagged `comp_ix` 5 rows in the arbitration read, or drop them at join time?
- Is 17% conflict volume the right operating point, or should τ move to p80 (13%) / p85 (10%)?

## Handoff

```
outputs/action_vocab.json
outputs/model1_identity_full/      weights.pt, config.json, metrics_val/test.json, confusion.csv
outputs/model1_identity_playeronly/ weights.pt, config.json, metrics_val/test.json, confusion.csv
outputs/model2_context/            weights.pt, config.json, metrics_val/test.json, confusion.csv, leakage.json
outputs/predictions/               val_probs.parquet, test_probs.parquet (keys + P_id, P_id_po, P_ctx)
outputs/conflict_set/              conflict_set.parquet, tau_selection.json
train_refs.py                      exact training source, ran on T4
reference_predictors_and_conflict_set.ipynb   this notebook (validation + figures)
```

Nothing about the combined transformer was trained, modified, or assumed beyond the join keys.
