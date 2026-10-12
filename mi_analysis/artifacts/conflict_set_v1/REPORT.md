# Implementation report — reference predictors + conflict set

Stage 1+2 of the three-stage pipeline: reference predictors, then conflict set. The combined transformer trains in parallel elsewhere and is untouched here.

## 1. Objective

Estimate two independent priors over the same 15-class action vocabulary and keep only the events where they genuinely disagree:

- Identity prior `P_id(a | player)` — what this player (plus team/role/competition) tends to do.
- Context prior `P_ctx(a | context)` — what the situation calls for, identity stripped out.
- Conflict event: `argmax P_id != argmax P_ctx` AND `JSD(P_id, P_ctx) > τ`, τ from val only.

Later, the transformer is evaluated on these events via `Δt = log Pθ(a_ctx|t) − log Pθ(a_id|t)`. The predictors define the question; they never feed the transformer.

## 2. Data

`research_events.parquet` (local copy `cleaned_dataset.parquet`, identical bytes): 658,905 rows × 40 columns. Splits by match, reused as-is: train 525,893 (280 matches), val 68,003 (35), test 65,009 (35). Vocabularies reused as-is: `player_ix` 351 (350 + UNK 0, covers 68.8% of rows), `team_ix` 214, `role_ix` 25, `comp_ix` 21.

Target: `action_id` 0–14 (PASS_SHORT 30.4%, CARRY 22.0%, PASS_LONG 20.1% ≈ 72%; SHOT 1.4%, OTHER 0.1%). Code 15 (`END`) appears only in `prev_action_id` on the 706 period-opening rows — output head is 15-wide, the `prev_action_id` table 16 rows.

Inputs (19): identity `player_ix, team_ix, role_ix, comp_ix`; situational `zone, period, minute_bin, score_diff_idx, play_pattern_id, poss_len, new_possession, prev_action_id, prev_outcome, prev_end_zone, same_player_as_prev` + continuous `x_norm, y_norm, dt, t_poss` (log1p, clipped; not re-transformed). Grouping follows EDA §17. One `comp_ix` value (5) has zero train rows — hits 2,127 val + 1,788 test rows, 650 in the conflict set; flagged via `has_untrained_comp`, kept per team decision.

## 3. Models

All three: summed embeddings → 15-logit head, unweighted cross-entropy, Adam 3e-3, batch 8192, 25 epochs, best val-NLL checkpoint.

| model | inputs | arch | params |
|---|---|---|---|
| identity-full (default `P_id`) | player+team+role+comp | 64-dim embeddings, summed → linear | ~41k |
| identity-player-only (`P_id_po`) | player only | 64-dim embedding → linear | ~23k |
| context (`P_ctx`) | 11 cat + 4 cont | 64-dim embeddings + linear cont projection, summed → MLP 64→64→15 (ReLU) | ~60k |

Team/role/comp sit inside identity per EDA §17 and the RisingBALLER-vs-Romero debate; player-only ships as an ablation. Loss is unweighted by design — these are base-rate priors. Encoding notes: `period` 1–4 → table of 5; `play_pattern_id` 1–9 shifted to 0–8.

## 4. Calibration and leakage

Independent temperature scaling per model on val (Guo et al. 2017): T = 0.970 / 0.975 / 1.011, NLL shift ≤ 0.001, ECE 0.01–0.02 before and after. Models were already calibrated; the JSD comparison is valid. JSD is the true divergence in nats (0–ln2), computed directly — `scipy.spatial.distance.jensenshannon` returns its square root and is not used.

Leakage check passes: rebuilding context inputs while perturbing identity columns yields byte-identical Model 2 logits (`byte_identical: true`; identity columns never enter its constructor). Per-player error std 0.34 — positioning carries weak identity signal, reported, not acted on.

## 5. Conflict set

Val argmax-disagree rate 0.646. τ swept over the disagree-JSD percentiles on val; picked p75 = 0.3492 (first τ with conflict rate 10–25% and top-pair share < 45%). Result: 22,472 events = 16.9% of val+test (10,990 val + 11,482 test).

Top `(a_id, a_ctx)` pairs: (PASS_SHORT, CARRY) 4,270; (PASS_SHORT, DUEL) 3,904; (PASS_SHORT, INTERCEPTION) 2,542; (PASS_SHORT, FOUL_WON) 1,545; (PASS_LONG, GK_ACTION) 1,264. Top pair 19% — no dominance. Actual-action mix: CARRY 0.24, DUEL 0.20, INTERCEPTION 0.13; PASS_SHORT 0.03 — the set surfaces non-habit events. Coverage: 96/96 zones, 21/24 minute bins, 9/9 play patterns, 70/70 matches, 83 teams (top-5: 27%). Players: 179 dedicated + shared UNK (83% of rows vs 68.8% base — expected, the prior is weakest where evidence is thinnest); top-5 dedicated hold 12%.

Each row carries `match_id, event_id, index, split, player_ix, a_id, P_id(a_id), a_ctx, P_ctx(a_ctx), JSD, action_id, zone, minute_bin, play_pattern_id, has_untrained_comp` — a plain merge against transformer logits, no re-deriving.

## 6. Results

| model | val acc / top-3 / NLL | test acc / top-3 / NLL |
|---|---|---|
| identity-full | 0.336 / 0.739 / 1.880 | 0.320 / 0.721 / 1.925 |
| identity-player-only | 0.323 / 0.736 / 1.956 | 0.301 / 0.717 / 2.000 |
| context | 0.613 / 0.943 / 0.931 | 0.613 / 0.938 / 0.936 |

Identity predicts habit (PASS_SHORT 0.83 recall, rest ~0). Team/role/comp add +1.3pp over player-only. Context spreads signal (CARRY 0.90, DUEL 0.87, SHOT 0.53, GK 0.90). Test matches val everywhere.

Tuning (context val acc): linear 32-dim baseline 0.572 → MLP head 0.594 → 25 epochs 0.603 → width 64 0.613. Kept the best config.

## 7. Files

- `train_refs.py` — exact training source (ran on T4, ~10 min).
- `reference_predictors_and_conflict_set.ipynb` — 24 cells: full pipeline code verbatim (sections 0–8) + validation replay from saved artifacts (section 9).
- `outputs/action_vocab.json`, `outputs/model{1_identity_full,1_identity_playeronly,2_context}/` (weights.pt, config.json, metrics_val/test.json, confusion.csv; context adds leakage.json), `outputs/predictions/val|test_probs.parquet` (keys + P_id, P_id_po, P_ctx), `outputs/conflict_set/conflict_set.parquet + tau_selection.json`, `outputs/writeup.md` (brief).
- EDA notebook untouched (its `DATA_PATH` placeholder left as documentation).

## 8. Open questions for the team

Primary `P_id`: identity-full or player-only? Keep or drop flagged `comp_ix` 5 rows at arbitration read? τ at p75 (17%) vs p80 (13%) / p85 (10%)? Untested: LR 1e-3, explicit player↔role interpolation.
