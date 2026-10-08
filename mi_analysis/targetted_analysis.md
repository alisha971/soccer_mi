# Targetted analysis: corrected identity and linkage diagnostics

Existing-checkpoint re-analysis, 2026-10-08. All results below are exploratory validation analyses, not fresh confirmation. No Transformer checkpoints were retrained and the test split was not opened; only the analysis probes were fitted.

## Scope and reproducibility

The benchmark is the available **4L/2H/K16, wd=0.1** model, seeds 100–104, primary seed 103—not the historical wd=0.01 selected model. Anonymous models are the existing **K32, wd=0.1** checkpoints, seeds 0–2. No direct performance attribution between those designs is attempted.

Run `python -B mi_analysis/targetted_analysis/run.py` with the existing rerun dependencies. `--section identity|history|interaction|slots|probe|report` runs individual phases; `--check-only` runs the small regression check. Full populations are used; no event cap or smoke mode. The runner reuses named notebook functions without executing notebook write cells. Machine-readable results and input fingerprints are in `mi_analysis/targetted_analysis/results/`.

Intervals are 95% match-bootstrap intervals (2,000 draws, seed 0); seed variability is displayed separately, not pooled into these intervals. New thresholds were not chosen from these results. Across-seed agreement below is a consistency check on the same validation data, not independent data replication.

## 1. Corrected identity specificity and habit-subspace interventions

Frozen reference codes are retained in separate columns; model identity codes come from the actual encoded inputs. Matched partners and role centroids use training-only model indices. Reference-player distributions are joined through physical player IDs. The runner asserts one-to-one positive-player mapping and equality of categorical/vector full swaps on previously misaligned events. The original 461/1,744 player-code mismatches are not discarded: all 1,744 P-player events remain eligible.

Estimation-half results (529 events before any undefined-control exclusions). C2 is |matched swap Δ| minus the norm-matched random-vector magnitude. C3 is the correlation across 15 action log-probability changes with the correctly mapped player-only reference change.

| Seed | Removal magnitude | C2 vs norm-random | Role-centroid player-specific share | Habit-part C3 | Rest-part C3 | Habit minus rest C3 |
|---|---|---|---|---|---|---|
| 100 | 0.4612 [0.3987, 0.5372] | -0.1182 [-0.2054, -0.0055] | 0.9462 [0.8199, 1.0476] | 0.1723 [0.1143, 0.2472] | 0.0097 [-0.0376, 0.0597] | 0.1626 [0.0956, 0.2379] |
| 101 | 0.4726 [0.4048, 0.5318] | 0.0062 [-0.0938, 0.1187] | 0.9494 [0.8783, 1.0013] | 0.1798 [0.1317, 0.2517] | 0.0199 [-0.0310, 0.0871] | 0.1599 [0.1112, 0.2132] |
| 102 | 0.5546 [0.4942, 0.6187] | -0.0863 [-0.2447, 0.0908] | 0.8919 [0.7767, 0.9873] | 0.2135 [0.1809, 0.2559] | 0.0150 [-0.0257, 0.0685] | 0.1984 [0.1430, 0.2519] |
| 103 | 0.4687 [0.3799, 0.5464] | -0.1029 [-0.1806, -0.0054] | 0.9208 [0.8390, 0.9801] | 0.2070 [0.1621, 0.2707] | -0.0326 [-0.0908, 0.0310] | 0.2396 [0.1779, 0.3138] |
| 104 | 0.4930 [0.4191, 0.5566] | -0.0760 [-0.1826, 0.0750] | 0.9238 [0.8158, 0.9974] | 0.1317 [0.0788, 0.2055] | 0.0427 [-0.0002, 0.1098] | 0.0889 [0.0103, 0.1670] |

C3 measures directional alignment across action scores, not the magnitude or fraction of the habit-mediated effect. These correlations alone do not support calling that component small, dominant or exclusive.

Under the original positive-C3 plus paired-superiority rule, habit-part use passes on **5/5 seeds**. Matched swaps exceed norm-matched random perturbations with CI above zero on **0/5 seeds**. These are different questions: generic player sensitivity does not establish identity specificity or an exclusively habit-based mechanism.

The beyond-role embedding-decoding results in MB21 are separate stored-information evidence; this phase corrects the causal swap/reference alignment, not that decoding grid. Discovery-half results and distribution-matched controls are retained in the JSON. These corrections do not silently replace every downstream result in MB04–MB21.

## 2. History versus current-event context with isolated donors

Donors are sampled from another match in the **same discovery/estimation half and at the same scoring position**, RNG seed 0. Current-event fields are asserted unchanged during history swaps; identity fields are restored during whole-context swaps.

Rows without a valid donor: **3**. Cross-half donors: **0**. The ratio CI resamples numerator and denominator jointly.

| Estimation measure | Mean [95% CI] | Events |
|---|---|---|
| history swap | 0.6292 [0.5943, 0.6635] | 5179 |
| scored-event context swap | 6.7109 [6.5639, 6.8441] | 5179 |
| whole-context swap | 6.6975 [6.5562, 6.8298] | 5179 |
| history / whole-context magnitude | 0.0939 [0.0891, 0.0988] | 5179 |

For genuine (non-coupled) events across validation, the history/whole-context ratio is **0.1076 [0.1013, 0.1130]**. This stratum exceeds the 10% cutoff and prevents a blanket claim that history is negligible.

The original <10% point-estimate reading rule passes after donor isolation. This concerns these intervention distributions; it does not prove history is universally irrelevant. Genuine/coupled strata and discovery/all-validation results are in the JSON.

## 3. Corrected identity–context interaction-source uncertainty

MB20 factorial/JVP runs are repeated with corrected matched-player indices. The source statistic is the paired sign-aligned ratio for own minus shared gains: its numerator **and** |exact interaction| denominator are resampled together. The unchanged-player factorial endpoint is checked before scoring.

| Component, partner intervention, estimation | Own retained | Shared retained | Source share [95% CI] | Original SOURCE rule |
|---|---|---|---|---|
| L0 MLP (whole) | 0.9246 | 0.8375 | 0.0871 [0.0329, 0.1288] | False |
| L0 MLP activation only | 1.0054 | 0.8882 | 0.1171 [0.0709, 0.1536] | False |
| L0 MLP LayerNorm only | 0.9245 | 0.9529 | -0.0284 [-0.0785, 0.0127] | False |
| L1 MLP (whole) | 0.9554 | 0.7397 | 0.2157 [0.1371, 0.2684] | False |
| L1 MLP activation only | 0.9972 | 0.8139 | 0.1833 [0.1338, 0.2197] | False |
| L1 MLP LayerNorm only | 0.9634 | 0.9593 | 0.0041 [-0.0214, 0.0233] | False |
| L2 MLP (whole) | 0.9777 | 0.6875 | 0.2902 [0.2322, 0.3288] | True |
| L2 MLP activation only | 1.0094 | 0.7304 | 0.2790 [0.2358, 0.3104] | True |
| L2 MLP LayerNorm only | 0.9811 | 0.9668 | 0.0143 [-0.0202, 0.0522] | False |
| L3 MLP (whole) | 0.9597 | 0.8252 | 0.1345 [0.0666, 0.2204] | False |
| L3 MLP activation only | 1.0017 | 0.9370 | 0.0647 [0.0236, 0.1349] | False |
| L3 MLP LayerNorm only | 0.9655 | 0.9302 | 0.0354 [0.0069, 0.0700] | False |
| final LayerNorm | 0.9805 | 0.7809 | 0.1996 [0.1116, 0.2745] | False |
| all four MLPs (whole) | 0.8966 | 0.0617 | 0.8349 [0.6989, 0.9494] | True |

The SOURCE rule still requires own-retained CI low ≥0.8 and source-share CI low >0.2. Components are intervened on separately; their source shares are **not additive attribution percentages**. Norm-matched random-vector factorials and discovery-half results are also retained. This source analysis remains primary-seed exploratory evidence.

## 4. Anonymous labels: link-breaking versus link-preserving controls

The original fresh-target-slot intervention is compared with an injective cyclic renaming of all positive slots in a window. An assertion verifies that the relabel control preserves every pairwise player-equality relation. Linked and unlinked events are scored separately across all three available anonymous seeds. The paired magnitude difference is a diagnostic contrast, **not a calibrated estimate of pure linkage use**: the two interventions perturb different embeddings.

| Seed | Estimation population | Fresh target-slot magnitude | Link-preserving relabel magnitude | Paired fresh minus relabel |
|---|---|---|---|---|
| 0 | linked (n=334) | 0.4013 [0.3629, 0.4477] | 0.4439 [0.3922, 0.4925] | -0.0426 [-0.0921, 0.0163] |
| 0 | unlinked (n=195) | 0.3613 [0.3108, 0.4112] | 0.3580 [0.3214, 0.3911] | 0.0034 [-0.0350, 0.0425] |
| 1 | linked (n=334) | 0.3727 [0.3453, 0.4054] | 0.3893 [0.3483, 0.4280] | -0.0166 [-0.0462, 0.0235] |
| 1 | unlinked (n=195) | 0.4322 [0.3874, 0.4871] | 0.4570 [0.3706, 0.5393] | -0.0248 [-0.1074, 0.0767] |
| 2 | linked (n=334) | 0.4791 [0.3929, 0.5978] | 0.4447 [0.3754, 0.5232] | 0.0344 [-0.0177, 0.1123] |
| 2 | unlinked (n=195) | 0.3469 [0.3104, 0.3949] | 0.4135 [0.3670, 0.4741] | -0.0666 [-0.1186, 0.0022] |

A second, better-matched control gives the target **the identical fresh label** in both arms: one arm changes only the target; the other consistently renames that player at every occurrence. All other player labels remain unchanged. For unlinked events, causal locality requires identical predictions, which is asserted. The linked contrast tests dependence on the earlier same-player label stream while holding the target embedding fixed; it still does not identify a unique matching or habit-retrieval algorithm.

| Seed | Linked estimation: broken minus consistent Δ | Linked estimation: magnitude of that contrast |
|---|---|---|
| 0 | 0.0010 [-0.0439, 0.0405] | 0.1803 [0.1567, 0.2087] |
| 1 | 0.1014 [0.0762, 0.1264] | 0.1792 [0.1594, 0.2111] |
| 2 | -0.0703 [-0.0925, -0.0385] | 0.1863 [0.1563, 0.2190] |

Read signed and magnitude contrasts separately: a replicated nonzero magnitude shows sensitivity, not a replicated directional preference for the context action. Differing signed effects across seeds prevent that stronger interpretation. Unlinked matched contrasts must be exactly zero; their results are retained in the JSON.

Non-dedicated players share slot 0 on **54112/68003 validation events**. The trained checkpoints cannot retrospectively acquire distinct anonymous identities for those players. Nonzero relabel sensitivity means the original MB23 identity effect must not be described as isolated link-breaking evidence. Existing circuit verdicts remain verdicts for the original slot intervention; circuits were not re-selected under a corrected linkage behaviour.

## 5. Probe-control diagnostic

An action probe at `final_normalised` is trained on discovery conflicts and evaluated on estimation conflicts. Its control independently permutes **event-level** action labels within each half, preserving frequencies without making control labels a function of the true class. C is selected by match-grouped CV within discovery only.

| Diagnostic | Estimation accuracy [95% CI] | C |
|---|---|---|
| actual action labels | 0.8923 [0.8768, 0.9069] | 0.01 |
| independently shuffled event labels | 0.2347 [0.2210, 0.2481] | 0.1 |

All probe CV/final fits completed without convergence warnings (iteration limit 2,000); final iteration counts are recorded in the JSON.

This is a targeted control sanity check, not a replacement for all MB05 player/role/context probe experiments. The original label-remapping selectivity estimates remain unsuitable as random-label baselines and should not support paper claims.

## Paper-facing conclusion and preservation

**Use the corrected tables above for the affected comparisons, with their scope and uncertainty.** Keep behavioural player sensitivity distinct from identity specificity; distinguish embedding decodability from causal habit-subspace evidence; describe anonymous interventions as slot sensitivity unless additional link-isolating evidence is obtained. No new uniquely identified or MEANINGFUL circuit is established here.

The unaffected player-removal, patching and component-sufficiency evidence can accompany this narrower interpretation. Broader all-player anonymous-linkage claims require a changed encoding and retraining; they are not prerequisites for the narrower predictive/MI paper. MB12 remains retrospective and invalidated MB16 legacy scrub evidence remains excluded. Predictive full-versus-restricted benchmarking was not rerun by this analysis.

Historical-file write guards and end-of-run SHA-256 comparisons passed in every phase (1563 existing tracked files protected). No historical notebooks, artifacts, manifests, test-access logs or `findings_short.md` were edited. Input snapshot SHA-256: `ac99ddfeb3315fafc8b55decb44ccdf9b863a868d50c89fd742482091235b7d9`. Baseline commit: `581a68eacb75cac7ede309d586c3b934f8efddb1`.

Execution: Python 3.12.14, Torch 2.10.0+cu128, NumPy 2.4.6, CPU (four Torch threads). No test view, retraining, category-adjacency edits or changes to historical hash guards.
