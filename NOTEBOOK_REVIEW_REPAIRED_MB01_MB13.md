# Review of the repaired notebooks through MB13

Review date: 2026-10-05. Scope: `mi_analysis_old/notebooks_repaired`, through MB13, including MB07R, MB11b and MB11c. This is a review of the repaired code **and its saved runs**, with reproducible checks and literature context. It does not treat previously identified defects as still unfixed when the corrected section resolves them.

## Scope, evidence and verification limits

The directory contains **11 in-scope notebooks**: MB02, MB03, MB04, MB07, MB07R, MB10, MB11, MB11b, MB11c, MB12 and MB13. Repaired MB01, MB05, MB06, MB08 and MB09 are absent. Their saved artifacts are dependencies where an in-scope notebook consumes them; their original notebooks are not reviewed as substitutes. MB16 and later notebook experiments are excluded. In particular, results after MB13 in `attention_audit_and_literature (1).md` are background, not evidence that a missing experiment has been performed within this review's scope.

Project context read: [mi.md](./mi.md), [attention audit and literature](./attention_audit_and_literature%20%281%29.md), [Literature_Review.pdf](./Literature_Review.pdf), [earlier review](./NOTEBOOK_REVIEW_MB01_MB13.md), and [repair report](./mi_analysis_old/REPAIR_REPORT.md). The literature PDF's architectural explanations and novelty claims are hypotheses or survey conclusions, not facts established by these notebooks. Relevant primary publications were checked online; links are attached to the associated recommendations below.

Evidence is distinguished throughout:

| Evidence type | What was checked | What it can establish |
|---|---|---|
| Saved trained-model runs | Notebook source, saved outputs, CSV/Parquet/JSON results and configs | The recorded observations and how the code computes them; not an independent reproduction of the trained model |
| Fresh model-free checks | Actual population tables, donor assignments with the notebook's seeds, graph canonicalization, interpolation, hashes, encoded validation inputs | Counts, selection logic, intervention support, provenance and reproducible design/labeling defects |
| Fresh implementation checks | All 20 repair regression checks on CPU, a random-initialized same-shaped 4-layer/2-head/128-dimensional model and real validation inputs | Correctness of tested implementation properties; not numerical replication of SEL-4L |
| Saved implementation checks | MB02 instrumentation tests, five-seed model verification, MB07R regression output, MB10 finite differences | The checks recorded during the trained-model runs |

**Important reproduction limit:** the exact weight-decay-0.01 SEL-4L checkpoints for seeds 100–104 recorded in `config/preregistration.json` are absent under `model_training_evaluation/experiments/hyperparameter_optimisation/runs/base/`. Other local checkpoints do not justify substituting a different trained model. The exact primary model path and the synthetic fallback are recorded in the verification JSON. No fresh trained-model effect, corrected confidence interval, or corrected ranking is asserted here.

The verification scripts execute read-only checks on the reviewed inputs and write into a separate evidence directory. They do not run notebook setup/`%%writefile` cells or unlock/evaluate test inputs. They hash existing locked-test artifacts for provenance. Evidence files:

- [verify_review.py](./mi_analysis_old/review_evidence_mb01_mb13/verify_review.py), [verification.json](./mi_analysis_old/review_evidence_mb01_mb13/verification.json), [verification.log](./mi_analysis_old/review_evidence_mb01_mb13/verification.log), [fresh regression results](./mi_analysis_old/review_evidence_mb01_mb13/regression_tests_cpu.csv).
- [additional_checks.py](./mi_analysis_old/review_evidence_mb01_mb13/additional_checks.py), [additional_checks.json](./mi_analysis_old/review_evidence_mb01_mb13/additional_checks.json): exact wrong-class donor/subset checks, graph sizes, per-node donor changes and signed-interaction counterexample.
- [cell_index.json](./mi_analysis_old/review_evidence_mb01_mb13/cell_index.json): notebook cell identities. **All notebook cell numbers in this report are 1-based and count both markdown and code cells.** Cell IDs and search strings disambiguate them from the earlier report's numbering.

The current repair report's initial synthetic-only/pending-rerun account is superseded by saved corrected outputs now present. Those corrected outputs are reviewed here. A fresh reproduction is still limited by the missing exact checkpoints.

## What the repairs demonstrably achieved

| Notebook / earlier finding | Verified corrected behavior and saved observation | Remaining boundary |
|---|---|---|
| MB02 instrumentation | Five seeds record identical recomputed logits and validation NLL to their recorded runs; all 10 instrumentation rows pass. Instrumented/no-op/self-patch/arithmetic checks are explicit. | Fresh checkpoint reproduction unavailable locally; see R18–R19. |
| MB03 / F10 | Correct calibration-in-the-large, per-action-pair aggregate comparisons and a 10-bin reliability curve replace the old mean-log-odds identity. | These diagnostics concern the selected conflict population; R01. |
| MB04 / F12–F13 | Explicit denominator table and post-hoc swap-coherence split. It preserves the failed C2 rather than changing the hypothesis after seeing it. | One-match coherent estimation subset and limited control coverage; R02–R04. |
| MB07 / F01, F08 | Writes repaired shared gradients and distinct-circuit utilities; MB07 itself does not use these changed functions. | Its signed head-pair interpretation still needs correction; R05. |
| MB07R | Saved **20/20** repair checks pass. Fresh CPU checks also pass **20/20**, using the synthetic model described above. All **15** in-scope `%%writefile` payloads match their corresponding `mi_analysis_old` files, ignoring the final newline. | Passing checks do not cover the new subset-labeling and control-matching defects found here. |
| MB10 / F01 | Correct internal receiver gradients; saved finite differences at **29** receiver sites. Screen/exact Spearman correlation improves from **0.2181 to 0.7538** on 605 originally tested edges, and is **0.7449** on 586 internal edges. Corrected completeness table has two optional identity edges. | No context optional edge is promoted; selection/completeness limits R06–R07. |
| MB11 / F08–F09 | Evaluates all **8** distinct size-matched component sets per seed. Implements a genuinely recursive identity edge-graph test: selected graph recovery **0.9311 [0.8972, 0.9878]**, component-induced graph **0.9984**, all edges ≈1, empty graph 0. | Dense component set, context graph not tested, named-edge random-control mismatch; R08–R09. |
| MB11b / F02–F04 | Paired donors, all three self-scrub invariants and scoring-preserving wrong-class labels. On identical donors, identity-effect recovery changes from old-rule **0.00349** to corrected **0.71376 [0.61654, 0.78220]**. Arbitration recovery **0.97935**. Independent floor check **0.00259 [-0.00036, 0.00664]**. Corrected wrong-class recovery **0.23687**. | The historical default is not a stated computation; H1 is post-hoc, and control/subset/mechanism issues remain; R10–R13. |
| MB11c / F05–F06, F22 | Frozen discovery scale, beta-zero no-op, an explicitly checked local range, and honest model comparison. Selected piecewise calibration has estimation RMSE **0.05301**, with **13/13** grid predictions inside estimation intervals. Low-rank projected recovery is reported without inferring rank from total variance alone. | Same-beta population transfer rather than unseen-beta/mechanistic prediction; R14. |
| MB12 | Annotates M4 as invalid, S1–S3 as failed, and M2/C2 as negative. Existing test results are loaded rather than reopened. Locked config hashes still match. | Corrected analyses are not held-out confirmations; R17. |
| MB13 / F14–F21 | Event-keyed resample-other; observed no-op coverage; unordered adjacency skipped; BH-on-positive-magnitudes removed; actual factor mean-ablation; joint and single changes use the same endpoints and signed non-additivity. Depth restoration is labelled a sanity check. | Unbounded shifts, joint-support issues and intervention-specific importance; R15–R16. |

For MB10, the saved maximum relative finite-difference error is directly available in `MB10_gradient_check_corrected.csv`; the fresh synthetic check records **3.81e-08**. These are separate runs. Corrected internal zero scores are not automatically defects: structurally irrelevant offsets or a genuine zero effect can produce zeros. The prior defect was severed gradient flow, which the checks now catch.

## Findings requiring review

### R01 — Population restriction and calibration aggregation still limit the central interpretation

**Status:** verified scope limitation; the repaired notebooks often disclose it correctly. **Affected:** MB03 behavior/calibration, MB04 counterfactuals, MB07 arbitration ablations, MB10–MB11c circuits and MB13 context rankings.

**Proof.** The actual validation population contains 10,691 reference-defined conflicts: **1,744 P-player** rows and **8,947 index-0** rows. Discovery and estimation conflict matches are disjoint: 18 and 17 matches respectively. Only P-player rows have the dedicated player factor studied by the identity-removal circuit. These counts are independently recomputed in `verification.json: population` and saved in `MB04_denominators_corrected.csv`. MB13 explicitly operates on P-player only. Specificity covers only **332/1,744** matched rows overall and **81/529** estimation rows; it does not assess the 1,412 unmatched P-player rows.

Conflict selection is conditioned on disagreement between the context and identity references. The context reference uses the current event's context fields. Thus a context-favoring full-model result on this selected population is not an unbiased estimate of the model's reliance on context over all events. Index-0 removal being a no-op establishes the intervention's scope, not that identity has no predictive or indirectly recoverable information on those rows.

The repaired calibration diagnostics are a real improvement. For example, estimation P-all has mean `p(a_ctx)` **0.86403**, realized frequency **0.87799**, difference **−0.01396 [−0.02133, −0.00663]**. This is evidence of aggregate underprediction of this selected action probability, not evidence for the former blanket overconfidence inference from mean log odds. The notebook also supplies per-action-pair aggregate comparisons, genuine/coupled and B5-agreement splits, and a **10-bin P-all reliability curve**. Its lowest probability bin has predicted mean **0.25647** versus realized share **0.35701**. These are useful calibration diagnostics already implemented; they still concern conflicts and the selected `a_ctx` probability rather than all classes over all events. Reliability-bin rows have no match-bootstrap uncertainty intervals.

**Effect on runs:** the recorded statistics are valid for their stated rows. No numerical correction is implied by this scope restriction. **Effect on interpretation:** avoid general statements that attention is unimportant, player identity is absent, or context dominates all next-event prediction. The primary identity circuit addresses 1,744 selected validation rows; broader all-event NLL effects in MB07 answer another question.

**Review direction:** retain denominators and explicit population labels; test context-uncertain, non-conflict and phase/action strata using a separate selection rule. If calibration becomes a central claim, extend the existing reliability diagnostics with match-level uncertainty and compatible classwise/all-event evaluation.

**Location:** MB03 cells 24–25, IDs `r9327eda80`, `rf23c201d0`; search `probability_calibration`, `reliability_bins` / `summary_corrected`. MB04 cell 29 `r21924bf00`, `denominators`. MB13 cell 28 `r0c0b64ca0`, `All of Section R uses the P-player rows only`.

### R02 — Matched controls cross the discovery/estimation match partition

**Status:** confirmed experimental-design defect in the in-scope population construction, with a related uncertainty limitation. **Affected:** validation conflict–control specificity in MB03/MB07/MB11 and MB12's saved validation comparison.

**Proof.** MB02 builds controls from the validation pool and assigns a control the **partner conflict's half**. It does not constrain candidate controls to the partner's discovery/estimation match partition. Fresh table checks find:

| Half assigned to the pair | Pairs | Control event belongs to a match in the opposite conflict half |
|---|---:|---:|
| Discovery | 251 | **12** |
| Estimation | 81 | **19** |

Across all 332 pairs, **139** control events come from a different match than their partner conflict. The P-player conflict split itself remains disjoint; the failure is that the controls do not respect that split. Proof: `verification.json: control_match_overlap`, `control_pairs_cross_match`, and the two population Parquets.

Paired specificity is bootstrapped by the conflict partner's match. A cross-match control contributes behavior from a second match. The same control match can influence pairs attributed to other match clusters. Resampling only partner matches does not preserve all dependence implied by this crossed matching structure.

**Effect on runs:** 31 validation pairs reuse match information across the nominal discovery/estimation boundary. **Not measured:** the change in specificity estimates or interval coverage under within-half matching or a dependence-aware resampling scheme. This does not establish that the estimates are biased upward, that all candidate selection used these controls, or that test controls contain validation events. The latter is not asserted.

**Effect on interpretation:** “independent estimation” is too broad for the control-based validation statistics. The matched-control inference has a weaker independence justification than the conflict-only half split.

**Review direction:** build candidates within the assigned half; report both event-match IDs; preserve match-connected dependencies when resampling. Recompute affected specificity only after an explicit matching rule is fixed.

**Location:** MB02 cell 26 `d9f2ce6e`, search `control_pool`, `partner_match_id`, and the assignment of `half`; MB07's specificity summary and MB11's `specificity` merge; MB12 cell 8 `8bee90b5`, `conflict_tests`, M3 merge followed by `summarise(..., paired['match_id'], ...)`. Exact construction is indexed again below.

### R03 — The coherent-swap estimation interval is based on one independent match

**Status:** confirmed uncertainty/interpretation problem; no model computation defect. **Affected:** MB04's repaired coherence diagnostic and its concluding reading rule.

**Proof.** `MB04_C2_by_coherence_corrected.csv` reports:

| Half / post-hoc coherence | Events | Matches | C2 mean | Match-bootstrap interval |
|---|---:|---:|---:|---|
| Estimation / coherent | **16** | **1** | −0.45698 | **[−0.45698, −0.45698]** |
| Estimation / incoherent | 513 | 12 | −0.05411 | [−0.13520, 0.05064] |
| Discovery / coherent | 285 | 5 | −0.11987 | [−0.14735, −0.05131] |

The code skips blocks with fewer than ten **events**, not fewer than an adequate number of **matches**. Resampling a single match always returns the same event aggregate, hence the degenerate interval. Sixteen events are not sixteen independent match clusters.

The coherence label is a limited post-hoc proxy: same competition matching rule, partner's modal training team equal to the event team, and original `same_player_as_prev == 0`. It does not establish actual lineup availability or a physically realized substitution. I additionally checked whether any of the 301 labeled coherent swaps makes the partner equal to the previous encoded actor while retaining flag 0: **none did in these saved rows**. Therefore that possible consistency gap is not listed as an observed run error.

**Effect on run:** the one-match interval is mechanically degenerate. **Effect on interpretation:** the negative value describes this match; it does not establish population-wide robustness of C2 to plausible substitutions. The repaired sentence “if C2 is also negative on coherent swaps, the negative result does not depend on implausible combinations” needs this sample-size qualification. Original C2's failure remains a legitimate negative result.

**Review direction:** report this row as descriptive, with one match and no population-level uncertainty claim. A stronger conclusion needs eligible substitutions over more independent matches and a coherence rule fixed before evaluation.

**Location:** MB04 cell 29 `r21924bf00`, `if len(block) < 10`, `coherent (post-hoc)`; cell 30 `r1d666a3f0`, the reading sentence. Proof table above and `additional_checks.json: coherent_label_history_contradictions`.

### R04 — The recorded player analysis does not follow the stated fallback rule

**Status:** confirmed protocol/documentation mismatch. **Affected:** justification for retaining player identity in downstream notebooks.

**Proof.** `preregistration.json: fallback_rule_MB04` says to set identity to team, then role if matched-swap effects do not exceed norm-matched random-vector effects. MB04 estimation C2 is **−0.06630 [−0.14801, 0.04513]**; discovery C2 is **−0.16810 [−0.22771, −0.10936]**. The superiority criterion is not met. Current `mi_config.json` nevertheless has `identity_variable: player`. The repaired MB04 still leaves the researcher decision/reason blank. Downstream saved runs are player analyses.

**Effect on runs:** player remains the analyzed variable. The failed criterion does not mathematically invalidate subsequent player-removal experiments, and no claim is made that team or role would have succeeded. **Effect on interpretation:** these runs cannot be presented as an uninterrupted execution of the stated fallback protocol. A deliberate decision to retain player is scientifically possible, but its deviation and rationale are not recorded here.

**Review direction:** record the deviation honestly; do not retrospectively describe an unwritten decision as preregistered. If team/role analyses are desired, define their own substantive hypotheses and controls rather than treating a change of variable as an automatic cure.

**Location:** MB02 cell 30 `461c704c`, `fallback_rule_MB04`; MB04 cells 25–27, especially cell 27 `63f5a064`, `Researcher decision`; `config/mi_config.json`; `results/MB04_summary.json: C2`.

### R05 — Signed head-pair non-additivity does not have the stated universal overlap/compensation sign

**Status:** confirmed interpretation error. **Affected:** MB07's head-pair heatmap and compensation narrative.

**Proof.** MB07 computes signed changes in Δ, then `interaction = pair_change − (change_a + change_b)`. Its markdown says a negative interaction means overlap and positive means compensation. Δ is a signed context-minus-identity logit contrast, so an ablation can move it in either direction.

A direct counterexample: intact Δ=1; either head alone maintains Δ=1; removing both gives Δ=0. The two heads are redundant context-supporting routes, yet the statistic is **−1**, which the notebook calls overlap rather than the claimed positive compensation. Reversing the behavioral direction reverses this statistic's sign without changing the redundancy structure. This is an algebraic proof, retained in `additional_checks.json: pair_sign_counterexample`.

**Effect on runs:** pair changes and intervals remain valid signed factorial contrasts; no values require correction. **Effect on interpretation:** the sign alone does not identify overlap, redundancy or a backup circuit. A large conditional damage effect can have either sign on Δ. Measuring an actual downstream response after removing a primary route is needed for a self-repair claim. [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940), [Explorations of Self-Repair](https://arxiv.org/abs/2402.15390).

**Review direction:** label the heatmap signed non-additivity; report individual and joint directions before using redundancy language; use an output-grounded conditional effect and inspect responding components.

**Location:** MB07 cell 25 `79b26223`, `interaction = pair_change`; cell 27 `1e602841`, `a negative interaction`; `results/MB07_pairs.csv`.

### R06 — Corrected EAP screening still provides no context completeness test outside the required candidates

**Status:** verified coverage and bootstrap-resolution limitation; corrected gradients work. **Affected:** MB10 optional-edge screening and completeness interpretation.

**Proof.** The corrected screen uses at most **500** bootstrap resamples, while the full exact analysis uses 2,000. The two-sided screen p-value floor is **2/501 ≈ 0.003992**. The actual corrected artifacts show:

| Contrast | Screened edges | Non-candidate-sender edges available | Screen BH passes | Optional edges exactly tested |
|---|---:|---:|---:|---:|
| Context | 3,794 | **3,389** | **0** | **0** |
| Identity removal | 254 | 54 | 31 | **2** |

With BH q=0.05 over 3,794 edges, even p-values at the floor cannot be rejected unless the corresponding ordered rank reaches approximately **303**. This is a resolution constraint, not proof that no context edge matters. By comparison, the 405 required context edges are tested exactly and **60** pass exact BH; the screen cannot be interpreted as an exhaustive negative result.

The two corrected optional identity results are:

- `L3H1 → logits`, recovery **0.01122 [0.00373, 0.02082]**, q≈0.02727.
- `L1ATTN → L3H0.k`, recovery **0.000366 [0.000088, 0.000803]**, q≈0.04953.

These are small, positive exact effects on selected edges. They are not evidence of two new large components: the corrected component list remains unchanged, and neither effect establishes the absence of other routes. EAP remains a first-order approximation; its improved correlation is about 0.75, not perfect ranking. [Attribution-patching paper](https://arxiv.org/abs/2310.10348).

**Effect on run:** 3,389 optional context edges receive no exact test after screening; two optional identity edges do. **Not measured:** which additional edges a better-powered, different or unscreened completeness design would find.

**Review direction:** describe completeness as conditional on this screen and budget. Assess p-value resolution before using a thousands-edge significance gate; evaluate a discovery-ranked, independently estimated sample of excluded edges or a prespecified expanded candidate set.

**Location:** MB10 cell 28 `r1ada1cc80`, `run_screen`, `min(N_BOOT, 500)`; cell 30 `rff1cfd470`, `select_edges`; cell 31 `rc4f045d80`, completeness. `verification.json: screen_by_contrast`; `MB10_edge_screen_corrected.csv`, `MB10_edges_corrected.csv`, `MB10_completeness_corrected.csv`.

### R07 — Optional corrected edges are selected and assessed on the same estimation data

**Status:** confirmed selective-inference limitation. **Affected:** optional MB10 completeness discoveries, not every required candidate edge.

**Proof.** Corrected `run_screen` calculates EAP effects on the estimation table; `select_edges` chooses optional edges by those scores and BH decisions; `exact_patch` evaluates them on the same estimation events. `summarise_edges` then applies BH to the **607 tested edges** rather than the whole 4,048-edge screen universe. The 605 required edges originated from the prior candidate rule and do not share this particular optional-screen selection step.

**Effect on run:** the same events influence both optional promotion and exact significance. Exact patching verifies the intervention's actual response, but does not create an independent inference sample. **Effect on interpretation:** the two optional discoveries are post-hoc selected exact effects. The reported q-values do not by themselves establish unconditional FDR control over the searched optional universe. This review does not quantify false positives or assert that either edge is false.

**Review direction:** freeze optional choices on discovery data and estimate their effects on independent matches, or specify an inference procedure that incorporates selection. Keep the repair's post-hoc label.

**Location:** MB10 cells 28 and 30, IDs `r1ada1cc80`, `rff1cfd470`; search `table = estimation`, `optional[optional['EAP passes BH']]`, and `benjamini_hochberg(out.loc[tested, 'p']...)`.

### R08 — The component set remains nearly the whole model; the context edge graph is still untested

**Status:** verified circuit-specificity/coverage limitation, now partly disclosed by the repair. **Affected:** MB11 sufficiency, minimality and descriptions of the combined identity–context circuit.

**Proof.** The primary set contains **11 of 12** heads/MLPs: all four MLPs and seven of eight heads. There are exactly **8** type/size-matched alternatives, each omitting one head. The corrected table enumerates them once for each seed. Selected-set ranks for seeds 100–104 are **3, 3, 5, 3, 2**. Thus the selected set is not even the highest-F member of this narrowly constrained space. Primary F≈**1.01116** establishes recovery under the specified mean ablation; it does not establish sparsity or uniqueness.

The repaired recursive edge test genuinely validates the **identity-removal, target-only, offset-0** graph jointly. It explicitly excludes the context contrast, which changes an entire window. Individual context edge effects remain individual tests. No full-context joint-graph result exists in the in-scope corrected table.

**Effect on run:** the new enumeration removes the misleading impression of 100 distinct component architectures; identity graph recovery is materially supported. **Effect on interpretation:** the validated identity graph is stronger evidence than the old component-set-only claim, but it does not validate all edges in a combined diagram or identify a compact, unique arbitration mechanism. A negative M2 is a legitimate outcome, not an implementation failure.

**Review direction:** distinguish component sufficiency, identity-edge faithfulness and context-edge support. Compare smaller component/edge subsets and structurally meaningful alternatives. Later work mentioned by the notebook is outside this review; its existence is not used to close this gap. [Path patching](https://arxiv.org/abs/2304.05969).

**Location:** MB11 cells 25–28, IDs `r34abf3890`, `r0e39f9500`, `rcdc269d00`, `r45d613270`; `MB11_random_circuits_distinct_corrected.csv`, `MB11_edge_graph_faithfulness_corrected.csv`; `src/graph.py` module scope restriction.

### R09 — “Same-size” random edge graphs have different numbers of actual connections

**Status:** confirmed control-matching defect; not a failure of the recursive graph arithmetic. **Affected:** MB11 corrected random-graph comparison and its joint-faithfulness rule.

**Proof.** The graph representation has aliases: `embedding` and `factor_player` mean the same identity-difference atom; `LxATTN` expands to both heads; explicit head edges can duplicate those connections. `kept_atoms` correctly deduplicates atoms per receiver. However, random graphs are sampled by **named edge count** before canonicalization.

Fresh reconstruction using the exact seed 0 and 100 random sets from cell 27 finds:

| Graph quantity | Named edges | Distinct atom→receiver connections |
|---|---:|---:|
| Whole identity universe | 254 | 175 |
| Selected identity graph | **36** | **30** |
| Each random graph | **36** | **35–46**, median **40** |

Proof: `additional_checks.json: graph_canonical_counts`; the random sets are reconstructed with `graph.random_edge_sets(..., 36, 100, 0)`.

**Effect on run:** the saved random 95th percentile **0.08727** is drawn from functionally larger and differently distributed graphs, not exactly 30-connection graphs. The selected graph's own **0.93114** recovery, all/empty endpoints and recursion remain valid. More connections need not monotonically improve this signed, nonlinear response, so neither the direction nor magnitude of a corrected baseline change is inferred.

**Effect on interpretation:** “same size” is true only for the alias-bearing input list. It is not matched circuit size at the level actually evaluated. A random graph can also be poor because it lacks a connected useful route, rather than because the selected edges constitute a uniquely identified mechanism.

**Review direction:** canonicalize once; sample the canonical connection universe; match connection count, and consider connected/degree/layer-distribution controls if making a stronger specificity claim. Re-evaluate the comparison before treating the random threshold as a size-controlled result.

**Location:** MB11 cell 27 `rcdc269d00`, `random_edge_sets`; `src/graph.py: sender_atoms`, `kept_atoms`, `all_edges`, `random_edge_sets` (lines 44, 76, 87 and 182 respectively).

### R10 — The repaired “non-self donor” subset is incorrectly labelled for wrong-class controls

**Status:** **confirmed new implementation/labeling defect**, reproduced without any model. **Affected:** MB11b R5 filtered right-versus-wrong hypothesis comparisons.

**Proof.** Cell 35 constructs `fully` from class sizes under the **true labels**, then uses the same mask for both right-class and wrong-class rows. Wrong-class assignment permutes complete labels. This preserves the total singleton count, but moves singleton locations to different events. Therefore a true-label nonsingleton event can receive a singleton wrong label and donate to itself.

I reconstructed all five wrong-class assignments with the exact R5 seeds and ordering. The table labelled “events with a non-self donor for every keyed node” includes:

| Hypothesis | Retained events | Events with a self-donor at ≥1 keyed node, per wrong-class draw |
|---|---:|---|
| H0 | 408 | **107, 95, 91, 91, 90** |
| H1 shared chain | 296 | **140, 134, 123, 134, 128** |
| H2 player-blind chain | 430 | **82, 79, 78, 80, 79** |
| H3 heads matter | 296 | **134, 127, 138, 132, 136** |

The proof is deterministic and saved in `additional_checks.json: wrong_class_nonself_subset`; `additional_checks.py` follows `permuted_label_assignment` exactly. This does not dispute the corrected preservation of joint tuple counts or the fact that scoring fields stay unchanged.

**Effect on run:** the filtered wrong-class values are not a non-self-only experiment. H1 filtered recovery **0.50981** includes many forced self-donors; its right-class value **0.75428** uses the true nonsingleton mask and excludes forced singleton self-donors. The amount of recovery due to the remaining self-donors has not been measured. It is inappropriate to subtract these values and interpret the gap as one controlled non-self abstraction comparison.

Singleton burden is independently substantial: historical identity-keyed H0 nodes have **121/529 = 22.9%** singleton rows; H1's keyed MLPs have **233/529 = 44.0%**. H1's all-event right-class recovery **0.85723** drops to **0.75428** after its true singleton rows are removed. The all-event result and wrong-class control are still the recorded comparisons; the labeling defect specifically affects the advertised filtered interpretation.

**Review direction:** track actual donor indices for every draw and variant. Either compare variants on a prespecified/common donor-feasible population or report their different retained populations explicitly. A correct mask must apply to each variant's **actual assignments**, including all averaged draws, not merely true class sizes.

**Location:** MB11b cell 35 `r477b913f0`, `labels = [...]`, `fully`, and the subset label; `src/scrub.py` lines 89–113, `permuted_label_assignment`; `MB11b_posthoc_hypotheses_corrected.csv`.

### R11 — The unrelated-class identity control changes the branch rule as well as the class structure

**Status:** confirmed control-interpretation mismatch; branch logic itself is intentional and documented. **Affected:** MB11b R3 “unrelated team” control and H1-versus-H2 identity comparisons.

**Proof.** `branch_modes` gives a node counterfactual donor activations only when `player` is in its keys. Changing every circuit node to `['team']` removes `player` from every class tuple. The control therefore uses **same clean donor activations in both branches at every scrubbed node**, whereas player-keyed H0 nodes use paired clean/player-removed donors. Likewise, H2 drops `player` from all four MLPs and simultaneously forces their paired-effect contribution to be invariant.

This is a coherent operational test of the stated invariance hypothesis, but it is not a granularity-matched control changing only the meanings of labels. Team class sizes are also not preserved relative to H0's classes. The repaired wrong-class permutation is the cleaner branch-rule-preserving comparator because its spec stays unchanged.

**Effect on run:** unrelated-team identity recovery is **0.00160**; H2 right-class identity recovery **0.00143** while arbitration recovery is **0.91736**. The tiny identity responses are partly built into the intervention rule. Direct player embedding/residual and normalization effects can remain, so literal zero is not guaranteed; a synthetic self-donor check confirms the branches can still differ. The exact fraction attributable to class meaning versus branch rule has not been isolated in SEL-4L.

**Effect on interpretation:** these results cannot independently establish that team is a bad abstraction, or that MLPs were discovered to ignore player information. H2's intended invariance is what its scrub enforces. The useful evidence for class content is the matched-spec right-versus-permuted comparison and other independently defined interventions.

**Review direction:** explicitly state the changed class granularity and branch rule. For an unrelated-class comparison isolating class content, preserve branch treatment and match tuple-size distributions; treat a branch-invariant hypothesis as a different test with a different prediction.

**Location:** MB11b cells 31, 34–35, IDs `re1e8d6130`, `r2eedeb0c0`, `r477b913f0`; `src/scrub.py` lines 69–81, `branch_modes`; `MB11b_controls_corrected.csv` and R5 hypothesis table.

### R12 — Corrected per-node drops also change other donors and do not have paired drop uncertainty

**Status:** confirmed attribution/design limitation. **Affected:** MB11b R4 ranking of load-bearing abstractions.

**Proof.** The baseline hypothesis uses donor seed **10**. Every node-freed run uses seed **60**. One RNG supplies assignments across all components, so changing a class grouping also changes RNG consumption. Fresh assignment comparisons show that donors for components other than the freed node change. For example, freeing L0H0 changes donors at **all 11 other components** in the first draw, affecting **196–529 of 529 events** per other component. Counts by component are retained in `additional_checks.json: per_node_other_donors_changed_first_draw`.

In addition, setting `spec[component] = None` changes that node from `paired` to `same` if its keys contained player. Thus the intervention tests removal of its permitted player response together with freer donor classes. The table's `drop from the full hypothesis` is a difference of two recovered-share point estimates. Its displayed CI is the CI for **recovery without the node's class**, not a CI for the drop itself. Five donor draws are averaged before the match bootstrap; the bootstrap conditions on those sampled donors.

**Effect on run:** each recorded drop includes Monte Carlo changes at other nodes, in addition to the intended abstraction edit. **Not measured:** the portion of any SEL-4L per-node drop arising from those extra donor changes, or its joint uncertainty. Independent donor draws can be unbiased in expectation, so this is not proof every per-node result is wrong; it prevents the point difference from cleanly isolating one node in the finite run.

**Review direction:** hold unaffected donor assignments fixed for a paired comparison; decide whether changing branch mode is part of the intended claim; retain draw-level paired contrasts and bootstrap the drop itself, with donor variability assessed separately or incorporated into inference.

**Location:** MB11b cell 29 `rb24d3cdc0`, `run(hypothesis_spec, 10)`; cell 33 `r6a041d290`, `run(spec, 60)`, `base_share - share['recovery']`; `src/scrub.py: averaged_scrub`, `assign_donors`, `branch_modes`; `MB11b_per_node_corrected.csv`.

### R13 — Corrected output resampling supports equivalence classes, not the stated serial computation

**Status:** verified mechanistic-evidence limitation. **Affected:** MB11b's “causal scrubbing of the proposed mechanism” and “shared chain” wording.

**Proof.** `scrub_pair` obtains each donor's head/MLP output from an **intact donor forward pass**, then installs the selected outputs at the recipient's target. Replaced later outputs are not recomputed from recursively scrubbed parent computations. R5 describes an MLP output as a function of a key tuple, but supplies no function, intermediate operation or edge correspondence showing how one MLP uses the previous MLP's result. The historical `computes` sentences are still empty; the repair correctly preserves and labels them. R5 adds meaningful coarse invariance hypotheses, which is progress.

Replacing all target MLP outputs is a valid behavior-preserving resampling experiment for a suitable coarse hypothesis. Its scope should not be confused with a recursive test of a specified chain. The canonical causal-scrubbing construction recursively resamples a mapped computational graph; it can distinguish claims about where and how computations occur. [Authors' formulation and appendix](https://www.alignmentforum.org/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix).

**Effect on runs:** corrected H0 identity recovery is **0.71376**, below the notebook's original **0.80** hypothesis bar, with upper CI **0.78220**; arbitration recovery ≈0.979. H1's 0.857 all-event response is post-hoc and singleton-sensitive (R10). The paired repair invalidates the former inference of almost zero identity preservation, but does not turn H0 into a fully explained mechanism.

**Effect on interpretation:** survival of the chosen scalar output under output swaps supports limited class invariance. It does not prove exact vector-valued dependence `out_c = g_c(keys)` or a serial algorithm; information outside those keys can survive through untouched positions/residual paths or be irrelevant to this measured behavior. High pooled scalar recovery is not full per-event/full-distribution fidelity.

**Review direction:** formulate the computation and its graph correspondence; test the claimed parent→child operations, target and relevant earlier positions, and behavior errors beyond one pooled statistic. Either implement the recursive test appropriate to that hypothesis or name this experiment target-output equivalence-class resampling and retain its narrower evidential claim.

**Location:** MB11b cells 26, 34–37; `src/scrub.py` lines 122–166, especially donor `hooks.forward` and subsequent replacements; `config/mechanism_hypothesis.json`, `config/mechanism_hypotheses_posthoc.json`; `MB11b_scrub_corrected.csv`.

### R14 — Steering passes same-grid population transfer; it does not demonstrate an unseen-dose mechanism prediction

**Status:** confirmed interpretation boundary, not a failure of corrected batch invariance. **Affected:** MB11c corrected calibration, subspace and steering claims.

**Proof.** The chosen model is `piecewise-linear through measured points`. Estimation evaluates exactly the **same 13 beta values** as discovery. At each such beta, interpolation returns that measured discovery mean exactly. Fresh verification finds maximum absolute difference between selected predictions and discovery grid means **0.0**. The saved estimation RMSE is **0.05301**, maximum error **0.09272**, and 13/13 estimation intervals contain the predictions. This genuinely supports transfer across these validation match populations at fixed doses.

Discovery leave-one-beta-out RMSE is **0.95700**, much larger than the same-grid transfer error. The local linear range is only **|beta|≤0.1**, slope≈**5.09848**. Testing the locally fitted line over the full −2…2 grid and seeing failure does not refute local linearity. In `steering.py`, `np.interp` clips outside its training beta range; when holding out an endpoint during LOO, that is a boundary-value predictor rather than an extrapolated linear segment.

The repaired scale is fixed at **5.31269**. Mean final-residual player-swap displacement norm is **1.06370**, so beta=1 shifts by approximately **5 times** that mean norm. This comparison is between displacement norms, not a claim that the direction equals five actual player substitutions. Large-beta behavior cannot be described as a small local identity perturbation.

The subspace evidence is real but task-specific: rank 4 restores **0.82986 [0.74814, 0.88811]** of the swap contrast; rank 8 **0.90601 [0.85634, 0.94181]**. Rank-4's point estimate crosses 0.8, but its lower interval does not. Neither SVD nor steering identifies a football style axis without external semantic evidence. The repaired header's correction of the prior high-rank implication is appropriate.

**Effect on runs:** the batch-dependent-dose defect is fixed. **Effect on interpretation:** the pass is empirical calibration across populations on measured doses; it is not an independently predicted response to a new beta, a globally linear mechanism, semantic player impact or a replacement for failed locked S1–S3.

**Review direction:** state this exact pass criterion; assess local models in their valid range; reserve unmeasured beta values or a new population for a frozen quantitative prediction. Keep doses in physical/model-relevant norm units and separate causal direction utility from semantic interpretation.

**Location:** MB11c cells 24–27, IDs `r64d4699d0`, `rc1a48eae0`, `rd0b04a750`, `r2efc90340`; `src/steering.py: fit_models`, `leave_one_beta_out_error`, `evaluate`; `MB11c_steering_calibration_corrected.csv`, `MB11c_steering_calibration_per_beta_corrected.csv`, `MB11c_projected_patching.csv`; `verification.json: selected_interpolator_is_discovery_grid`.

### R15 — Repaired continuous ±1 SD edits still leave the physical feature domain

**Status:** **confirmed intervention-support problem**, independently computed on the real encoded P-player inputs. **Affected:** MB13 corrected continuous-shift rows and any ranking that treats them as realistic context changes.

**Proof.** Corrected R2 reuses `shift_edit`, which adds/subtracts the population SD with no domain bound. There are 1,744 scored rows; both directions are run and averaged. Actual counts:

| Feature | SD in stored units | Minus edit below physical minimum | Plus edit above physical maximum | Extreme edited value |
|---|---:|---:|---:|---|
| x_norm | 0.28819 | **735** below 0 | **315** above 1 | −0.28485 to 1.28819 |
| y_norm | 0.25919 | **380** below 0 | **340** above 1 | −0.25794 to 1.25919 |
| dt | 0.73733 | **1,308** below 0 | Not applicable | Minimum −0.73733 |
| t_poss | 1.14782 | **270** below 0 | Not evaluated against an upper time cap | Minimum −1.14782 |

`dt` and `t_poss` are stored transformed time quantities. A negative log(1+seconds) corresponds to a negative time interval; ±1 SD in these units is not ±a fixed number of seconds. Proof: `verification.json: continuous_shift_domain`; timing semantics in Track A's data module. The repaired label correctly says “stored transformed units,” but does not constrain the edits.

**Effect on run:** the saved dt ±1 SD effect **4.41656 nats** includes many impossible negative-gap inputs. x/y shift responses likewise include off-pitch values. **Not measured:** their effects after admissible-domain or raw-time perturbations; direction/rank changes cannot be inferred.

**Review direction:** call these unconstrained numerical sensitivity probes. For physically meaningful counterfactuals, define valid units/bounds and regenerate dependent encodings; report boundary handling because clipping also changes intervention magnitude and no-op rates.

**Location:** MB13 cell 11 `G_uY0MlM2Bas`, `shift_edit`, `own + direction * sd`; corrected cell 29 `r20e3aafb0` reuses it. `model_training_evaluation/src/data.py` lines 134–159 and 171; `MB13_single_inputs_corrected.csv`.

### R16 — Context resampling is marginal factor sensitivity, with inconsistent joint representations and unequal edit sizes

**Status:** verified interpretation/support limitation, partly explicitly disclosed. **Affected:** MB13 input-importance ranking, group non-additivity and MB04 whole-context interaction wording.

**Proof.** Resample-other deliberately changes one target field while the rest of the event/window remains fixed. Values are drawn marginally from P-player. Making each edit nonzero repairs dilution, but does not make the resulting joint event plausible or make intervention magnitudes comparable.

One concrete consistency check: all **1,744** original rows satisfy the empirical 12×8 column-major relation between `zone` and `(x_norm, y_norm)` used in the proof script. A zone-only resample-other changes zone on every row while retaining coordinates, so **all 1,744** edited rows break that observed relation. This relation is an **empirically verified encoding pattern**, not a source-confirmed codebook; it is not used to fabricate ordinal/grid edits. The notebooks correctly leave `ADJACENCY_CONFIRMED=False`.

Other edits such as changing the same-player flag while leaving identities/history fixed are tests of model response to a field, not realized event sequences. R3's joint donor copies preserve relationships **within** a selected factor group better than independent single-field edits, but preserve neither relationships to other groups nor the whole window. The notebook itself acknowledges this. A whole-window donor swap also changes many history and identity fields; writing player back does not isolate one context variable.

Corrected single-edit magnitudes are consequently intervention-specific: dt resample-other **5.94847**, possession flag **4.47817**, previous action **4.07239**, previous outcome **3.02217**, same-player flag **2.86153**. A forced binary flip, a categorical redraw and a transformed-time redraw are different doses. Normalization by the whole-window mean is a descriptive ratio, not a fraction of uniquely attributable behavior; single-field shares need not sum to 1.

R3 now computes valid same-endpoint signed non-additivity. For history, mean `joint−sum(single)` is **1.75864 [1.56352, 1.96015]** and mean absolute non-additivity/joint ratio is **0.51312**. This demonstrates non-additivity under the fixed seed-77 donor edit. It does not by itself identify an identity×context gate, a specific processing layer or a conditional player-sensitive mechanism. The match bootstrap conditions on those fixed endpoints; it does not quantify variation over alternative donor assignments.

**Effect on runs:** exact factor-sensitivity and non-additivity responses exist; some single edits create inconsistent redundant representations. **Not measured:** the contribution of such inconsistency to each effect, or a coherent-donor ranking. The old magnitude-BH and group-absolute-gap defects are fixed and are not reintroduced as findings here.

**Review direction:** retain the current literal interpretation; add coherent joint/conditional donors and matched-dose comparisons if using importance or real-world counterfactual language. Cross a selected context edit with player intervention on the same events to test the central identity×context hypothesis. [EventGPT](https://arxiv.org/abs/2512.17266) motivates player substitution, but does not make these factor edits real-world causal impact. [Leemann et al.](https://arxiv.org/abs/2405.13536) motivates testing non-additivity rather than assuming additive embeddings imply additive behavior.

**Location:** MB13 cells 28–32; `resample_other_edit` and `pool_of` in cell 29 `r20e3aafb0`; `donors_r`, `signed_changes` and `joint−sum(single)` in cell 31 `r84ef640d0`. MB04's whole-window C4 design. `verification.json: zone_encoding_consistency_original`, `zone_resample_breaks_xy_consistency`; `MB13_groups_corrected.csv`.

### R17 — Corrected validation results do not repair the original held-out confirmation

**Status:** confirmed evidence boundary, correctly recognized by much of the repaired MB12. **Affected:** claims of preregistration, held-out confirmation and the evidence chain.

**Proof.** `logs/test_access_log.jsonl` records one MI test access on **2026-10-02**. The corrected modules/sections are dated **2026-10-04**. MB11b R5 explicitly says the hypotheses were written after test access, and the corrected files state they are validation-only and outside the MB12 lock. Current circuit, historical mechanism and historical steering hashes match their stored locked hashes; the lock's SHA-256 is `2ec96f6888898138ab745c7daff590b2f39e1cfe5f89151dd30e0156b2bcf4e2`.

The annotated existing test table says:

- C1/C3/C4/M1/M3 meet their directional rules with narrowed interpretations.
- C2 and M2 are preserved negative results.
- M4 is **invalidated** by the historical same-donor implementation and unstated computation.
- S1–S3 are preserved failed quantitative predictions.

The lock's `mechanism_statement` is empty. M4's locked criterion is only recovery **>0**, weaker than the blueprint/default scrub **≥0.80** criterion. Thus even a positive historical M4 would not certify that stronger hypothesis-preservation standard. The repair appropriately removes the earlier Level-6 confirmation sentence.

**Effect on runs:** the saved test observations remain historical records; corrected validation graph/scrub/calibration success is new exploratory evidence. The one-line MI access log is not proof that Track A never used the test matches: MB12 itself acknowledges earlier use. It is evidence of the notebook's guarded MI test access only.

**Effect on interpretation:** do not substitute corrected validation numbers into the old held-out claim or describe the Oct-4 hypothesis as prospectively preregistered. A discovery/estimation split can still give useful within-repair population transfer, but the overall research process has already seen the old test outcomes.

**Review direction:** retain the annotated table and distinctions. A new confirmation requires a frozen concrete hypothesis, implementation/scale/thresholds and data genuinely unseen by the relevant selection process. Reusing the same test split cannot restore its original freshness merely by a new log entry.

**Location:** MB12 cells 2, 6–8, 12, 23 and 27; IDs `rdf2f536b0`, `1cc46f6f`, `8bee90b5`, `7a7dcaa0`, `ee7f996f`, `r036fb5760`; `config/locked_predictions.json`, `MB12_confirmation_annotated.csv`, `logs/test_access_log.jsonl`.

### R18 — In this checkout, repaired notebook setup targets the other project directory

**Status:** confirmed local reproducibility/path defect. **Affected:** all 11 repaired notebooks and source-writing cells.

**Proof.** Every in-scope notebook resolves the project root by requiring a sibling named `mi_analysis`, then assigns `MI_DIR = ROOT / 'mi_analysis'` and changes directory there. The reviewed notebooks/files are under `mi_analysis_old`. This checkout contains both directories. `SOCCER_MI_ROOT` changes the root, not the hard-coded MI subfolder. Source-writing notebooks would therefore write into the active `mi_analysis` directory, while downstream relative artifact reads target that directory as well.

Stored manifests/snapshots also contain absolute Colab paths. `fingerprint.verify_files` and `compare_snapshot` use the recorded paths literally; unlike checkpoint run directories, they are not rebased. The read-only verifier explicitly rebases these paths for auditing, without modifying them.

**Effect here:** a direct top-to-bottom local run can read/write the wrong project or fail on recorded paths/missing exact checkpoints. This is why the review's verification imports the intended old source explicitly. **Not established:** that the historical Colab runs used the wrong source; they may have used the old project under its then-current `mi_analysis` name. Saved writefile bodies agree with the local old source, which is favorable evidence.

**Review direction:** make MI output/source root explicit and assert the intended directory before executing writes. Rebase or relativize artifact manifests in a controlled copied workspace and provide the exact checkpoint artifacts. Do not count a different weight-decay run as replication.

**Location:** setup cell of every notebook, search `MI_DIR = ROOT / 'mi_analysis'` / `os.chdir(MI_DIR)`; examples MB02 cell 2 `f8020c58`, MB03 cell 3 `c04d0133`, MB11b cell 3 `zYOZrtfA1a7U`; `src/fingerprint.py` lines 62–84 and 110–127; `verification.json: notebooks`, `requested_checkpoint`, `writefile_comparisons`.

### R19 — Current provenance checks do not justify a complete immutable-history claim

**Status:** confirmed audit-coverage limitation and unresolved current source-hash mismatch. **Affected:** MB02/MB07R/MB12 integrity and exact reproducibility.

**Proof.** The saved MB12 integrity table explicitly reports **`mi_config.json: false`** and **`reference_models.json: false`** for fingerprint coverage. Those configs can affect the identity/reference setup. The lock stores some key file hashes, which now match, but does not cover every input or source implementation. `record_files` intentionally refreshes entries; current hash agreement establishes consistency with the latest entry, not an immutable chain of historical inputs.

Fresh read-only rebasing finds the current `model_training_evaluation/src/data.py` differs from the saved Track A snapshot digest, despite the saved Oct-4 MB12 integrity output reporting no changes in its environment. Original source bytes corresponding to that digest are not available here for a semantic diff. No claim is made that the change occurred during MI, altered a run, or is more than a checkout/version difference. It is nevertheless an unresolved barrier to claiming the present checkout exactly reproduces that snapshot.

The pre-repair snapshot also differs for several files currently present, including old MB13 result tables, MB12 integrity, identity subspace config and preregistration. Some regenerations are expected from running preserved legacy sections and integrity checks. The preregistration backup-versus-current comparison changes **only `written_at`**, from Oct-4 in the MB02 rerun copy to Oct-2 in the current file. That comparison supplies no evidence of changed scientific preregistration rules. The original pre-snapshot bytes are not inferred from that backup.

**Effect on runs:** no trained-model effect of these source/history differences is measured. **Effect on interpretation:** saved successful integrity checks describe their execution environment, not automatically this checkout or the whole immutable run history. Fingerprinting should be described with its actual coverage and timestamp.

**Review direction:** preserve original bytes/hashes per run and a map to the exact source/checkpoint versions; hash all behavior-changing configs and relevant source modules; resolve the Track A source mismatch before asserting exact local reproduction. Avoid either asserting misconduct or dismissing the mismatch without the missing original evidence.

**Location:** MB07R cell 15 `rac19a2f60`, snapshot comparison; MB12 cell 23 `ee7f996f`, `config_status`, `track_a_changed`; `src/fingerprint.py: record_files`; `manifests/track_a_snapshot.json`, `manifests/pre_repair_snapshot.json`, `results/MB12_integrity.json`; fresh `verification.json` snapshot sections and `additional_checks.json: prereg_backup_vs_current_differences`.

## Findings not asserted, and prior defects that should stay closed

This review found no counterexample to the tested repaired gradient flow, paired self-scrub invariants, fixed steering batch independence, corrected same-endpoint non-additivity arithmetic, score-table preservation or all/empty recursive identity graph endpoints. The regression suite proves the properties it checks; R09/R10 are additional properties it does not check.

No non-upstream legacy control edge was found in the saved MB10 control table, despite a potential concern when reading fallback selection logic. That concern is **not** counted as an observed defect. No actual previous-actor contradiction was found among the 301 post-hoc coherent swaps. No changed scientific preregistration rule is inferred from a timestamp-only backup diff. No claim is made that sparse dictionaries or DePass are required for a valid analysis, that a failed registered test is a bug, or that later notebooks have not done work outside this cutoff.

The old MB13 depth restoration remains guaranteed recovery for the target-only clean residual patch, but its repaired markdown now says this explicitly. It is a sanity check, not an unresolved mislabeled depth-localization result. The old BH magnitude test, code-zero-as-removal label, guessed category adjacency and absolute-group-gap interpretation are likewise corrected/invalidated rather than listed as active defects.

## Precise location index

Cell numbering is 1-based across **all** cells. Search strings refer to the source inside that cell; file line numbers below refer to the Python source file. The [cell index](./mi_analysis_old/review_evidence_mb01_mb13/cell_index.json) records every in-scope cell ID, and the [source index](./mi_analysis_old/review_evidence_mb01_mb13/source_index.json) gives actual function definition lines. Notebook links all target the repaired directory. Short `src/`, `config/`, `results/`, `artifacts/`, `manifests/` and `logs/` paths refer to `mi_analysis_old/` unless Track A is explicitly named.

| Finding | Notebook / exact cell locator | Additional source or artifact locator |
|---|---|---|
| R01 | [MB03](./mi_analysis_old/notebooks_repaired/MB03_behaviour_on_conflicts.ipynb), cell **24**, `r9327eda80`, probability/frequency summary; [MB04](./mi_analysis_old/notebooks_repaired/MB04_input_counterfactuals.ipynb), **29**, `r21924bf00`, `denominators`; [MB13](./mi_analysis_old/notebooks_repaired/MB13_context_input_survey.ipynb), **28**, `r0c0b64ca0`, P-player coverage | `results/MB03_calibration_corrected.csv`; `artifacts/populations/val_conflicts.parquet`; `verification.json: population` |
| R02 | [MB02](./mi_analysis_old/notebooks_repaired/MB02_model_verification_and_preregistration.ipynb), **26**, `d9f2ce6e`, matching construction and half assignment; [MB12](./mi_analysis_old/notebooks_repaired/MB12_locked_confirmation.ipynb), **8**, `8bee90b5`, M3 paired summary | `artifacts/populations/matched_controls.parquet`: `match_id`, `partner_match_id`, `half`; `verification.json: control_match_overlap` |
| R03 | [MB04](./mi_analysis_old/notebooks_repaired/MB04_input_counterfactuals.ipynb), **29**, `r21924bf00`, `if len(block) < 10`; **30**, `r1d666a3f0`, coherent-swap reading | `results/MB04_C2_by_coherence_corrected.csv`, estimation/True row; `additional_checks.json: coherent_label_history_contradictions` |
| R04 | [MB02](./mi_analysis_old/notebooks_repaired/MB02_model_verification_and_preregistration.ipynb), **30**, `461c704c`, fallback rule; [MB04](./mi_analysis_old/notebooks_repaired/MB04_input_counterfactuals.ipynb), **25–27**, especially `63f5a064`, blank decision | `config/preregistration.json: fallback_rule_MB04`; `config/mi_config.json: identity_variable`; `results/MB04_summary.json: C2` |
| R05 | [MB07](./mi_analysis_old/notebooks_repaired/MB07_attention_heads.ipynb), **25**, `79b26223`, `interaction = pair_change`; **27**, `1e602841`, sign interpretation | `results/MB07_pairs.csv`; `additional_checks.json: pair_sign_counterexample` |
| R06–R07 | [MB10](./mi_analysis_old/notebooks_repaired/MB10_path_patching.ipynb), **28**, `r1ada1cc80`, `run_screen` / bootstrap count; **30**, `rff1cfd470`, optional selection, exact table and tested-only BH; **31**, `rc4f045d80`, completeness | `results/MB10_edge_screen_corrected.csv`; `results/MB10_edges_corrected.csv`; `verification.json: screen_by_contrast` |
| R08 | [MB11](./mi_analysis_old/notebooks_repaired/MB11_faithfulness_and_replication.ipynb), **25**, `r34abf3890`, exact alternatives; **26–28**, graph scope/rule | `config/circuit.json: components`; `results/MB11_random_circuits_distinct_corrected.csv`; `results/MB11_edge_graph_faithfulness_corrected.csv` |
| R09 | [MB11](./mi_analysis_old/notebooks_repaired/MB11_faithfulness_and_replication.ipynb), **27**, `rcdc269d00`, named-edge random sets | `src/graph.py`: lines **44**, **76**, **87**, **182**; `additional_checks.json: graph_canonical_counts` |
| R10 | [MB11b](./mi_analysis_old/notebooks_repaired/MB11b_causal_scrubbing.ipynb), **35**, `r477b913f0`, true-label `fully` used for wrong-class result | `src/scrub.py` line **89**, `permuted_label_assignment`; `additional_checks.json: wrong_class_nonself_subset`; R5 CSV filtered rows |
| R11 | [MB11b](./mi_analysis_old/notebooks_repaired/MB11b_causal_scrubbing.ipynb), **31**, `re1e8d6130`, unrelated keys; **34–35**, H1/H2 specs | `src/scrub.py` line **69**, `branch_modes`; `results/MB11b_controls_corrected.csv` |
| R12 | [MB11b](./mi_analysis_old/notebooks_repaired/MB11b_causal_scrubbing.ipynb), **29**, `rb24d3cdc0`, seed 10 baseline; **33**, `r6a041d290`, seed 60 and point drop | `src/scrub.py: averaged_scrub`; `additional_checks.json: per_node_other_donors_changed_first_draw` |
| R13 | [MB11b](./mi_analysis_old/notebooks_repaired/MB11b_causal_scrubbing.ipynb), **26**, formal hypothesis; **34–37**, chain claims and interpretation | `src/scrub.py` line **122**, `scrub_pair`; `config/mechanism_hypothesis.json`; `config/mechanism_hypotheses_posthoc.json` |
| R14 | [MB11c](./mi_analysis_old/notebooks_repaired/MB11c_identity_subspace_and_steering.ipynb), **24–27**, especially **26**, `rd0b04a750`, calibration | `src/steering.py: fit_models`, `leave_one_beta_out_error`, `evaluate`; `verification.json: selected_interpolator_is_discovery_grid`; calibration/subspace CSVs |
| R15 | [MB13](./mi_analysis_old/notebooks_repaired/MB13_context_input_survey.ipynb), **11**, `G_uY0MlM2Bas`, `shift_edit`; **29**, `r20e3aafb0`, corrected reuse | `verification.json: continuous_shift_domain`; `model_training_evaluation/src/data.py` lines **134–159**, **171** |
| R16 | [MB13](./mi_analysis_old/notebooks_repaired/MB13_context_input_survey.ipynb), **29**, `r20e3aafb0`, marginal pools; **31**, `r84ef640d0`, fixed donor endpoints | `verification.json: zone_encoding_consistency_original`; `results/MB13_groups_corrected.csv`; `results/MB13_single_inputs_corrected.csv` |
| R17 | [MB12](./mi_analysis_old/notebooks_repaired/MB12_locked_confirmation.ipynb), **6**, `1cc46f6f`, M4 threshold; **12**, `7a7dcaa0`, empty statement; **27**, `r036fb5760`, annotations | `config/locked_predictions.json`; `results/MB12_confirmation_annotated.csv`; `logs/test_access_log.jsonl` |
| R18 | Every notebook setup: `MI_DIR = ROOT / 'mi_analysis'`; [MB02](./mi_analysis_old/notebooks_repaired/MB02_model_verification_and_preregistration.ipynb) **2**, `f8020c58`; [MB11b](./mi_analysis_old/notebooks_repaired/MB11b_causal_scrubbing.ipynb) **3**, `zYOZrtfA1a7U` | `src/fingerprint.py`: `verify_files`, `compare_snapshot`; `verification.json: requested_checkpoint`, `notebooks` |
| R19 | [MB07R](./mi_analysis_old/notebooks_repaired/MB07R_repaired_sources_and_regression_tests.ipynb), **15**, `rac19a2f60`; [MB12](./mi_analysis_old/notebooks_repaired/MB12_locked_confirmation.ipynb), **23**, `ee7f996f` | `src/fingerprint.py: record_files`; `results/MB12_integrity.json`; both snapshot JSONs; fresh snapshot/hash sections |

## Literature coverage still missing or useful through the repaired MB13 cutoff

These are **coverage gaps or optional research extensions**, not additional proven bugs. “Missing” means missing in the reviewed repaired notebook set and its in-scope consumed results; it does not imply the whole project or later notebooks never implemented it. A technique appearing in a paper is not automatically necessary for this project. Recommendations below are concrete proposals, not empirical findings about SEL-4L.

### L01 — Freeze and confirm a concrete computational hypothesis

**Basis:** `mi.md`'s progression from localization to mechanism; [causal-scrubbing authors](https://chanlawrence.me/publication/chan-2022-causal/).

**Already implemented:** interventions, individual exact edges, a repaired recursive identity graph, paired equivalence-class resampling, and a historically locked test table.

**Still missing:** an explicit algorithm/graph correspondence with successful independent confirmation of the corrected implementation. Historical defaults and lock statement remain empty; later repaired hypotheses are post-hoc.

**Useful next experiment:** write a falsifiable operation at each claimed node, expected scalar/full-distribution effects and quantitative predictions before collecting a new confirmation population. Include error analysis and the donor/branch/scale choices in the lock. R10–R13 and R17 identify the current boundaries.

### L02 — Joint context-path validation and counterfactual-specific route comparison

**Basis:** [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969).

**Already implemented:** broad candidate localization and exact individual paths; corrected identity graph recovery.

**Still missing:** a jointly evaluated context graph and a graph explaining a specific identity×context interaction under one coherent context change. The current whole-window contrast changes multiple sources simultaneously.

**Useful next experiment:** define a context-variable contrast and compare complete recursive identity/context routes, including a held-out interaction prediction. Canonicalize aliases and use matched graph controls (R09), then report what remains unexplained.

### L03 — DePass-style tracking of input groups through the network

**Basis:** [DePass](https://arxiv.org/abs/2510.18462), literature PDF §4.3.

**Already implemented:** additive input factors and an exact frozen-final-LayerNorm decomposition into residual component terms, with reconstruction tests.

**Still missing:** tracking original player/team/context summands through every intermediate hidden state in a decomposed forward pass. A layer-output decomposition is not an attribution of each output back to the original input group.

**Useful next experiment:** propagate selected input groups with reconstruction checks and compare their attributed writes with exact counterfactual/path effects. DePass's decomposition fixes attention scores and MLP activations for attribution; lossless reconstruction is not itself causal proof or global behavioral additivity. It is a cross-check, not a replacement for interventions.

### L04 — An additive prediction baseline or NATT comparison

**Basis:** [NATT authors' publication](https://mcml.ai/publications/trk%2B24/), literature PDF §4.3.

**Already implemented:** input-level addition, reference predictors and group interventions.

**Still missing:** a trained additive-output comparator whose decomposition can be compared with this full model's context/identity response. Input addition does not make full predictions a sum of independent feature effects.

**Useful next experiment:** train an appropriate additive comparator with aligned features/data and compare accuracy and interaction residuals with path evidence. An additive output restriction changes the model class; agreement would be a cross-check, not evidence the full transformer obeys that restriction.

### L05 — Interaction analysis beyond additive residual arithmetic

**Basis:** [Attention Mechanisms Don't Learn Additive Models](https://arxiv.org/abs/2405.13536), literature PDF §4.4.

**Already implemented:** whole-window identity×context factorial comparison, a LayerNorm-only floor, and repaired same-endpoint group non-additivity.

**Still missing:** identifying which specific context field changes the player intervention's effect, and which computation carries that interaction under coherent endpoints.

**Useful next experiment:** cross player edits with possession/history/time/space edits separately, retaining signed factorial contrasts and reporting a normalization-only control. Rank candidate interactions on discovery, then estimate on independent matches. A nonzero group interaction is evidence under its intervention, not a layer-specific gate.

### L06 — Conditional backup/self-repair tests with a behavioral metric and responding components

**Basis:** [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940), [self-repair study](https://arxiv.org/abs/2402.15390), literature PDF §4.4 and the attention audit's bibliography.

**Already implemented:** 28 head-pair ablations, single-unit ablations and circuit removals.

**Still missing:** mixed head/MLP conditional searches after a specified primary route is removed, with measured growth of the remaining route's effect and inspection of changed downstream computation. Signed Δ interaction alone is insufficient (R05).

**Useful next experiment:** compare each unit's effect intact versus conditional on removing a primary set, using full-output/loss effects as well as arbitration. Test whether later writes compensate or amplify damage. Retraining exclusive models measures a different adaptation from immediate repair in a frozen model; neither automatically proves the other.

### L07 — Test representation overlap/superposition rather than inferring it from low rank

**Basis:** [Superposition Is Not Necessary](https://arxiv.org/abs/2605.05151), literature PDF §4.4 and appendix.

**Already implemented:** swap-response SVD, team-subspace removal, matched random subspaces, complement restoration and steering.

**Still missing:** determining whether semantic identity/context features are packed into overlapping directions or whether the measured low-rank effect simply reflects the chosen contrast and output readout. Causal low-rank response is not equivalent to semantic disentanglement or superposition.

**Useful next experiment:** define feature targets, measure cross-group subspace overlap and selective causal edits with baselines. If learning a sparse dictionary, validate held-out reconstruction and causal selectivity, and compare with simple linear directions. An SAE is an option; it is not required just because language-model literature uses one.

### L08 — Team/player representation effects under controlled architecture manipulations

**Basis:** literature PDF §§2.2–2.3 and its proposed RisingBALLER/Romero comparison; [RisingBALLER](https://arxiv.org/abs/2410.00943), [Romero institutional record](https://biblio.ugent.be/publication/01K885GG3ZAY9PDF9ZSTYQC22A).

**Already implemented:** player/team removal, representation diagnostics and team-subspace subtraction.

**Still missing:** the proposed same-data factorial comparison of summed-token versus separate-slot architectures, each trained with and without explicit team information. Removing team after training cannot establish the effect of omitting it during training.

**Useful next experiment:** align datasets, target, capacity, regularization and split; compare representations with team/role/style labels and performance. The local PDF's proposed architectural explanation for apparently opposite clustering effects is a **hypothesis**; these papers do not establish it for SEL-4L. Avoid promoting that proposed explanation to a fact.

### L09 — Fair identity-only/context-only/full sequence-model comparisons

**Basis:** literature PDF introduction/appendix and Romero comparison.

**Already implemented:** context and identity reference models useful for defining disagreement and habits; trained full sequence model.

**Still missing within the repaired notebooks:** the originally proposed matched three-transformer comparison and its connection to circuit analysis. Current reference predictors differ in architecture, input/history access and inductive bias; they are not interchangeable with full sequence-exclusive models.

**Useful next experiment:** treat it as a separate modeling study, aligning architecture, capacity, tuning and history where scientifically appropriate. Keep reference disagreement as an operational phenomenon rather than a claim about optimal exclusive-model performance. No claim is made here that Track A lacks every related ablation; its training notebooks are outside scope.

### L10 — Indirect identity recovery from supposedly identity-excluded context

**Basis:** literature PDF appendix; [universal football event representation](https://arxiv.org/abs/2606.09327).

**Already implemented:** direct identity-column exclusion and reference invariance checks; context/B5 comparisons.

**Still missing:** a targeted analysis of player/team/role information recoverable from context/history and whether that information affects the measured predictions. Excluding explicit player IDs does not establish statistical independence from identity.

**Useful next experiment:** test context-only identity recovery under match/time/player-aware splits, then intervene on the recoverable information or matching conditions. Include baselines for team/location/role predictability. Avoid calling every statistical correlation label leakage; distinguish prohibited target information from legitimate indirect identity cues.

### L11 — Coherent substitutions and semantic validation of identity directions

**Basis:** [EventGPT](https://arxiv.org/abs/2512.17266), literature PDF §2.5.

**Already implemented:** matched and random player-factor swaps, habit-vector alignment, low-rank directions and a post-hoc coherence table.

**Still missing:** a prespecified eligible substitution design over enough independent matches, derived-field reconciliation and externally validated style/role semantics for directions.

**Useful next experiment:** separate factor-only sensitivity from a coherent modeled substitution, preserving or recomputing team/role/history according to the stated question. Link a direction to measurable behavior/style data and test specificity before naming it. Real-world player impact requires assumptions beyond changing a neural model input.

### L12 — Distinguish earlier sequence information from summaries already in the target token

**Basis:** literature PDF's sequence-model papers and the attention audit's proposed explanation; [LEM](https://arxiv.org/abs/2402.06815).

**Already implemented:** attention census, own/previous-player relation controls, target-versus-all-position ablations and current-token summary interventions.

**Still missing in the reviewed repaired set:** a coherent crossed experiment changing earlier events while fixing target summaries, changing summaries while fixing history, and reconciling both together. The after-MB13 audit reports later experiments, which cannot be imported as cutoff evidence.

**Useful next experiment:** quantify the extra history contribution on conflict and context-uncertain events. Distinguish sequence attention from attention across feature slots. Weak arbitration effects do not show that all temporal information is ignored when the target token already contains history summaries.

### L13 — Robustness across strata, seeds, populations and perturbation designs

**Basis:** literature PDF appendix; [activation-patching guidance](https://arxiv.org/html/2404.15255v1).

**Already implemented:** five-seed component replication, matched controls, coupled/genuine diagnostics, multiple ablation modes and some random controls.

**Still missing:** corrected graph/scrub/steering/context-survey replication across model seeds and prespecified action/phase/context-uncertainty populations. MB11's component replication does not replicate every later corrected experiment. Robustness to donor choices, class granularity and admissible perturbation dose is also not established by one seed-specific averaged run.

**Useful next experiment:** select a small set of conclusions and replicate their full corrected estimands with consistent controls, retaining failure cases. Compare noising/denoising and alternative corruptions deliberately; their behavioral questions differ. Avoid multiplying tests without declaring families and scientific effect thresholds.

### L14 — Confirm category ordering and zone topology before local discrete edits

**Basis:** literature PDF's event-schema discussion and the actual skipped repaired MB13 plan.

**Already implemented:** ordinal/grid utilities with synthetic special-code tests, safe skip behavior and explicit `SPECIAL_PREV_ACTION` handling for resample-other.

**Still missing:** a source-confirmed codebook, `ORDINAL_CODES`, `ZONE_LAYOUT`, and executed true local-step comparisons. All corrected ordinal/grid runs are skipped because `ADJACENCY_CONFIRMED=False`.

**Useful next experiment:** obtain the original preprocessing definitions, verify regular/special code semantics, and execute true neighbors with boundary coverage. The empirical zone relation in R16 is useful evidence to investigate, not authorization to invent the missing preprocessing specification. Adjacent code numbers need not be adjacent places or ordered game states.

### L15 — Data-scale and summation-versus-concatenation sensitivity

**Basis:** literature PDF appendix, RisingBALLER and Seq2Event reading notes.

**Already implemented:** an additive event architecture with causal factor edits; depth/seed variation reported in the project's broader context.

**Still missing within this repaired review:** controlled training-data-size curves and a capacity-matched summed-versus-concatenated factor comparison linked to the observed mechanism. Architecture and data explanations remain hypotheses without that manipulation.

**Useful next experiment:** hold prediction target, features, tuning budget and evaluation population fixed; vary data amount and input combination; compare performance, representation effects and causal routes. This is a costly modeling extension, not a reason to invalidate current input-sensitivity results.

### L16 — Joint spatiotemporal prediction or football action value, if scope expands

**Basis:** literature PDF §§2.4/3.1; [NMSTPP](https://arxiv.org/abs/2302.09276); SPADL/VAEP reference in the local PDF.

**Already implemented:** action-type prediction and action-pair arbitration with spatial/time inputs.

**Still missing:** joint next-event time/location/action modeling and value-based player-impact outcomes. The current Δ is a preference between two reference-selected action classes, not a football value metric.

**Useful next experiment:** only if the scientific question expands, add an appropriately validated joint target or value estimator and reevaluate what identity/context effects mean. These are optional research directions, not omissions required to repair the current notebooks.
