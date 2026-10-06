# Verified review of MB01–MB13

## F01 — MB10's gradient screen disconnects all internal receivers from the output

**Status: confirmed implementation problem; high priority.**

**Location:** `mi_analysis/src/circuits.py`, `receiver_gradients`, lines 178–199; written by MB07 cell 3 and used in MB10 cell 8. `receiver_site('logits')` returns `final_resid`; `mi_analysis/src/hooks.py` line 115 applies that replacement before the final normalization and unembedding.

**Verification:** `receiver_gradients` installs a replacement at every receiver. Each replacement returns `value.detach().requires_grad_(True)`. Detaching the final residual makes the output depend on a new leaf rather than on the computed residual and its upstream receiver leaves. Earlier leaves therefore have no gradient to the output. The function silently substitutes zero when `leaf.grad is None`.

The saved screen confirms the predicted failure: `MB10_edge_screen.csv` contains 4,048 edges. Every one of the **4,012 internal-receiver edges has EAP mean exactly zero**. Of the 36 output-receiver edges, 35 have nonzero means. Joining the screen to `MB10_edges.csv` shows **82 exactly patched tested edges pass BH despite having a zero EAP screen score**. Examples include:

| Contrast | Edge | Exact pooled recovery R |
|---|---|---|
| context | embedding → L0MLP | 0.6116 |
| identity removal | factor_player → L0MLP | 0.5763 |
| identity removal | L0MLP → L1MLP | 0.3140 |
| identity removal | L2MLP → L3MLP | 0.2455 |

The current `circuits.py` matches the source embedded in MB07, so this is not merely a later source edit unrelated to the reviewed run.

**Affected parts:** MB10 first-order screening, optional-edge ranking, completeness analysis and screen calibration. Later circuit narratives that treat the screen as exhaustive evidence inherit the limitation.

**Effect on the run:** internal optional edges cannot be meaningfully ranked by this screen. Many optional internal edges tie at zero. MB06 candidate edges still undergo separate exact patching; those exact measurements are not made zero by this bug. The recorded screen-versus-exact Spearman correlation is about 0.218, but that is not just ordinary approximation error: the graph has been disconnected.

**Effect on interpretation:** “no additional edge survives the completeness check” cannot establish that no relevant non-candidate edge exists. The optional search is compromised. This does not invalidate every exact edge or the MB03 behavioural findings.

**Manual review direction:** distinguish recording gradients of existing receiver tensors from inserting independent detached leaves. Check whether gradients should represent ordinary downstream propagation or an explicitly frozen edge graph. A verification should compare screen derivatives with small exact perturbations at more than the output receiver. Do not silently rewrite the old completeness conclusion after viewing test results.

## F02 — The identity-effect scrub uses identical clean donor outputs in both branches

**Status: confirmed implementation problem relative to the stated preservation test; high priority.**

**Location:** MB11b cell 8, `scrub_measures`; MB12 cell 5, its duplicate `scrub_measures`. The relevant sequence is the ordinary donor forward pass followed by two calls to `delta_under` with the same `replace` dictionary.

**Verification:** donor activations are collected only from the recorded donor inputs. Those exact component outputs are inserted into both the original recipient run and the player-removed recipient run. No second donor cache is collected under player removal. In the hypothesis run every head/MLP at the target is replaced: circuit components use within-class donors; the sole outside component uses unrestricted donors.

Consequently, the paired comparison no longer lets these overwritten target outputs express their normal response to player removal. At the target, the residual updates supplied by those components are identical between branches. The difference mainly reduces to the unoverwritten input difference and normalization, rather than the ordinary mediated player effect. Earlier positions are not changed by a target player edit because the model is causal.

**Affected parts:** MB11b identity-effect hypothesis/floor runs, per-node identity analysis, identity wrong-class comparisons, and MB12 M4. Absolute arbitration under the scrub is a different measurement and should not be conflated with preservation of the paired player effect.

**Effect on the run:** saved MB11b identity recovery is 0.000323 [-0.012830, 0.013398], whereas arbitration recovery is 0.985475 [0.968829, 0.997231]. MB12 M4 is likewise close to zero. The computation structurally blocks most target-mediated differences, explaining why near-zero player-effect recovery cannot cleanly diagnose a bad abstraction. The precise result under a valid paired scrub is unknown without a corrected experiment.

**Effect on interpretation:** do not claim that this result falsifies all proposed player computation, or that high arbitration recovery validates preservation of identity effects. It tests behaviour with fixed donor outputs, a materially narrower intervention.

**Manual review direction:** specify how donor resampling should respond to the identity counterfactual. A paired clean/counterfactual donor construction or another explicitly justified abstraction is needed; merely applying the same donor index is not the same as applying appropriate branch-specific donor activations.

## F03 — The scrub's self-check validates only one branch

**Status: confirmed verification gap.**

**Location:** MB11b cell 8, block following `identity_assignment`.

**Verification:** `clean_self, corrupt_self` are both computed, but `largest` compares only `clean_self` with `clean_intact`. `corrupt_self` is never checked against `corrupt_intact`, and their identity-effect difference is not checked either.

**Affected parts:** the self-scrub assertion presented as evidence that the scrub is implemented correctly.

**Effect on the run:** the saved zero error is true for the original branch and the assertion passes. It provides no assurance for the second branch where F02 matters. This is why the existing check does not catch identical clean donors overriding the corrupted computation.

**Effect on interpretation:** “self-scrub passes” is not evidence that the paired identity experiment passes its identity/no-op check.

**Manual review direction:** inspect all three invariants separately: original logits, counterfactual logits, and original-minus-counterfactual effect.

## F04 — The wrong-class control changes the scoring action and does not preserve joint class sizes

**Status: confirmed implementation problems in the control.**

**Location:** MB11b cell 12, loop over all class keys and the call `averaged(..., shuffled, ...)`.

**Verification:** class keys include `a_ctx`, `team_change` and `player`. Each column is independently permuted in a copy of the full event table. That table is then used not just for donor-class assignment but for the evaluation itself. The scorer reads `a_ctx` from the shuffled table, so the wrong-class run compares a different action against `a_id`. Some shuffled pairs can even have `a_ctx == a_id`. The original/floor arrays used for comparison still use the original action pair.

Separately, independent column permutations preserve each column's marginal counts, not the counts of the joint tuples `(a_ctx, team_change, player)`. The Markdown promises preserved class numbers and class sizes, which this construction does not guarantee.

**Affected parts:** `MB11b_controls.csv` wrong-class row, its paired difference/interval, and claims that wrong classes involve the same amount of resampling.

**Effect on the run:** the recorded wrong-class identity recovery is -0.002601 with a paired hypothesis-minus-control mean 0.001329 and interval [-0.004168, 0.005718]. Those numbers mix changes in class assignment with changes in the metric and joint resampling granularity. They are further affected by F02.

**Effect on interpretation:** neither superiority nor similarity to this control cleanly measures the content of the proposed equivalence classes.

**Manual review direction:** keep scoring fields immutable; permute a separate vector of complete class labels, not their constituent columns independently, when exact class-size preservation is intended.

## F05 — The steering fit is not a successful local quantitative model

**Status: confirmed model-fit and interpretation problem; not evidence that the observed interventions themselves failed.**

**Location:** MB11c cells 3, 12 and 14; `config/steering_prediction.json`; MB12 confirmation table.

**Verification:** the fitted range is beta = [-2, -1, -0.5, 0.5, 1, 2], with no beta = 0 calibration point. The fitted change is:

`predicted change in Delta = 2.898292 * beta - 2.219539`.

Zero steering must give zero change, whereas this fit predicts -2.219539 at zero. This does not make an unconstrained least-squares fit mathematically illegal; it demonstrates that this fitted line is not a good local description around the known no-op point. The intervention magnitude is beta times the residual norm, so beta = 1 is not intrinsically a small perturbation. The largest validation residual is 2.385628. The tolerance is set to twice that residual, 4.771256, rather than requiring the fit to meet a small-error criterion before confirmation.

| beta | Locked prediction | Validation measured change | Test measured change | Test CI |
|---|---|---|---|---|
| -1 | -5.1178 | -5.9278 | -5.8234 | [-6.1815, -5.4364] |
| 0.5 | -0.7704 | 1.5997 | 1.6154 | [1.3314, 1.8355] |
| 1 | 0.6788 | 1.8194 | 1.7998 | [1.2880, 2.1886] |

**Affected parts:** MB11c calibration and S1–S3 numerical claims; MB12's Level-6 language.

**Effect on the run:** all three predictions fail the locked combined rule because their predictions fall outside the test CIs. All three nevertheless satisfy the loose absolute-error tolerance. At beta = 0.5 the prediction already has the wrong sign on validation. The measured validation and test changes are relatively similar: the numerical failure is already present before test transfer.

**Effect on interpretation:** this is failed quantitative calibration, not clean evidence that the direction has no causal effect or that its intervention fails to generalize. It does not support a successful numerical mechanism prediction.

**Manual review direction:** examine a genuinely small neighbourhood of zero, include the no-op point, choose a fit form appropriate to the observed curvature, and establish a fit-quality criterion before locking predictions. These are suggestions, not changes made by this review.

## F06 — Steering magnitude depends on the batch composition

**Status: confirmed intervention-definition issue; numerical contribution to prediction errors unmeasured.**

**Location:** MB11c cell 12, `steer`; MB12 cell 5, `steer_delta`; expression `norm(...).mean()` inside the batch loop.

**Verification:** a single scale is computed as the mean target residual norm of the current batch. Every event in that batch receives the same scale. Regrouping events or changing batch size changes the intervention assigned to an otherwise identical event if the batch-average norm changes. MB11c and MB12 use the same rule, but they use different populations/batches.

**Affected parts:** all identity/team/random-direction steering measurements and locked steering predictions.

**Effect on the run:** beta denotes a batch-normalized intervention, not a fixed dataset-scale or per-event-scale intervention. The saved tables do not isolate how much this contributed to S1–S3 error. Do not attribute all failure to batching; F05 directly demonstrates poor fitting independently.

**Effect on interpretation:** reported steering is conditional on batching and event order. This is not simply ordinary measurement noise.

**Manual review direction:** choose and document a frozen reference scale or an explicit per-event norm rule. Then verify that the intended intervention is invariant to batching.

## F07 — The mechanism description is empty, despite a completed lock

**Status: confirmed documentation/hypothesis gap, already acknowledged by notebook text.**

**Location:** `config/mechanism_hypothesis.json`, all eleven `nodes.*.computes`; `config/locked_predictions.json`, `mechanism_statement`; MB11b cell 6; MB12 cells 4 and 9.

**Verification:** all eleven `computes` strings and the locked mechanism statement are empty. MB11b prints this fact and still runs. MB12 correctly says that empty compute sentences make M4 a resampling check rather than a stated-mechanism test.

**Affected parts:** mechanistic synthesis and claims of confirmation of an articulated abstraction. Configurations still contain executable class keys and directional/numerical predictions.

**Effect on the run:** no effect on forward computations; default classes still determine donors. Fingerprinting protects those defaults, not an explanation of each node's computation.

**Effect on interpretation:** a locked file is not equivalent to a locked scientific explanation. This is not hidden wrongdoing: the notebook warns about it. The missing content cannot be retroactively called a pre-test statement after filling it in now.

**Manual review direction:** write explicit competing hypotheses and what each node is proposed to compute. Treat any new description as post hoc unless it demonstrably existed before the relevant results.

## F08 — The circuit is almost the entire head/MLP network; random draws contain only eight distinct alternatives

**Status: confirmed limitation of circuit specificity, not a sampling-code bug.**

**Location:** MB10 cell 16; `config/circuit.json`; `circuits.random_circuits` lines 141–152; `MB11_random_circuits.csv`; `MB11_replication.csv`.

**Verification:** the component set includes four of four MLPs and seven of eight heads: 11 of 12 components, excluding L3H0 in the primary checkpoint. The generator matches numbers of heads and MLPs. There are only C(8,7) * C(4,4) = **8** distinct sets. The saved 100 draws contain all eight sets, repeated 8–17 times; the selected circuit itself appears 12 times.

**Affected parts:** MB11 sufficiency/completeness claims, M1/M2 and cross-seed circuit distinctiveness.

**Effect on the run:** mean-ablation validation identity faithfulness is 1.0112; this is recovery with nearly the whole network retained. Test F is 0.9875, below the random 95th percentile 1.0582. All five seed rows have `F above own random 95th percentile = False`. Sampling with replacement is legal, but “100 random circuits” must not imply 100 different architectures.

**Effect on interpretation:** strong recovery does not establish a sparse or uniquely explanatory component set. Failure to exceed the random threshold is a valid negative finding under this control. The test is especially narrow: it compares which single head to omit while always retaining every MLP.

**Manual review direction:** report the distinct alternative count and whether smaller mechanisms are actually supported. Do not call high F alone proof of minimality. The existing leave-one-out minimality table is useful but is not a global smallest-subset search.

## F09 — Component faithfulness does not validate the selected edge graph

**Status: methodological limitation with a confirmed mismatch in inference level.**

**Location:** MB10 cells 15–16; MB11 faithfulness code; MB12 cell 5, `conflict_tests` M1/M2 implementation; `circuits.path_patch` lines 128–133.

**Verification:** MB10 produces both a component set and 95 selected edges. Subsequent faithfulness tests retain components and ablate the complement; they do not restrict retained components to only the selected edges. Other pathways among retained components remain active. MB10 can also retain an edge into a receiver that is not included as a component.

**Affected parts:** descriptions of the circuit diagram as the complete causal mechanism. Individual exact edge interventions remain informative.

**Effect on the run:** the reported F values are component-set F values, not edge-pruned graph F values. No numerical effect of enforcing the selected graph has been measured in this review.

**Effect on interpretation:** “this set of components reproduces the effect” does not imply “these selected edges alone reproduce the effect.” The path intervention changes one receiver input, after which its descendants can change normally; do not literally say every other downstream component remains corrupted and unchanged.

**Manual review direction:** distinguish direct sender-to-receiver entry from its downstream propagation, and distinguish isolated edge effects from a joint path-localization hypothesis. This follows the distinction between localized interactions and component membership in [Goldowsky-Dill et al.](https://arxiv.org/abs/2304.05969).

## F10 — The mean-Delta versus empirical-log-odds comparison is not a calibration identity

**Status: confirmed interpretation problem; numerical summaries themselves are descriptive.**

**Location:** MB03 cell 4 definition of data log-odds; cells 8–11 comparison and `MB03_delta.png` diagonal.

**Verification:** the notebook suggests that calibrated probabilities should make mean event-level log odds close to log odds of aggregate realized counts. In general:

`mean(log(p_ctx / p_id)) != log(mean(p_ctx) / mean(p_id))`.

For two events with p_ctx = [0.99, 0.50] and p_id = [0.01, 0.50], an exactly correct probability model has mean log odds 2.297560, while aggregate expected log odds are 1.072121. Heterogeneity alone creates the discrepancy; finite sample counts and the 0.5 smoothing add other differences. Grouping by the same candidate-action pair does not remove within-pair heterogeneity.

**Affected parts:** MB03 statements interpreting model mean Delta 6.305 versus data log-odds 3.788, and points above/below the equality diagonal.

**Effect on the run:** no effect on logits, Delta values or C1. The plot and summaries compare different statistical quantities.

**Effect on interpretation:** the discrepancy alone does not establish overconfidence or a preference for context beyond matching the data. Class-prior adjustment also does not remove all conditional feature or selection effects.

**Manual review direction:** compare sums/means of predicted probabilities with realized frequencies for the selected actions, and examine calibration with compatible aggregation and uncertainty.

## F11 — Conflict selection narrows the scientific question and favours context-decisive events

**Status: methodological limitation, explicitly discussed in the project audit.**

**Location:** MB01 reference definitions and threshold selection; MB01 conflict exploration; MB03 behaviour and tau sensitivity; attention audit section 2.2.

**Verification:** conflicts are defined by disagreement between a current-event identity reference and a current-event context reference, not by matched perturbations of the full model. Validation NLL is approximately 1.8786 for IdFull and 0.9506 for Ctx. Context reference confidence and the realized context action frequency are high on conflicts. MB03 reports realized a_ctx on 87.4% of validation conflicts; the full model chooses it as top action on 93.1%. Increasing tau strengthens context preference and raises realized a_ctx frequency to 97.1% at the 95th-percentile threshold.

**Affected parts:** global identity/context claims and interpretation of weak attention/history effects.

**Effect on the run:** this selection determines every downstream conflict population and its candidates. It does not make C1 false. It makes C1 a statement about events selected partly for strong context-reference predictions.

**Effect on interpretation:** do not turn “context wins on these conflicts” into a general result about every event or a fair contest between equally strong identity/context systems. A reference-defined conflict is not necessarily a deliberative arbitration computation in the model.

**Manual review direction:** keep the qualifier; compare with context-uncertain, non-conflict and phase-specific populations if the broader claim is important.

## F12 — Most conflicts lack dedicated player vectors, and most player conflicts lack matched controls

**Status: confirmed coverage limitation, already disclosed in MB02.**

**Location:** MB02 population/control construction and result cell 25; `artifacts/populations/val_conflicts.parquet`, `matched_controls.parquet`; MB12 test population/results.

**Verification:** there are 10,691 validation conflicts: 1,744 P-player and 8,947 index-0. Only 332 P-player rows have a matched non-conflict control; 1,412 do not. The estimation half has 529 P-player rows and only 81 matched pairs. Test M3 has 362 pairs; M1/M2/M4 have 1,967 P-player rows, while the valid-partner swap tests use 1,948.

**Affected parts:** player-specific claims and specificity comparisons.

**Effect on the run:** statistics use different populations and sample sizes. A zero player ablation on index-0 is expected because its dedicated player vector is already zero; team/role/competition/context can still encode identity-related information.

**Effect on interpretation:** P-all results cannot automatically be called player-style results. Matched-control findings apply to the matchable subset. Lack of a dedicated player vector is not proof of absence of all identity information.

**Manual review direction:** retain denominators and report matchable/unmatchable population differences before broadening specificity claims.

## F13 — A player-only swap and a donor-window swap do not guarantee realistic counterfactual events

**Status: confirmed intervention scope; ecological interpretation is limited.**

**Location:** MB04 input/factorial interventions; `interventions.build_partner_table` line 95 onward; `interventions.write_identity`; MB06/MB10 donor corruption; MB12 C4; MB13 `whole_context_edit`.

**Verification:** partners are matched by modal role, competition where possible, and training-event count. Team compatibility is not required. Only the player ID is changed, while the event's team, history and `same_player_as_prev` remain. The donor-context operation replaces an entire window and writes back only the target player. It therefore also changes donor team/role/competition and previous events; it does not hold the entire identity group fixed.

**Affected parts:** C2/C3 player swap, C4 identity-by-context interaction, context patching and MB13 whole-context comparison.

**Effect on the run:** the measured effects belong to these input edits, including potentially unusual player-team/history combinations. C4 is an interaction between player ID and a broad donor background. The exact amount attributable to incompatibility rather than normal model computation is unmeasured.

**Effect on interpretation:** valid input-level causal sensitivity is not the same as a realistic substitution of a football player. “Context” in these experiments is broader than the model's non-identity groups.

**Manual review direction:** decide whether the goal is factor sensitivity or plausible player substitution. For the latter, choose coherent donors and explicitly reconcile derived identity/history fields. Report relaxed matching cases rather than assuming all swaps are equally natural.

## F14 — MB13's single-field resampling is only marginally supported by the data

**Status: confirmed overstatement of intervention validity.**

**Location:** MB13 cell 4, “On-distribution by construction”; cell 8, `resample_edit`, `shift_edit`; model/data feature definitions.

**Verification:** a value is drawn from the field's conflict-population marginal while every other field remains fixed. This does not preserve joint support. `zone` is supplied alongside x/y; previous action/outcome/location summaries are supplied alongside earlier sequence events; identity-related history summaries remain alongside identities. +/- one SD shifts do not clip or validate the transformed feature domain. In particular, dt is a log-transformed elapsed time, not raw seconds.

**Affected parts:** MB13 realism claims, ranking and causal interpretation of its input effects.

**Effect on the run:** large responses can include reactions to inconsistent or unsupported combinations. Ranking does not separate that from natural sensitivity. The saved results alone do not quantify the fraction of implausible edits or their causal contribution, so this report does not invent such a fraction.

**Effect on interpretation:** say “values drawn from the empirical marginal” rather than “events on-distribution.” Changes of one SD in transformed time are not comparable to a universal one-SD physical-time change.

**Manual review direction:** use joint/conditional donor designs and explicit feature-consistency checks when realistic perturbations matter. Distinguish minimally scoped factor interventions from coherent event counterfactuals.

## F15 — The resampler can retain the original value despite promising otherwise

**Status: confirmed implementation/documentation mismatch.**

**Location:** MB13 cell 4 definition; cell 8, `resample_edit`, expression `np.where(drawn == own, rng.choice(pool, ...), drawn)`.

**Verification:** there is only one redraw when the first draw equals the original. The second draw can equal the original again. For an original category with marginal frequency p, the probability of an unchanged final value is p². Averaging over original rows gives sum(p³). Using the saved P-player population and original feature values yields expected unchanged fractions:

| Field | Expected fraction unchanged by this two-draw scheme |
|---|---|
| same_player_as_prev | 44.44% |
| possession_flag / team_change | 31.67% |
| prev_action | 1.27% |
| zone | 0.063% |

These are analytically expected rates, not measured realized draw counts. The generator is also restarted for each batch at the same repeat-specific seed, which reuses draw-index sequences across batches.

**Affected parts:** MB13 resampled-feature effect sizes and ranking.

**Effect on the run:** some nominal edits are actual no-ops. Concentrated categorical fields receive a smaller effective edit rate, so observed averages mix genuine changes and zero changes. No corrected averages have been computed.

**Effect on interpretation:** the table ranks an intervention including its category prevalence and no-op rate, not equally enforced changes across inputs.

**Manual review direction:** either deliberately allow unchanged draws and report the rate, or sample from the conditional distribution excluding the original value. Keep random generators persistent or make seeds identify events if batch invariance is desired.

## F16 — “Adjacent” edits are applied to unordered categories and can hit special categories

**Status: confirmed semantic problem in the intervention.**

**Location:** MB13 cell 8, `adjacent_edit` and `if top >= 2`; current training source `data.ACTION_NAMES` and `END_ACTION_CODE`.

**Verification:** all categorical inputs with at least three observed codes receive numeric index +/-1. The function actually moves +1 except at the maximum, where it moves -1; it does not test both directions. Prev_action IDs encode action names, not an ordered football variable. ID 14 is OTHER and ID 15 is END/no previous event; incrementing 14 creates a period-boundary symbol regardless of the actual history. A flattened spatial bin also requires topology-aware neighbours: numeric increment is not a general two-dimensional adjacent move. Outcome codes need explicit semantic justification before calling neighbouring integers “smaller.”

**Affected parts:** MB13 previous-action adjacent result (mean absolute Delta change about 3.1361), spatial adjacent comparisons, and depth candidates.

**Effect on the run:** those effects are responses to a particular code remapping, including special-state edits. They are not necessarily local realistic perturbations. Ordered fields such as minute bins are a different case, although period/minute consistency still matters.

**Effect on interpretation:** do not compare “adjacent” and “resampled” as uniformly small versus large interventions across all fields.

**Manual review direction:** define allowed transitions by field: grid neighbours for space, ordinal steps for truly ordered inputs, and named semantic alternatives for actions/outcomes.

## F17 — Setting context categories to zero is not removing their embedding contribution

**Status: confirmed mismatch between operation and label.**

**Location:** MB13 cell 12, `group_removal_edit`; `interventions.remove_feature`; training model source lines 63–68.

**Verification:** only identity features use `padding_idx=0` and fixed zero vectors. Context categorical tables are ordinary learned embeddings. Setting context category indices to zero selects the learned category-0 embedding. Continuous zero removes the bias-free linear contribution, but categorical zero does not remove the categorical vector. Zero can correspond to a real zone, state or action category.

**Affected parts:** MB13 whole-group “removed” table and its comparison with member effects.

**Effect on the run:** the time/state/history/space rows are interventions to a chosen all-zero input-code configuration, not actual removal of their complete factor vectors. For example, the reported history “removal” effect 4.9218 includes the effects of category-zero embeddings.

**Effect on interpretation:** the experiment cannot be read as the model with a context group absent. This concern does not apply in the same way to player/team/role/competition zeroing.

**Manual review direction:** label the intervention literally, or choose an explicitly justified activation-level baseline for a true group ablation.

## F18 — The group-minus-sum statistic cannot establish additivity or interactions

**Status: confirmed mathematical interpretation problem.**

**Location:** MB13 cell 11 description and cell 12 calculation; `MB13_groups.csv`.

**Verification:** the whole-group operation uses category-zero/continuous-zero replacement, while member operations use marginal resampling and +/-SD shifts. They are different endpoint interventions. In addition, the code compares absolute joint change against the sum of absolute individual changes. Even for a perfectly additive function, two individual changes +1 and -1 have joint absolute change 0 but sum of absolute changes 2. The resulting gap -2 is cancellation, not evidence of nonlinear mixing.

**Affected parts:** the group additivity section and any attribution of its gaps to joint processing.

**Effect on the run:** every group has a negative recorded gap: space -0.1661, time -3.1029, state -0.8643, history -4.2388. These discrepancies mix endpoint differences, sign cancellation, no-op rates and possible true nonlinearities. Their separate contributions are not measured.

**Effect on interpretation:** neither “matches the sum = independent reading” nor “differs from sum = nonlinear combination” is warranted from this table.

**Manual review direction:** use the same baseline/donor endpoints for single and joint edits and compute signed factorial interaction contrasts before aggregating magnitudes.

## F19 — “Whole-context upper bound” and “random-vector floor” are not valid bounds

**Status: confirmed interpretation/label problem, not wrong measured values.**

**Location:** MB13 cells 4 and 7; cell 8 `whole_context_edit`, `random_vector_edit`; cell 9 labels.

**Verification:** even a linear function can have a single-input change of magnitude 1 and a combined +1/-1 change of magnitude 0. Joint change need not upper-bound individual changes. A random norm-matched direction can be more or less disruptive than a meaningful change, so it is not a lower bound. F13 further shows that the whole-context swap changes more than current non-identity context fields.

**Affected parts:** interpretation of MB13 values relative to the whole-window swap 6.7990 and group random-vector comparisons (e.g. history 5.1486).

**Effect on the run:** no change to measured numbers; wrong bounding language imposes unsupported structure on them. It is not a contradiction if a single edit exceeds the donor-window edit.

**Manual review direction:** call these reference interventions or perturbation controls; specify their scope and scale without calling them upper/lower bounds.

## F20 — The BH-significant magnitude tests do not establish meaningful input importance

**Status: confirmed limitation of the selected null/test.**

**Location:** MB13 cell 9; `mi_metrics.bootstrap_pvalue` lines 64–79.

**Verification:** the tested quantity is per-event absolute Delta change, which is nonnegative. The one-sided bootstrap asks whether its mean exceeds zero. If every match-bootstrap sample includes positive values, no draw is <=0; the returned value is the numerical floor 1/(2000+1). All **22** single-edit rows have the same adjusted q-value, 0.0004997501.

**Affected parts:** BH significance column and claims that every input is scientifically meaningful because q<0.05.

**Effect on the run:** tiny effects such as y_norm mean absolute change 0.0439 and large effects such as dt 4.4166 receive identical q-values. The magnitude estimates and bootstrap intervals still describe the interventions; the test contributes little discrimination. This report does not claim BH implementation itself is wrong.

**Effect on interpretation:** exceeding an exact-zero magnitude null does not establish relevance beyond a noise/damage control, a practically meaningful threshold or another input's effect.

**Manual review direction:** compare against a meaningful paired control or pre-specified effect threshold and report effect sizes. Do not promote a scientifically useful ranking solely from identical minimum q-values.

## F21 — MB13's all-one residual recovery profile cannot locate when an input is processed

**Status: confirmed limitation of the experimental design and interpretation.**

**Location:** MB13 cell 13 explanatory text and cell 14; `MB13_depth.csv`.

**Verification:** each corruption changes only a target-position input. Earlier positions are therefore unchanged in a causal model. Patching the clean full residual at that target after block 0 restores the entire target state at that depth, including both already transformed and still-available input information. The future computation can then proceed from the clean target state and unchanged earlier states. Full recovery is expected regardless of whether later layers normally transform that information. The same logic applies at deeper blocks.

**Affected parts:** the claim that a flat high profile means the input simply rides the residual stream and a rising profile means later processing.

**Effect on the run:** all 16 saved depth entries are exactly R=1 with CI=[1,1]. This is useful as a restoration check but cannot discriminate passive transport from later transformation for this target-only corruption design.

**Effect on interpretation:** these profiles do not establish early processing, a lack of late computation, or a per-input mechanism.

**Manual review direction:** use more selective component/projection/path interventions or explicit tests of later transformation. The exact residual-restoration result can remain documented as a sanity check.

## F22 — Low-rank output effects do not establish player style, memorization or superposition

**Status: confirmed overstrong inference in the stated framing; useful geometry evidence remains.**

**Location:** MB11c introductory purpose, cells 6 and 9; `MB11c_spectrum.csv`, `MB11c_projected_patching.csv`.

**Verification:** the measured objects are final-residual differences under matched player swaps, not an identified dictionary of semantic style features. Identity top-four directions explain **56.33% of variance**, below the printed criterion of rank<=4 explaining >80%. Random-vector perturbations are even more concentrated at rank four (**63.43%**). In contrast, rank-four identity projections recover **82.99% of the selected Delta effect**; variance explained and output-effect recovery are different criteria.

Projected recovery beats random subspaces and is genuine evidence of a compact output-sensitive part of these swap effects. However, that can reflect the output metric, a shared response geometry or limited action directions. Random perturbation contributions being low-rank demonstrates why low-rankness alone is not identity-specific semantics. Team-subspace removal is analysed descriptively; it is not the basis used for the main projected-patching/steering confirmation.

**Affected parts:** statements equating low rank with style and high rank with per-player memorization; claims that MB11c tests superposition.

**Effect on the run:** the SVD and projected recovery numbers remain valid descriptions under their interventions. No semantic style variable or superposition mechanism has been causally identified by these experiments.

**Manual review direction:** distinguish variance rank, behavioural rank and semantic interpretation. Test identifiable style/role variables against team and random-response alternatives before assigning a style label.

## F23 — Probes and direct logit decomposition answer narrower questions than learned causal use

**Status: interpretation boundary; the notebooks already include important caveats and controls.**

**Location:** MB05 probe tables and introductory evidence label; MB09 decomposition/routes; `src/attribution.py` lines 38 onward.

**Verification:** acting-player probe accuracy at the initial residual is 0.9915, while random-init accuracy is 0.9897 and control-task accuracy 0.9557. At the final block player accuracy is about 0.8902 while random-init accuracy remains about 0.9830. These controls are already implemented. They show that easy decoding need not arise from learned football structure.

MB09 freezes the final LayerNorm scale to construct an exact component-to-logit sum, with largest reconstruction error about 6.68e-6. This checks algebra at that output, not propagation of identity/context groups separately through nonlinear layers. The direct/indirect shares are sign-aligned pooled effect ratios, not percentages of total prediction signal. The ~97% indirect share describes the route of the measured player-removal effect.

**Affected parts:** any prose that promotes probe accuracy to learned player knowledge or promotes component DLA to an additive identity/context mechanism.

**Effect on the run:** no numerical corruption has been found in these basic outputs. This finding is a guard against stronger interpretation, not a demand to discard MB05/MB09.

**Manual review direction:** preserve representation-versus-use and decomposition-versus-causation distinctions. [Leemann et al.](https://arxiv.org/abs/2405.13536) is directly relevant to additive interpretation limits.

## F24 — Test access and later audit conclusions need explicit scope qualifiers

**Status: methodological/provenance limits already partly disclosed.**

**Location:** MB01 test reference scoring/exploration; MB12 limitations cell 22; `attention_audit_and_literature (1).md` sections 1–4; `mi.md` evidence ladder.

**Verification:** MB01 scores and explores validation plus test reference conflicts. MB12's test is therefore not data nobody has ever inspected. Its limitations correctly disclose that training evaluation already used these matches and that this is their sixth use. The MB12 lock still separates its full-model tests from the earlier steps, but a single access-log line only records that particular gate, not all historical access.

The attention audit's strongest MLP-only/head-only and serial-chain percentages come from MB14–MB18. They cannot be counted as already demonstrated by MB13. `mi.md` also presents a pedagogical evidence ladder whose final labels need qualification: adding a method does not guarantee that its mechanism test succeeds.

**Affected parts:** fresh-test/Level-6 claims and the chronology of the attention explanation.

**Effect on the run:** none on the recorded predictions. This changes the strength and scope of claims about independent confirmation and what had been established at the cutoff.

**Manual review direction:** say “locked Track B full-model confirmation on previously used test matches,” where appropriate. Keep later audit findings separate. Do not call every failed confirmation evidence that the full project is invalid.

## Qualifications and corrections to the earlier review

- A failed test is not automatically a defect: C2 and M2 are real negative results under their registered definitions.
- Sampling repeated random circuits is not illegal; the issue is the small distinct alternative space and interpretation of a near-full circuit.
- An unconstrained fit with a nonzero intercept is not intrinsically a coding error. Here it demonstrably fails as the claimed local no-op-centred steering model.
- Mean/resample ablations, noising/denoising patching, probe controls, pairwise head ablations and match-level uncertainty are already implemented. Literature gaps below concern what these do not yet establish.
- “On-distribution,” “upper bound,” “floor,” “group removal,” and “adjacent” are the problematic claims in MB13; many underlying measurements remain useful as precisely named sensitivity experiments.
- Empty hypothesis sentences are already flagged by the notebooks. The problem is unfinished scientific interpretation, not an undisclosed hidden mechanism claim.
- A before/after corrected run was not performed. This report does not invent corrected values or claim to know exactly how a repaired circuit selection would change.

## Precise location index

The index below is generated from the actual notebook cell source and supporting files. It gives searchable anchors for every finding. Literature gaps and suggestions follow the index as the final section of this document.

| Finding | Exact notebook source anchors |
|---|---|
| F01 | [MB10](./mi_analysis/notebooks/MB10_path_patching.ipynb), cell **8**, source line **54**, search `receiver_gradients`<br>[MB07](./mi_analysis/notebooks/MB07_attention_heads.ipynb), cell **3**, source line **179**, search `def receiver_gradients` |
| F02 | [MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **8**, source line **17**, search `donor_cache =`<br>[MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **8**, source line **34**, search `corrupt_out[part]`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **5**, source line **31**, search `corrupt_out[part]` |
| F03 | [MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **8**, source line **44**, search `largest = float` |
| F04 | [MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **12**, source line **11**, search `shuffled[key]`<br>[MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **12**, source line **13**, search `shuffled, DRAWS` |
| F05 | [MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **3**, source line **33**, search `BETAS =`<br>[MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **12**, source line **41**, search `slope, intercept`<br>[MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **14**, source line **7**, search `tolerance =` |
| F06 | [MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **12**, source line **10**, search `scale =`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **5**, source line **64**, search `scale =` |
| F07 | [MB11b](./mi_analysis/notebooks/MB11b_causal_scrubbing.ipynb), cell **6**, source line **57**, search `have no 'computes' sentence`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **9**, source line **7**, search `"mechanism_statement"` |
| F08 | [MB10](./mi_analysis/notebooks/MB10_path_patching.ipynb), cell **16**, source line **12**, search `components |= set`<br>[MB11](./mi_analysis/notebooks/MB11_faithfulness_and_replication.ipynb), cell **0**, source line **11**, search `random_circuits` |
| F09 | [MB10](./mi_analysis/notebooks/MB10_path_patching.ipynb), cell **15**, source line **7**, search `does not add that receiver`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **5**, source line **148**, search `for name, ablated in` |
| F10 | [MB03](./mi_analysis/notebooks/MB03_behaviour_on_conflicts.ipynb), cell **4**, source line **9**, search `If the model`<br>[MB03](./mi_analysis/notebooks/MB03_behaviour_on_conflicts.ipynb), cell **10**, source line **25**, search `label="model = data"` |
| F11 | [MB01](./mi_analysis/notebooks/MB01_conflict_set_construction_and_exploration.ipynb), cell **11**, source line **38**, search `conflict  ⇔`<br>[MB01](./mi_analysis/notebooks/MB01_conflict_set_construction_and_exploration.ipynb), cell **21**, source line **7**, search `Selected:`<br>[MB03](./mi_analysis/notebooks/MB03_behaviour_on_conflicts.ipynb), cell **17**, source line **9**, search `97.1` |
| F12 | [MB02](./mi_analysis/notebooks/MB02_model_verification_and_preregistration.ipynb), cell **25**, source line **10**, search `1,412`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **11**, source line **57**, search `test conflict rows:` |
| F13 | [MB04](./mi_analysis/notebooks/MB04_input_counterfactuals.ipynb), cell **21**, source line **21**, search `with_b =`<br>[MB10](./mi_analysis/notebooks/MB10_path_patching.ipynb), cell **8**, source line **5**, search `write_identity`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **66**, search `def whole_context_edit` |
| F14 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **4**, source line **6**, search `On-distribution`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **53**, search `def shift_edit` |
| F15 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **38**, search `drawn = np.where`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **35**, search `rng = np.random.default_rng(rng_seed` |
| F16 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **44**, search `def adjacent_edit`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **85**, search `if top >= 2` |
| F17 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **12**, source line **1**, search `def group_removal_edit` |
| F18 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **11**, source line **16**, search `A group that`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **12**, source line **21**, search `parts = summary` |
| F19 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **4**, source line **9**, search `upper bound`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **7**, source line **10**, search `the upper bound`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **8**, source line **71**, search `def random_vector_edit` |
| F20 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **9**, source line **18**, search `bootstrap_pvalue`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **9**, source line **22**, search `benjamini_hochberg` |
| F21 | [MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **13**, source line **8**, search `A flat profile`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **14**, source line **11**, search `SITES =`<br>[MB13](./mi_analysis/notebooks/MB13_context_input_survey.ipynb), cell **14**, source line **21**, search `patching.patch_batch` |
| F22 | [MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **0**, source line **8**, search `high-rank`<br>[MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **6**, source line **27**, search `def spectrum`<br>[MB11c](./mi_analysis/notebooks/MB11c_identity_subspace_and_steering.ipynb), cell **9**, source line **39**, search `projected_recovery(estimation, identity_vectors` |
| F23 | [MB05](./mi_analysis/notebooks/MB05_representations.ipynb), cell **0**, source line **9**, search `representational`<br>[MB05](./mi_analysis/notebooks/MB05_representations.ipynb), cell **11**, source line **27**, search `random-init accuracy`<br>[MB09](./mi_analysis/notebooks/MB09_logit_decomposition.ipynb), cell **6**, source line **8**, search `decompose(` |
| F24 | [MB01](./mi_analysis/notebooks/MB01_conflict_set_construction_and_exploration.ipynb), cell **25**, source line **3**, search `for split_name, rows`<br>[MB12](./mi_analysis/notebooks/MB12_locked_confirmation.ipynb), cell **22**, source line **4**, search `sixth use` |

Additional supporting-file anchors (ordinary one-based file lines):

| Finding | File | Line(s) | Search anchor |
|---|---|---|---|
| F01 | [mi_analysis/src/circuits.py](./mi_analysis/src/circuits.py) | 178 | `def receiver_gradients` |
| F01 | [mi_analysis/src/circuits.py](./mi_analysis/src/circuits.py) | 188 | `leaf = value.detach()` |
| F01 | [mi_analysis/src/hooks.py](./mi_analysis/src/hooks.py) | 115 | `final = site("final_resid"` |
| F08 | [mi_analysis/src/circuits.py](./mi_analysis/src/circuits.py) | 141 | `def random_circuits` |
| F09 | [mi_analysis/src/circuits.py](./mi_analysis/src/circuits.py) | 128 | `def path_patch` |
| F13 | [mi_analysis/src/interventions.py](./mi_analysis/src/interventions.py) | 95 | `def build_partner_table` |
| F13 | [mi_analysis/src/interventions.py](./mi_analysis/src/interventions.py) | 45 | `def write_identity` |
| F17 | [model_training_evaluation/src/model.py](./model_training_evaluation/src/model.py) | 64 | `if feature in IDENTITY_FEATURES:` |
| F17 | [model_training_evaluation/src/model.py](./model_training_evaluation/src/model.py) | 68 | `self.embeddings[feature] = nn.Embedding(table_size, config.d_model)` |
| F20 | [mi_analysis/src/mi_metrics.py](./mi_analysis/src/mi_metrics.py) | 64 | `def bootstrap_pvalue` |
| F23 | [mi_analysis/src/mi_metrics.py](./mi_analysis/src/mi_metrics.py) | 92 | `def pooled_recovery` |
| F23 | [mi_analysis/src/attribution.py](./mi_analysis/src/attribution.py) | 38 | `def decompose` |
| F23 | [mi_analysis/src/attribution.py](./mi_analysis/src/attribution.py) | 56 | `def direct_indirect` |
| F07 | [mi_analysis/config/locked_predictions.json](./mi_analysis/config/locked_predictions.json) | 2 | `"mechanism_statement"` |
| F07 | [mi_analysis/config/mechanism_hypothesis.json](./mi_analysis/config/mechanism_hypothesis.json) | 4, 16, 28, 40, 50, 60, 72, 84, 96, 108, 118 | `"computes"` |

## Literature coverage still missing or incomplete by MB13

This section maps your literature review to experiments, not a claim that every cited technique is mandatory. “Suggested application” means a reasonable project experiment inferred from the literature; it is not necessarily a direct instruction in the paper. The relevant cutoff is MB13: later work may cover some items. No training-project absence is inferred from not reviewing its notebooks.

### L01 — Fully specified mechanism hypotheses and appropriate confirmation

**Literature/project basis:** IOI, path patching, causal scrubbing, `mi.md`; [activation-patching guidance](https://arxiv.org/html/2404.15255v1).

**Already done:** discovery/estimation split, input and component interventions, exact edge patches, a locked test table and scrub code.

**Still incomplete:** node computations are blank; paired identity scrub is compromised; edge graph is not jointly validated; steering predictions fail. Numerical tests existing in a notebook do not by themselves establish the proposed mechanism.

**Suggested application:** write two or more explanations that predict different responses, state what each component represents, and test the discriminating predictions with suitable controls. Preserve the current failed results and label new hypotheses as new/post hoc. This is the closest match to your stated goal of understanding the runs.

### L02 — Player/team dependence and embedding clustering

**Literature basis:** your RisingBALLER and Romero discussion; [RisingBALLER](https://arxiv.org/abs/2410.00943).

**Already done:** player/team/role removal comparisons, player/team probes, team-removal response geometry and removal of four team directions for a descriptive spectrum.

**Still incomplete:** direct embedding-clustering analysis separating team from role/style, plus with/without-team training comparisons. Removal effect size does not answer whether player embeddings cluster by team.

**Suggested application:** compare within-team versus within-role similarities with frequency-matched controls and multiple seeds. Include team/player nesting and competition. Treat embedding similarities as descriptive until linked to outputs. The opposite directions discussed in your review need an actual controlled comparison, not an assumed architectural explanation.

### L03 — Architecture dependence of the team/player result

**Literature basis:** your proposed explanation of RisingBALLER versus Romero; Seq2Event concatenation versus this project's summation.

**Already done:** one fixed summed-feature architecture receives detailed MI.

**Missing within this scope:** matched summed-vector versus separate-slot/concatenated designs, with/without team, holding data, objective, parameter budget and training protocol sufficiently comparable.

**Suggested application:** regard this as a separate architecture study. It is not needed to explain existing MB04 values, but it is needed to claim that embedding placement causes the literature difference.

### L04 — Identity leakage through apparently identity-free context

**Literature basis:** Yang's identity/ability-confounding concern as described in your review; Romero information-set comparisons.

**Already done:** direct identity-column exclusion and invariance checks in the context reference; B5-agreement comparisons.

**Still incomplete:** statistical recovery of player/team/role from context alone and whether such recoverable information contributes to prediction. Direct feature exclusion does not imply independence from identity.

**Suggested application:** decode identity from context using match-separated controls, then compare with role/zone/competition baselines. Separate explicit identity signal, context-derived identity proxies and causal use. A successful probe is not itself a leakage mechanism.

### L05 — Self-repair versus serial amplification

**Literature basis:** your Gong/CoAx entry; [Conditional Co-Ablation](https://arxiv.org/abs/2607.01940); Hydra/self-repair papers in the attention audit.

**Already done:** single and pairwise head ablations, MLP ablations and several scope/baseline comparisons.

**Still incomplete:** systematic conditional co-ablation and measurement of downstream changes after a primary pathway is removed, distinguishing compensation, redundancy and amplification.

**Suggested application:** compare a unit's marginal effect with its conditional effect after removing another unit; inspect downstream writes and changes under matched controls. Pairwise damage by itself does not identify a backup circuit. Retrained exclusive models answer a different question from immediate self-repair in the frozen full model.

### L06 — Superposition and semantic feature geometry

**Literature basis:** [Yıldırım's time-series superposition analysis](https://arxiv.org/abs/2605.05151), as discussed in your review.

**Already done:** SVD of final-residual intervention effects and causal projected recovery, with team/random comparisons.

**Missing:** a direct test of overlapping semantic feature directions, dictionary size/feature sparsity sensitivity and causal validation of any dictionary features. Effect-vector rank does not establish superposition or its absence.

**Suggested application:** first identify meaningful candidate features and test whether multiple variables share directions; use dictionaries/SAEs only if they answer that question. This is optional and should not become a checklist exercise.

### L07 — Lossless decomposed forward passes versus direct logit attribution

**Literature basis:** DePass in your review.

**Already done:** exact final-residual component DLA with a fixed LayerNorm scale and causal interventions.

**Missing:** carrying specified input-group attributions through nonlinear layers with the proposed decomposed-forward-pass method.

**Suggested application:** if attribution across identity/context groups is still useful, compare that decomposition with interventions on the same events. Exact reconstruction is an algebraic property, not automatic evidence that the attribution defines a unique causal mechanism. Verify the proposed method's assumptions for this custom model before adopting it.

### L08 — Additive-model agreement with causal interaction evidence

**Literature basis:** NATT and Leemann entries in your review; [Leemann et al.](https://arxiv.org/abs/2405.13536).

**Already done:** MB04 factorial interaction, MB08 MLP freezing and MB09 component decomposition.

**Still incomplete:** a compatible additive baseline or attribution cross-check, and signed same-endpoint interaction tests for individual context factors. MB13's group-minus-sum statistic cannot serve as this check.

**Suggested application:** compare an explicitly additive predictor with a model allowing interactions, or use suitable factorial contrasts in the frozen model. A NATT head would be a model change, so it should not be described as a missing post hoc explanation that can simply be attached without consequences.

### L09 — Which context variable gates the player effect

**Literature/project basis:** identity-by-situation question, path patching, MB13's own limitations.

**Already done:** broad identity-by-donor-window factorial and a single-input sensitivity survey.

**Missing:** an identity swap crossed with one coherent context change, with discovery-stage selection and independent estimation/confirmation.

**Suggested application:** select a small number of plausible candidates such as previous action, possession change or time gap, then test a signed 2x2 interaction with controlled endpoints. A large context main effect does not establish that the variable gates identity.

### L10 — Event-type, phase, player/team and non-conflict generalization

**Literature basis:** soccer evaluation recommendations in your review; attention audit's open checks.

**Already done:** action-pair breakdowns, genuine/coupled strata, B5-agreement groups, tau sensitivity and seed replication.

**Still incomplete:** mechanism-level comparisons across context confidence, match phase, player/team, stoppages/possession changes and non-conflict cases. The matched subset is too selected to substitute for a broad population analysis.

**Suggested application:** measure all-head removal on scored validation events by context-reference confidence; distinguish these NLL effects from changes in conflict Delta. Test whether engineered summaries or earlier events matter on ambiguous rather than strongly context-decisive rows. These are hypotheses, not conclusions established by MB13.

### L11 — Data-scale and out-of-distribution robustness

**Literature basis:** RisingBALLER's small-data limitation and soccer modelling evaluation in your review.

**Already done:** five trained seeds and a match-separated test comparison.

**Missing within reviewed scope:** data-volume curves and new season/competition/team/player evaluations for the same mechanistic claims. Seed agreement does not imply robustness to dataset size or a changed football population.

**Suggested application:** treat learning curves and new-domain testing as a separate follow-up study; distinguish raw prediction performance from stability of the discovered mechanism.

### L12 — Coherent counterfactual player substitutions and player impact

**Literature basis:** EventGPT substitution idea in your review.

**Already done:** matched player-ID substitutions, random player/vector controls and habit-alignment measures.

**Still incomplete:** substitutions coherent with team/role/history and comparisons that separate identity magnitude, semantic alignment and unrealistic input damage.

**Suggested application:** present current swaps as controlled input-factor interventions. If interpreting them as player impact, explicitly define eligible substitutions and coherent derived fields. Real-world causal player impact requires stronger assumptions than neural-network input causality.

### L13 — Temporal information versus engineered current-event summaries

**Literature basis:** Seq2Event and event-history papers in your review/audit.

**Already done:** history-related input edits and attention/MLP interventions.

**Missing by this cutoff:** a clean factorial separating raw earlier events from current-token previous-event summaries and evaluating where each source helps, especially outside conflict events.

**Suggested application:** compare coherent history edits with summary edits and their joint effect, keeping the scored event's remaining identity/context fixed. Do not assume weak attention on conflicts proves history is generally irrelevant.

### L14 — Broader prediction targets and downstream value

**Literature basis:** NMSTPP time/location modelling; VAEP, LEM and EventGPT downstream tasks; Seq2Event spatial analyses described in your review.

**Current scope:** next-action classification and identity/context conflict effects.

**Not implemented in this MI scope:** joint time/location/outcome prediction, action-value/goal-risk interpretation and downstream possession/attack targets.

**Suggested application:** these are optional scope expansions, not flaws in the current classifier. A spatial shot-probability sanity check may help understand the existing model; joint prediction and value claims require additional models/targets.

### L15 — Honest distinction between literature motivation and required implementation

The literature motivates controls, competing hypotheses and cautious interpretation. It does not require every project to implement DePass, SAEs, NATT, CoAx, every architecture or joint spatio-temporal targets. The highest-value review tasks here are to understand F01–F07, preserve the negative confirmation results, and narrow the claims to the actual populations/interventions. Optional techniques should be adopted only when they distinguish a concrete unresolved explanation.
