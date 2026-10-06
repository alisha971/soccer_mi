# MI notebook review, surgical repairs and benchmark rerun

Last updated: 2026-10-06. Project root: `/home/aayush/code/soccer_mi`.

**Current state: stopped at the user's request. Seven of the 16 notebooks in the execution sequence completed and were promoted to `mi_analysis/`. MB07 stopped at cell 23/31; MB08–MB13 have not executed in this benchmark rerun.** This document records the work and the information needed to continue. It does not authorize restarting computation.

## 1. Requested work and how the scope evolved

The original task was to review MI notebooks through notebook 13 against the local project descriptions and literature review, identify verified problems with proofs, explain their locations and effects on runs and interpretations, and list literature recommendations not implemented. The user subsequently requested the same kind of review for the repaired notebooks only. The resulting review is [NOTEBOOK_REVIEW_REPAIRED_MB01_MB13.md](NOTEBOOK_REVIEW_REPAIRED_MB01_MB13.md); the earlier review is [NOTEBOOK_REVIEW_MB01_MB13.md](NOTEBOOK_REVIEW_MB01_MB13.md).

The user then asked which issues were critical, whether they affected results, and what was required to fix and rerun the analysis. A supplied backup and benchmark checkpoints made a new run possible. The user authorized surgical repairs, execution, and incremental addition of successfully completed notebooks and their dependencies to `mi_analysis/`.

The historical requested SEL-4L weight-decay-0.01 checkpoints were unavailable in the checkout. The available benchmark uses weight decay 0.1. The user explicitly selected: **“Use the available 0.1 checkpoints as a clearly labelled new benchmark analysis.”** The user also explicitly selected: **“Keep adjacency edits disabled for this rerun.”** Missing category-ordering and zone-topology source definitions must not be replaced by assumed semantics.

After execution had resumed, the user requested all background work be stopped and a short progress note written. The runner and kernel were terminated, and [mi_analysis/PAUSE_HANDOFF.md](mi_analysis/PAUSE_HANDOFF.md) was created. This longer root document supplements that note. No new execution was started to write it.

## 2. Directory and evidence map

| Location | Role |
|---|---|
| `mi_analysis_old/` | Historical analysis and repaired notebook material; retained for comparison. The detailed review links notebooks under `mi_analysis_old/notebooks_repaired/`. |
| `mi_analysis_backup_2026-10-04/` | Supplied backup; source of missing upstream notebooks and dependencies. |
| `.mi_analysis_rerun/` | Working staging tree, containing the full repaired notebook sequence, shared sources, partial execution outputs and logs. The leading dot makes it hidden in ordinary directory listings. |
| `mi_analysis/` | Published successful notebooks and regenerated dependencies. Pending notebook files are not promoted merely because they were edited. |
| `model_training_evaluation/` | Existing training/evaluation code, prepared data, checkpoints and benchmark configuration used as external dependencies. |
| `NOTEBOOK_REVIEW_REPAIRED_MB01_MB13.md` | Detailed R01–R19 findings, historical evidence, precise cell/file locations, and L01–L16 literature gaps. Its saved numerical results describe the reviewed historical material, not the new benchmark results. |
| `mi_analysis/SURGICAL_FIXES.md` | Repair treatments and implementation locations. Some listed output paths refer to pending notebooks and do not yet exist as completed benchmark evidence. |
| `mi_analysis/RUNNING.md` | Environment, execution order, packaging and scope instructions. |
| `scripts/prepare_mi_rerun.py` | Initial staging/recovery setup. Do not rerun blindly over existing outputs. |
| `scripts/run_mi_notebooks.py` | Executes staged notebooks and promotes successful notebooks/dependencies. |
| `scripts/finalize_mi_rerun.py` | Final audit/report generator, intended to run only after every required notebook completes. It has not completed this audit. |

Paths in this document are relative to the project root unless explicitly absolute. Large model/data inputs remain external dependencies; they were not duplicated into the published MI package.

## 3. Benchmark, experiment plan and environment

- Analysis identifier: `benchmark_4L_2H_K16_wd0.1_2026-10-05`.
- Full-model architecture: 4 layers, 2 heads, embedding width 128, feed-forward width 512, context length K16; available weight-decay-0.1 benchmark.
- Full-model seeds: 100, 101, 102, 103 and 104. Primary mechanistic-analysis seed: **103**. Checking predictions in five seeds does not establish that an equivalent circuit was discovered independently in every seed.
- Benchmark root: `model_training_evaluation/benchmarks/track_a_4layer/`. Checkpoints are in its `runs/`; the selected configuration is `config/frozen_config.json` beneath that benchmark root.
- Exact run paths are in `mi_analysis/config/benchmark_final_runs.csv`, restricted-model paths in `benchmark_restricted_runs.csv`, and external input hashes in `mi_analysis/manifests/external_model_inputs.json`.
- Frozen reference artifacts and conflict definitions are reused and hash-verified. MB01 was replaced for this rerun by dependency verification; it does **not** retrain the reference predictors.
- Player remains the declared identity variable irrespective of the C2 diagnostic. There is no outcome-dependent fallback to another identity variable.
- Matched non-conflict controls now come from the same match as their conflict partner, without event reuse; context donors are restricted to the recipient's discovery/estimation half.
- Full mode uses 2,000 bootstrap draws, 5 intervention resampling draws and 5 scrubbing draws, as recorded in the plan. Smoke mode changes scientific sample sizes and is not an equivalent completed run.
- Python environment used: `/home/aayush/code/ML_DL/.venv/bin/python` (Python 3.12). CPU execution, four numerical-library threads; CUDA was unavailable for these runs. Packages and versions are recorded in `mi_analysis/config/execution_environment.json`, with dependencies in `requirements-rerun.txt`.
- Important package versions include Torch 2.10.0, NumPy 2.4.6, pandas 3.0.0, SciPy 1.17.0, scikit-learn 1.8.0, PyArrow 23.0.1 and nbclient 0.10.4.

The current Track A source is the execution baseline. A historical `data.py` hash mismatch was disclosed; current source hashes and source snapshots are recorded. This run cannot establish immutable historical reproduction. Changing the checkpoint family also means a difference from historical results cannot be attributed solely to a code fix.

### Previously accessed test split

Historical test outcomes were already known. No rerun can turn those outcomes into fresh blind confirmation. MB12's staged repair labels any renewed known-test measurements as **retrospective diagnostics**, records provenance and amendments, and withholds fresh confirmation claims. At the stop point MB12 has not run, so the currently published `analysis_plan.json` still contains its earlier validation-only test policy. The staged MB12 implementation is intended to record an explicit policy amendment before any retrospective test access. Do not treat that future amendment or the intended audit as already executed.

## 4. Findings and repair treatments

The review has exact historical proofs and affected notebook cells. The following table summarizes treatment; shared-source regression checks and the successful early notebooks provide partial execution evidence. Repairs in later notebooks still require their actual runs and final audit.

| Finding | Problem and possible impact | Treatment in this benchmark |
|---|---|---|
| R01 | Conflict-selected populations and aggregated calibration limit generalization; conclusions cannot cover all events automatically. | Retain explicit population denominators, action-pair calibration diagnostics and conflict-only scope. |
| R02 | Controls could cross the match split and introduce dependence between discovery and estimation. | Match within the partner's match; restrict context donors to the recipient half. MB02 generated the new populations. |
| R03 | A coherent-swap estimation subgroup contains only one independent match; apparent narrow uncertainty is unsupported. | One-match results are descriptive; no inferential CI or p-value. Shared regression passed and MB04 output reflects this. |
| R04 | Actual identity analysis did not follow the recorded fallback rule. | Declare player retention in the new plan and report C2 outcomes without changing the variable after observing them. |
| R05 | Signed pair non-additivity was assigned a universal redundancy/compensation meaning it does not have. | Correct the interpretation; retain the signed arithmetic and avoid universal sign claims. MB07 execution is incomplete. |
| R06–R07 | Edge search/completeness excluded candidates and reused estimation data for selection and assessment. | Discovery ranking for optional candidates; exact estimation testing; remove finite-bootstrap significance as a promotion gate; qualify completeness by testing budget. Staged MB10 also uses repaired receiver gradients in its initial screen. |
| R08 | Selected components can be nearly the whole model; identity graph evidence does not validate a context graph. | Disclose density, non-uniqueness and identity-only graph scope. This limitation is not eliminated by wording changes. |
| R09 | Named edges and aliases produce different actual connection counts in supposedly size-matched graph controls. | Canonicalize connections before matching controls. Added regression passed; MB11 application is pending. |
| R10 | Wrong-class controls were filtered using another condition's donor mask, mislabelling self-donations as non-self. | Track actual donors across every node and draw; compare conditions on their common non-self intersection. Added regression passed; MB11b run pending. |
| R11 | A class-key control also changed the branch rule, confounding its interpretation. | Preserve the wrong-class branch rule; label team-keyed comparisons as branch/granularity sensitivity. MB11b pending. |
| R12 | Per-node perturbations changed other donors too, and point drops lacked paired uncertainty. | Hold other donors and branch rules fixed; compute paired drop CIs separately from donor-draw SD. Added regression passed; MB11b pending. |
| R13 | Output resampling/equivalence classes were described as a demonstrated serial computation. | Use output-key hypotheses and limit claims to what scrubbing actually tests. No serial-algorithm claim. |
| R14 | Same-grid steering transfer was presented as unseen-dose mechanistic prediction. | Fixed discovery reference scale and origin-respecting diagnostics; discovery-fit models plus additional frozen doses on estimation. Those doses remain diagnostic, not independent confirmation. MB11c pending. |
| R15 | Continuous ±1 SD edits could exceed physical feature support. | Bound shifts and report clipping, no-ops and realized doses. Added regression passed; MB13 pending. |
| R16 | Marginal factor edits, inconsistent joint representations and unequal doses can be overread as realistic causal importance. | Preserve observed spatial tuples for joint donors; report literal factor sensitivity, dose coverage and same-endpoint non-additivity. No real-world importance ranking. |
| R17 | Corrected validation cannot repair an already exposed held-out confirmation. | MB12 is retrospective, with paired diagnostics, distinct component alternatives and provenance audit; no new blind confirmation. Pending execution. |
| R18 | Notebook setup could resolve the wrong analysis directory. | Explicit staging override via `MI_ANALYSIS_DIR`, repository-root resolution and successful-only promotion. MB01/MB12 root handling was amended; dependencies/provenance record amendments. |
| R19 | Existing checks did not establish complete immutable history. | New model/config/source hashes, archived source versions and per-notebook dependency records; disclose historical differences. Final package audit remains pending. |

Recovered upstream MB05, MB06, MB08 and MB09 are rerun with corrected shared sources and the new model because downstream outputs depend on them. No old full-model measurement is imported as a fresh benchmark result.

An additional MB05 limitation was documented: `src/probing.py::control_labels` remaps the true target class to another class. It is not an independent control task and may retain target information. Its legacy “selectivity” column is accuracy minus this remapping diagnostic, not Hewitt–Liang control selectivity. Probe accuracy alone does not establish causal use. The upstream computation was retained and the interpretation corrected. MB05 also emitted convergence warnings with its existing 300-iteration limit; completed execution does not guarantee every probe fit converged.

## 5. Actual execution progress

The required execution sequence contains **16 notebooks** because it includes MB07R and MB11b/MB11c in addition to MB01–MB13. Cell counts below are notebook cells, including markdown, as reported by the runner.

| Notebook | State at stop | Successful runtime (seconds) | Evidence / role |
|---|---|---:|---|
| MB01 | Completed and promoted | 2.26 | Frozen reference dependency verification. |
| MB02 | Completed and promoted | 47.06 | All five checkpoint prediction reproductions; instrumentation and population generation. |
| MB07R | Completed and promoted | 13.22 | 20 existing regression checks plus 4 added repair checks, all passed. |
| MB03 | Completed and promoted | 50.78 | Conflict behavior and corrected calibration diagnostics. |
| MB04 | Completed and promoted | 149.85 | Input counterfactuals, denominators and coherence diagnostics. |
| MB05 | Completed and promoted | 336.89 | Representation/probe analysis; convergence and selectivity limitations disclosed. |
| MB06 | Completed and promoted | 577.09 | Full activation-patching sweep. |
| MB07 | Interrupted; machine status `failed` | — | Last reached cell 23/31, all-event head ablation. Not promoted. |
| MB08 | Not executed in this benchmark | — | Staged MLP analysis. |
| MB09 | Not executed in this benchmark | — | Staged logit decomposition. |
| MB10 | Not executed in this benchmark | — | Staged path patching and corrected edge screening. |
| MB11 | Not executed in this benchmark | — | Staged component/edge graph faithfulness and replication. |
| MB11b | Not executed in this benchmark | — | Staged causal scrubbing. |
| MB11c | Not executed in this benchmark | — | Staged identity subspace/steering. |
| MB12 | Not executed in this benchmark | — | Staged retrospective diagnostics and provenance. |
| MB13 | Not executed in this benchmark | — | Staged context survey with category adjacency disabled. |

The live status at shutdown is [mi_analysis/logs/rerun_status.json](mi_analysis/logs/rerun_status.json). MB07's `failed` state records interruption, not a demonstrated numerical or scientific failure; its last attempt has an empty error string and cleanup traceback in the log.

### Verified numerical evidence from completed notebooks

MB02 reproduced the same saved validation events for all five models and passed a maximum absolute logit-difference tolerance of 1e-5:

| Seed | Maximum absolute logit difference | Recomputed validation NLL |
|---|---:|---:|
| 100 | 6.4373e-6 | 0.90694314 |
| 101 | 5.7220e-6 | 0.90211815 |
| 102 | 7.6294e-6 | 0.90412605 |
| 103 | 6.3777e-6 | 0.90377575 |
| 104 | 7.1526e-6 | 0.89842623 |

Evidence: `mi_analysis/results/MB02_verification.csv` and `MB02_instrumentation_tests.csv`. These establish checkpoint inference reproduction and the tested instrumentation properties, not the validity of every downstream inference.

MB07R evidence is split between `MB07R_regression_tests.csv` (20 existing checks) and `MB07R_surgical_regression_tests.csv` (4 added checks). Added checks cover canonical graph control size, donor masks/coupled node controls, bounded continuous shifts, and one-cluster uncertainty. All 24 passed. These checks do not replace full execution of later notebooks.

MB04's population denominator table reports:

| Population | Total | Estimation | Discovery |
|---|---:|---:|---:|
| Dedicated-player conflicts | 1,744 | 529 | 1,215 |
| Conflicts with a swap partner (C2/C3 eligible) | 1,717 | 529 | 1,188 |
| Index-0 events without a dedicated player vector | 8,947 | 4,651 | 4,296 |
| Matched non-conflict controls | 252 | 65 | 187 |

For the saved C2 quantity `mean |dd|(I2) - |dd|(I4)`, the post-hoc coherent estimation subgroup has 16 rows from **one match**, mean −0.40670549 and no inferential interval. The incoherent estimation subgroup has 513 rows from 12 matches, mean −0.07831325 and match-bootstrap interval [−0.16165485, 0.03129993]. Discovery coherent: 285 rows/5 matches, mean −0.13213126, interval [−0.16084589, −0.07149220]. Discovery incoherent: 903 rows/15 matches, mean −0.18028555, interval [−0.25656069, −0.10970893]. These are descriptive counterfactual/coherence results within their stated populations, not evidence of a complete mechanism.

Evidence: `mi_analysis/results/MB04_denominators_corrected.csv` and `MB04_C2_by_coherence_corrected.csv`. Later graph, scrubbing, steering and context-survey findings must wait for their runs; historical values in the review cannot fill that gap.

## 6. Runtime interruptions and exact stop point

1. The first MB05 attempt ran for 1,485.68 seconds before interruption to address severe numerical-library thread contention. With thread counts limited to four, the full rerun completed in 336.89 seconds.
2. The first MB07 attempt ran for 9,534.95 seconds (about 2 hours 39 minutes), reaching cell 23 without completing the cell. A server restart interrupted it. There were no saved per-head results from that attempt.
3. MB07 was amended to evaluate its ablation panel with batch size 512 and to checkpoint each completed head. All scored events, interventions and draw counts remain in scope. The cell covers eight individual heads under target/all-position scopes, mean and five-draw resampling ablations, followed by 28 pair ablations.
4. The next MB07 attempt began at `2026-10-05T21:02:51Z` and was stopped at the user's request after 491.97 seconds. Its last recorded progress was `Resuming after 0 of 8 heads; batch=512`. No head completed before shutdown, and `.mi_analysis_rerun/results/MB07_head_checkpoints/` is empty.
5. Runner PID 344217 and kernel PID 344291 were sent TERM. A process inspection then found no remaining `run_mi_notebooks.py` or `ipykernel_launcher` processes. This verification occurred at shutdown; it is not a claim about unrelated user processes.

The latest log is `.mi_analysis_rerun/logs/execution_MB07_MB08_MB09_MB10_MB11_MB11b_MB11c_MB12_MB13.log`. The staging notebook retains completed earlier cells and interrupted-cell output; saved partial outputs must not be treated as a completed MB07 result.

### Checkpoint behavior and remaining runtime risk

Each completed head would produce parquet data and JSON metadata. The resume contract includes analysis identifier, primary model checkpoint hash, MI config hash, circuit source hash, other notebook code hash, batch size, draw count, NLL row count and head list. RNG state is restored so resumed resampling continues the same stream. Data and metadata are published through temporary files; an incomplete pair is recomputed. Mismatching contracts are refused instead of mixing analyses.

The resumed code path has not yet been exercised with a completed saved head. The 28 pair ablations occur after the head checkpoints and do not have equivalent per-pair checkpoints in the current cell. A later interruption during pairs may require redoing that pair phase. Increasing batch size was an attempted throughput improvement; no completed MB07 runtime demonstrates its eventual speed or peak-memory suitability. CPU execution remains expensive. Do not reduce samples/draws silently to make it finish.

## 7. Provenance and publication rules

- `config/analysis_plan.json` defines the benchmark and declared scientific scope; `config/adjacency_policy.json` records the disabled-adjacency decision.
- `manifests/external_model_inputs.json` records external checkpoint/config paths and hashes.
- `manifests/track_a_snapshot.json`, `pre_repair_snapshot.json` and `run_input_snapshot.json` retain source/input baselines.
- `manifests/source_change_log.jsonl` and `artifacts/source_versions/` preserve intentional source amendments.
- `manifests/MB*_completed_dependencies.json` records the notebook hash and dependency state at that notebook's completion. Later notebooks may change shared outputs/configuration, so an earlier dependency record is a historical completion snapshot, not necessarily a hash inventory of the latest package.
- `manifests/sha_manifest.json` records a generated hash inventory. A whole-package final audit remains pending.
- The runner saves the staging notebook after each executed cell, updates staging/published status, and copies the notebook into `mi_analysis/notebooks/` only on full success. Shared directories are copied during each successful promotion.
- Final audit/report code is an implementation awaiting execution. Its existence does not prove that all inputs, notebook outputs or final locks have been verified in the completed package.

The finalizer requires all 16 notebooks to be completed; checks notebook error outputs and completion hashes, the final dependency snapshot, external inputs, Track A snapshot, MB12 code lock, regression results and control-match consistency; then writes `mi_analysis/RERUN_REPORT.md`. That report has not been generated for this unfinished run.

## 8. Remaining work and resume instructions

**The stop request remains in effect. Resume only after the user asks to continue.** Do not rerun preparation/fix initialization over the staged execution tree without checking its overwrite guards and existing results.

On resumption, inspect current status/logs and staged sources first. Confirm the expected Python environment, external model/data paths and hashes are still available. Then the existing remaining queue is:

```sh
/home/aayush/code/ML_DL/.venv/bin/python scripts/run_mi_notebooks.py MB07 MB08 MB09 MB10 MB11 MB11b MB11c MB12 MB13
```

Run from the project root. The runner sets `SOCCER_MI_ROOT`, `MI_ANALYSIS_DIR=.mi_analysis_rerun`, full mode, thread limits, staging Jupyter path and the plotting backend itself. Its configured kernel is `soccer-mi-rerun`; preserve or verify the staging kernel setup. Local Jupyter execution may need the same sandbox approval for kernel sockets as the previous runs.

MB07 will restart notebook setup and earlier computations; its head checkpoints skip completed heads only when valid files exist. There are currently none, so it will begin the expensive head phase again. Monitor progress and memory rather than interpreting a quiet long cell as a failure. If a real execution error occurs, preserve its traceback, repair the smallest verified problem, record provenance and rerun the failed notebook and remaining dependencies.

After all pending notebooks succeed, run:

```sh
/home/aayush/code/ML_DL/.venv/bin/python scripts/finalize_mi_rerun.py
```

Resolve actual audit failures before reporting completion. Update this handoff and the short pause note so stale stop-state text is clearly historical. Report numerical changes with model-family and population caveats. Keep category adjacency disabled unless the user changes the decision and source definitions become available.

## 9. Scientific limitations and further work

The surgical repairs address verified implementation defects and misleading interpretations. Several limitations require new experiments or data rather than a local patch:

- Conflict-only evidence does not automatically generalize to all events; small control populations and one-match subgroups limit inference.
- Component density, graph non-uniqueness and untested context routes remain relevant even if identity graph recovery is high.
- A graph assessed after selecting edges on its estimation effects is exploratory; it does not receive independent validation merely by recomputing recovery.
- Scrubbing establishes properties of the specified donor/output-key interventions, not a serial algorithm. Empty common non-self subsets must remain unavailable rather than receiving fabricated estimates.
- Additional estimation-half steering doses are useful diagnostics, not a fresh independent confirmation population.
- Previously exposed test outcomes cannot provide blind confirmation; a new untouched population is required for that claim.
- Bounded edits reduce invalid-domain interventions, but clipping/no-ops and unequal realized doses still constrain interpretation. Marginal edits and joint-donor interactions do not rank realistic causal importance.
- Category order, minute/score/possession-length code semantics and zone topology remain source-definition gaps; no guessed adjacency should be executed.
- Probe diagnostics retain the remapped-label/selectivity limitation and recorded convergence warnings.
- Current full-model verification covers five seeds, while the main mechanistic discovery uses primary seed 103.

The detailed review's literature section lists 16 extensions that were not all implemented by these surgical repairs:

1. Freeze and independently confirm a concrete computational hypothesis.
2. Validate joint context routes and compare counterfactual-specific recursive paths.
3. Track input-group effects through the network using DePass-style analysis.
4. Compare an additive prediction baseline or NATT approach.
5. Study interactions beyond additive residual arithmetic.
6. Test conditional backup/self-repair with behavioral effects and responding components.
7. Test representation overlap/superposition directly rather than inferring it from low rank.
8. Study team/player representation with controlled architecture manipulations.
9. Make fair identity-only/context-only/full sequence-model comparisons.
10. Measure indirect identity recovery from ostensibly identity-excluded context.
11. Validate coherent substitutions and the semantics of identity directions.
12. Distinguish earlier sequence information from target-token summaries.
13. Extend robustness across strata, seeds, populations and perturbation designs.
14. Confirm ordering and spatial topology before local discrete edits.
15. Test data-scale and summation-versus-concatenation sensitivity.
16. Consider joint spatiotemporal prediction or football action value if the scientific scope expands.

See L01–L16 in the detailed repaired-notebook review for source attribution and proposed experiments. These are future research directions, not outstanding promises to implement during this surgical rerun. No new internet verification or scientific experiment was performed while writing this handoff.
