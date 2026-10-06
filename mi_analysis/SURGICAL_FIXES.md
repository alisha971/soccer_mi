# Surgical fixes and new benchmark rerun

The user selected available weight-decay-0.1 4L/2H/K16 checkpoints, seeds 100–104. This is a new exploratory benchmark analysis. Original notebooks and results stay in mi_analysis_old and the backup. The current Track A source is the baseline; MB02 independently reproduces every benchmark checkpoint's saved validation predictions.

| Review | Treatment |
|---|---|
| R01 | Retain population denominators and conflict-only scope; no all-event generalization from conflict behavior. |
| R02 | Match controls within the partner's match, preventing crossed match dependence; restrict context donors to the recipient's half. |
| R03 | One-cluster summaries are descriptive and have no inferential CI/p-value. |
| R04 | New plan explicitly retains player regardless of C2; report failures without post-result fallback. |
| R05 | Signed head-pair non-additivity has no universal redundancy/compensation sign. |
| R06–R07 | Optional edge rank selection on discovery; exact test on estimation; no finite-bootstrap significance gate for promotion; completeness conditional on budget. |
| R08 | Disclose density/non-uniqueness and identity-only joint graph coverage. |
| R09 | Canonicalize atom-to-receiver connections before matching random graph size. |
| R10 | Track actual donors over all draws; compare right/wrong controls on the same intersection of non-self rows. |
| R11 | Preserve wrong-class branch rule; label team-keyed comparison as a branch-and-granularity sensitivity test. |
| R12 | Hold other donors and branch rules fixed; compute paired drop CIs and donor-draw SD separately. |
| R13 | Rename chain hypotheses as output-key hypotheses; no serial computation claim. |
| R14 | Fixed steering scale, origin-respecting diagnostic; discovery-fit corrected models and additional frozen unseen doses on estimation. |
| R15 | Bound continuous edits; report clipping, no-ops and realized doses. |
| R16 | Factor sensitivity and joint-donor non-additivity retain their literal scope; spatial joint donors preserve observed tuples. No real-world ranking claim. |
| R17 | MB12 reruns known-test measurements as retrospective diagnostics using paired scrubbing and distinct component alternatives, audits provenance, and explicitly withholds fresh confirmation. |
| R18 | Explicit MI_ANALYSIS_DIR staging path; relative artifact manifests; promote only completed notebooks and dependencies. |
| R19 | New model/source/config hashes, archived source versions and per-notebook dependency records; no historical reproduction claim. |

MB01 verifies reused frozen reference artifacts instead of retraining reference predictors. MB05/06/08/09 are recovered upstream notebooks rerun with the corrected shared sources, because their outputs depend on the newly selected benchmark model. No old full-model result is imported into the new analysis. Codebook-dependent ordinal/spatial adjacency remains unexecuted pending source-confirmed semantics. Literature extensions requiring new training or new data are outside these surgical fixes.

Additional scope note for the recovered upstream MB05 notebook: `src/probing.py::control_labels` randomly remaps the true target class itself to another class. This is a target-class remapping diagnostic, not an independent control task; it can retain information about the original target. Its legacy “selectivity” column is only accuracy minus this remapping diagnostic and must not be interpreted as Hewitt–Liang selectivity. No causal-use claim follows from any probe accuracy. The existing probe computation is retained for upstream continuity; this limitation is disclosed rather than silently changing the probe experiment.

## Where to inspect the repairs

| Area | Current implementation / evidence |
|---|---|
| Match dependence, donors, identity plan | MB02 cells `d9f2ce6e`, `ceb73687`, `461c704c`; `config/analysis_plan.json`; populations parquet tables |
| Single-match uncertainty | `src/mi_metrics.py::summarise`, `bootstrap_pvalue`, `pooled_recovery`; `results/MB04_C2_by_coherence_corrected.csv` |
| Discovery edge selection and budget | MB10 cells `P7p0Zs2ciOS7`, `3QNwaOM2iOS_`, `r1ada1cc80`, `rff1cfd470`; corrected screen and exact-edge tables |
| Actual graph connection size | `src/graph.py::canonical_edges`; MB11 cell `rcdc269d00`; corrected graph faithfulness table |
| Actual donors and paired node perturbations | `src/scrub.py::averaged_scrub`; MB11b corrected R3–R5 sections and corrected result tables |
| Steering | MB11c fixed scale and corrected R2–R3; `results/MB11c_unseen_dose_validation.csv` |
| Continuous edit bounds / dose | `src/context_edits.py::bounded_shift`; MB13 corrected section; `results/MB13_shift_coverage.csv` |
| Retrospective test status and code provenance | MB12; `config/locked_predictions.json`; `manifests/plan_amendments.jsonl`; `results/MB12_retrospective_diagnostics.csv` |
| Packaging and source provenance | `src/fingerprint.py`; `manifests/source_change_log.jsonl`, per-notebook dependency records and external model-input hashes |
| Regression evidence | MB07R and its two regression CSVs: 20 existing plus four added repair checks |

The detailed finding definitions and historical proofs remain in the repository-root `NOTEBOOK_REVIEW_REPAIRED_MB01_MB13.md`. This file records their treatment in the new benchmark rather than rewriting those historical observations.

User decision during the rerun: keep category-adjacency edits disabled. This is recorded in `config/adjacency_policy.json`; the rest of MB13 remains in scope.

MB07's ablation cell saves each completed head with hashes for the benchmark checkpoint, config, repaired circuit source and all other notebook code. It resumes only matching checkpoints. Batch size 512 changes evaluation batching for CPU throughput; all events, interventions and resampling draws stay the same.
