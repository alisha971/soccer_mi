# Completed wd=0.1 benchmark rerun

All 16 notebooks in the execution sequence completed successfully and were published with their dependencies. This is a **new exploratory 4L/2H/K16, weight-decay-0.1 benchmark**, using seeds 100–104 and primary seed 103. It does not reproduce historical weight-decay-0.01 SEL-4L runs. Historical notebooks and results were preserved.

MB01 verifies reused frozen reference/conflict artifacts; reference predictors were not retrained. Missing upstream MB05/06/08/09 notebooks were recovered from the supplied backup and rerun so downstream inputs refer to the new benchmark. All other model-dependent measurements were regenerated.

Prediction reproduction and behavioral seed comparisons cover seeds 100–104. The main mechanistic analysis uses primary seed 103; this is not a claim that equivalent circuits were independently discovered in every seed.

See [SURGICAL_FIXES.md](SURGICAL_FIXES.md) for R01–R19 treatments, [RUNNING.md](RUNNING.md) for execution instructions, and [config/analysis_plan.json](config/analysis_plan.json) for scope. Completed notebook dependency hashes are in `manifests/MB*_completed_dependencies.json`; current output hashes are in `manifests/sha_manifest.json`. External checkpoint/config hashes are in `manifests/external_model_inputs.json`.

Verification: **24/24 regression checks passed**; 78 current manifest entries verified; 34 external model/config inputs verified; the recorded Track A source/config snapshot is unchanged. No completed notebook contains an error output. All five checkpoint prediction reproductions passed the 1e-5 tolerance. There are 252 same-match matched controls.

[Population denominators](results/MB04_denominators_corrected.csv) separate discovery/estimation and intervention eligibility. There are 1,744 dedicated-player conflict rows (529 estimation, 1,215 discovery), and 252 same-match controls (65 estimation, 187 discovery). Differences from historical numbers cannot isolate the effect of code repairs because the checkpoint family also changed from wd=0.01 to wd=0.1.

| Notebook | Successful execution seconds |
|---|---:|
| [MB01](notebooks/MB01_frozen_reference_dependencies.ipynb) | 2.26 |
| [MB02](notebooks/MB02_model_verification_and_preregistration.ipynb) | 47.06 |
| [MB07R](notebooks/MB07R_repaired_sources_and_regression_tests.ipynb) | 13.22 |
| [MB03](notebooks/MB03_behaviour_on_conflicts.ipynb) | 50.78 |
| [MB04](notebooks/MB04_input_counterfactuals.ipynb) | 149.85 |
| [MB05](notebooks/MB05_representations.ipynb) | 336.89 |
| [MB06](notebooks/MB06_activation_patching.ipynb) | 577.09 |
| [MB07](notebooks/MB07_attention_heads.ipynb) | 88.14 |
| [MB08](notebooks/MB08_mlps.ipynb) | 405.70 |
| [MB09](notebooks/MB09_logit_decomposition.ipynb) | 13.78 |
| [MB10](notebooks/MB10_path_patching.ipynb) | 44.43 |
| [MB11](notebooks/MB11_faithfulness_and_replication.ipynb) | 89.13 |
| [MB11b](notebooks/MB11b_causal_scrubbing.ipynb) | 34.10 |
| [MB11c](notebooks/MB11c_identity_subspace_and_steering.ipynb) | 38.14 |
| [MB12](notebooks/MB12_locked_confirmation.ipynb) | 27.69 |
| [MB13](notebooks/MB13_context_input_survey.ipynb) | 31.11 |


Scientific limits retained explicitly:

- MB04's coherent estimation subset has one match: its point estimate is descriptive and its interval/p-value is undefined.
- Edge completeness is conditional on the declared exact-testing budget. Canonical connection counts determine the random graph controls; component density and non-uniqueness remain measured limitations.
- The estimation-half graph diagnostic evaluates edges selected using estimation-half exact effects; its joint recovery is exploratory and is not an independent validation of that selected graph.
- Scrubbing tests output-key hypotheses and component sufficiency. It does not establish a serial algorithm. Wrong-class comparisons use the intersection of actual non-self assignments over all nodes/draws; empty intersections yield no estimate.
- Steering's unseen doses provide an extra estimation-half diagnostic, not independent confirmation from fresh data.
- MB12 recomputes known-test diagnostics retrospectively. Prior test access prevents a fresh blind confirmatory claim.
- Continuous edits respect support and report clipping/no-ops/realized dose. Factor-only edits and their non-additivity do not rank realistic causal importance. Category-adjacency edits remain disabled pending a source-confirmed codebook.
- MB05's legacy “selectivity” column subtracts target-class remapping accuracy, not an independent control-task accuracy. Probe accuracy does not establish causal use.
- MB05 emitted 5 recorded convergence warnings under its existing 300-iteration probe limit; probe fits are diagnostic and convergence is not guaranteed.

MB07’s first CPU run was interrupted by the server restart. The rerun records per-head checkpoints bound to the model/config/code hashes and uses 512 evaluation windows per batch for this CPU workload; sample rows, ablations and draw counts are unchanged. The interrupted run is preserved in the execution log and status history.
The first MB05 attempt was also interrupted to correct numerical-library thread contention; its failed attempt and the successful 336.89-second rerun are in the execution log/status history.
