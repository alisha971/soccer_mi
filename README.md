# Tracing Identity–Context Arbitration in a Soccer Event Transformer

Code and saved results for the anonymous ARR submission of the same title.

The repository has two parts:

- **Track A (predictive models):** data preparation, baselines, Transformer training, model
  selection, input ablations and test evaluation.
- **Track B (mechanistic analysis):** conflict-set construction, input interventions, activation
  and path patching, MLP and attention analyses, circuit sufficiency and causal scrubbing on
  the analysed model (REF-4L, seeds 100–104, primary seed 103).

## Repository structure

```
model_training_evaluation/         Track A source, configs, notebooks
mi_analysis/                       Track B notebooks, source, configs, results
mi_analysis/targetted_analysis/    final player, history and interaction analyses (run.py)
```

## Setup

- Python 3.12, PyTorch 2.10 (CUDA 12.8 used for GPU runs; CPU also works, more slowly)
- `pip install -r mi_analysis/requirements-rerun.txt`
- Notebooks resolve paths relative to the repository root. `SOCCER_MI_ROOT` overrides the root
  when it cannot be inferred; `MI_ANALYSIS_DIR` overrides the Track B output folder.

The recorded environments are in `mi_analysis/config/execution_environment.json` and
`mi_analysis/config/followup_execution_environment.json`.

## Data

We use StatsBomb Open Data (events, lineups and match metadata; 360 freeze frames are not used).
The raw data is not redistributed here. The repository references the public mirror at
https://github.com/statsbomb/open-data; clone or download it and point the data-loading code at
your local copy. Track A notebook `00_statsbomb_subset_construction_and_cleaning.ipynb` builds
the event table from that source, and `01_dataset_audit.ipynb` reports the dataset statistics
and class distribution.

StatsBomb Open Data is provided under its own licence and terms of use:
https://github.com/statsbomb/open-data. Please read them before use.

## Reproducing the paper

Notebooks are provided **with saved outputs**, so results can be inspected without re-running.
Re-running everything is a multi-GPU-hour job and needs the trained checkpoints, which are not
redistributed here.

Run order:

1. Track A: `00`, `01`, `02`, `03`, then the hyper-parameter and architecture-selection
   notebooks (`04`, `05`, `08`–`11`), model selection (`12`), final training and test
   (`13`, `14`), benchmark and ablations (`15`, `16`, `17`), correctness tests (`15_Correctness`).
2. Track B: MB01, MB02, MB07R, MB03, MB04, MB05, MB06, MB07, MB08, MB09, MB10, MB11, MB11b,
   then the follow-up notebooks MB14, MB15, MB16, MB17, MB18, MB19, MB21, then
   `mi_analysis/targetted_analysis/run.py`.

Use full mode (`MI_RUN_MODE=full`). Smoke mode uses smaller samples and does not reproduce the
paper's numbers.

The Track B runner accepts individual phases:

```bash
python -B mi_analysis/targetted_analysis/run.py --check-only     # small regression check
python -B mi_analysis/targetted_analysis/run.py                  # all phases, rebuild the report
python -B mi_analysis/targetted_analysis/run.py --section identity
```

`--section` accepts `identity`, `history`, `interaction`, `slots`, `probe`, `report` or `all`.
It reuses named functions from the notebooks without executing notebook write cells, and it
writes `mi_analysis/targetted_analysis/results/<section>.json`. Intervals are 2,000 whole-match
bootstrap draws at seed 0; seed-to-seed variation is reported separately and is not folded
into those intervals.

## Where each result comes from

Every row below was checked against the repository: the notebook exists and writes the named
result file.

| Paper item | Source (verified) |
|---|---|
| Dataset statistics, action vocabulary, class distribution | `model_training_evaluation/notebooks/00_statsbomb_subset_construction_and_cleaning.ipynb`, `01_dataset_audit.ipynb` |
| Benchmark and input-ablation table | `model_training_evaluation/notebooks/15_final_benchmark.ipynb`, `16_ablations.ipynb` |
| Model selection (REF-2L, REF-4L, SEL-4L) | `12_final_model_selection.ipynb`, `17_selected_model_release.ipynb`, `04_4layer_training_and_hyperparameter_selection.ipynb` |
| REF-4L vs REF-2L validation NLL gap (0.0049 nats, 95% CI [0.0025, 0.0073]) | `05_4layer_architecture_validation_and_seed_analysis.ipynb`, `06_4layer_final_training_and_test_evaluation.ipynb` |
| Conflict set and populations | MB01 (frozen conflict files), MB02, MB03 → `results/MB03_behaviour.csv` |
| Player interventions and tendency subspace | `targetted_analysis/run.py` → `identity.json` |
| Earlier-window vs scored-event context | `targetted_analysis/run.py` → `history.json` |
| Identity–context interaction sources | `targetted_analysis/run.py` → `interaction.json` |
| Anonymous-slot controls | `targetted_analysis/run.py` → `slots.json` |
| Action decoding vs shuffled labels | `targetted_analysis/run.py` → `probe.json` |
| Linear probes and logit lens | MB05 → `results/MB05_probes.csv` |
| Input counterfactuals | MB04 → `results/MB04_factorial.parquet` |
| Activation patching | MB06 → `results/MB06_site_summary.csv` |
| Attention heads | MB07 → `results/MB07_attention_relations.csv` |
| MLP and hidden-unit analysis | MB08 → `results/MB08_neurons.csv`; MB17 → `results/MB17_units.csv` |
| Logit decomposition and direct/indirect routes | MB09 → `results/MB09_decomposition.parquet` |
| Path patching and candidate circuit | MB10 → `results/MB10_paths.parquet` |
| Circuit atlas and pruned circuit figure | MB19 → `results/MB19_edges_estimation.csv` |
| Circuit sufficiency and replication | MB11 → `results/MB11_faithfulness.csv`; MB14 → `results/MB14_shapley.csv` |
| History comparison and interaction strata | MB15 → `results/MB15_history.csv` |
| Decision redundancy and cross-model comparison | MB18 → `results/MB18_redundancy.csv` |
| Player knowledge stored vs used | MB21 → `results/MB21_stored.csv` |

**Unconfirmed:** the causal-scrubbing table is produced by both `MB11b_causal_scrubbing`
(`results/MB11b_summary_corrected.json`) and `MB16_corrected_scrub_and_steering_geometry`
(`results/MB16_summary_corrected.json`). Which one is the source of the reported table could not
be determined from the repository and must be confirmed by the author before submission. Both
notebooks are retained, each with a note describing its unreported sections.

## Notes

- Some notebooks contain sections that are not reported in the paper. These notebooks start
  with a note naming those sections: MB04, MB06, MB07, MB08, MB09, MB15, MB16, MB17, MB18,
  MB19, MB21.
- Notebook MB01 checks the frozen conflict files against an archive that is not included, so
  that hash-comparison cell will not run as is. The frozen files themselves are included.
- Hash manifests under `mi_analysis/manifests/` were recorded before notebook metadata and
  local path strings were anonymized, and before unreported analyses were removed. Manifests
  that listed a now-removed unreported result therefore reference a file that is no longer
  present; this is expected. No code, output or reported result was changed.
- `mi_analysis/src/twotoken.py`, `mi_analysis/src/twotoken_followup.py` and the
  `artifacts/twotoken/two_token_anon_s*` checkpoints are retained although the notebooks that
  originally produced them are not part of this release: `targetted_analysis/run.py` imports
  them at runtime to compute `slots.json`. Notebook outputs are archived as they were run, so
  a saved output may still mention a file that is no longer present.
- Several large inputs are not included: model checkpoints, prepared event tables and the
  intermediate `.parquet`/`.npz` arrays. The saved CSV/JSON result files are included for every
  reported table. Regenerating the checkpoints requires re-running the Track A training
  notebooks named in the run order above.
- The analysis in `targetted_analysis/` is exploratory validation on existing checkpoints. It
  reuses the same validation events across seeds, so agreement across seeds is not independent
  replication. Correlation measures directional alignment, not the fraction of a player effect
  explained by a subspace.

## License

Code: see the licence declared by the repository owner. Data: see the StatsBomb Open Data
licence and terms at https://github.com/statsbomb/open-data.
