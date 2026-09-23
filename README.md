# Soccer MI — Player Prediction from Match Events

Predicts the on-ball player from soccer event sequences. The project compares
statistical baselines (class prior, zone prior, Markov, logistic regression)
against a transformer sequence model, across two context lengths (`T100`,
`T300`) and two architecture depths (2-layer main track, 4-layer "Track A"
benchmark).

## Repo layout

All project content lives under `model_training_evaluation/`:

| Path | Contents |
|---|---|
| `src/` | Core library: data prep (`data.py`), baselines (`baselines.py`, `mlp_baseline.py`), metrics (`metrics.py`), transformer model (`model.py`), batching (`batching.py`), training loop (`training.py`), evaluation (`evaluation.py`), analysis (`analysis.py`) |
| `notebooks/` | Staged pipeline notebooks, run in order (see below) |
| `tests/` | Unit tests for `src/` |
| `artifacts/`, `artifacts_b3/` | Prepared dataset, feature spec, player encodings, dataset audit report, baseline model checkpoints |
| `config/` | `frozen_config.json` — the final locked-in run configuration |
| `manifests/` | `experiment_manifest.jsonl` — log of every experiment run |
| `logs/` | Access/audit logs |
| `results/` | Metrics, tables, and `test_predictions/` per pipeline stage |
| `figures/` | Plots per pipeline stage |
| `runs/` | Main-track hyperparameter sweep: one folder per run, with `run_config.json`, `best.pt`/`last.pt` checkpoints, `history.csv`, predictions |
| `benchmarks/track_a_4layer/` | Parallel 4-layer architecture benchmark — mirrors the top-level `artifacts_b3/config/figures/logs/manifests/results/runs` layout |

`runs/` and `benchmarks/` hold the actual trained checkpoints and are the
bulk of the repo's size (several GB). See **Cloning** below for ways to avoid
pulling all of it if you don't need it.

## Setup

Requires Python. The original runs were produced with **Python 3.13**,
**PyTorch 2.11 (CUDA 12.8)**, **numpy 2.1.3**, **pandas 2.2.3** on a GPU
(Tesla T4) — CPU works too, just slower for training.

```bash
cd model_training_evaluation
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want CUDA acceleration, install the
CUDA-enabled `torch` build for your driver version from
https://pytorch.org/get-started/locally/ instead of the plain PyPI wheel.

## Cloning

**Default — full clone** (recommended if you'll be training, tuning
hyperparameters, or adding new code/experiments — you'll want the existing
run history to compare against and will be writing into `runs/`/`benchmarks/`
yourself anyway):

```bash
git clone https://github.com/alisha971/soccer_mi.git
```

This pulls everything, including all existing checkpoints — expect a large,
slow clone.

**Lightweight — code only** (for reading code or running the test suite
without pulling the checkpoint history):

```bash
git clone --filter=blob:none --no-checkout https://github.com/alisha971/soccer_mi.git
cd soccer_mi
git sparse-checkout init --cone
git sparse-checkout set model_training_evaluation/src model_training_evaluation/tests model_training_evaluation/notebooks
git checkout main
```

With this method, `model_training_evaluation/runs/` and `benchmarks/` (and
anything else not listed in `sparse-checkout set`) **will not exist at all**
in your working directory — not present as empty folders, just absent —
until you pull them in. If you start hyperparameter tuning or need to look
at an existing run, add the paths you need:

```bash
git sparse-checkout add model_training_evaluation/artifacts model_training_evaluation/config model_training_evaluation/runs
```

You can keep narrowing/widening this at any time; it only affects what's
materialized on disk, not what's in the remote history.

## Running the pipeline

Notebooks are numbered and meant to be run in order; each stage reads the
previous stage's outputs from `artifacts/`/`results/`:

1. `01_dataset_audit.ipynb` — data audit, feature prep, player encodings → `artifacts/`, `figures/01_*`
2. `02_baselines_and_encoding_analysis.ipynb` — baseline models, threshold tuning → `results/02_*`, `artifacts/b3_checkpoints/`
3. `03_transformer_implementation_and_tests.ipynb` — model/batching implementation, sanity tests → `results/03_*`
4. `04_training_and_hyperparameter_selection.ipynb` — hyperparameter sweep, pilot training → `runs/`, `results/04_*`
5. `05_architecture_validation_and_seed_analysis.ipynb` — depth/head/context ablations, seed sensitivity → `results/05_*`
6. `06_final_training_and_test_evaluation.ipynb` — final training run and held-out test evaluation → `results/06_*`, `results/test_predictions/`

The `04_4layer_*` / `05_4layer_*` / `06_4layer_*` notebooks are the parallel
"Track A" 4-layer benchmark and write into `benchmarks/track_a_4layer/`
instead of the top-level `runs/`/`results/`.

Run notebooks from `model_training_evaluation/` (so the `src.*` imports
resolve), e.g.:

```bash
cd model_training_evaluation
jupyter lab
```

**Note:** rerunning a notebook overwrites the checkpoint/result files it
produces in place. If you commit after a rerun, you're adding a new full
copy of every changed binary to git history (git doesn't diff binaries
efficiently) — be deliberate about when you commit regenerated runs rather
than doing it after every experimental rerun.

## Training in Google Colab (via a Google Drive copy)

For teammates without a local GPU, training/tuning notebooks can be run in
Colab against a copy of this repo kept in Google Drive:

1. Get the repo into Drive once, e.g. `MyDrive/soccer_mi`. Easiest is to
   clone it from a Colab cell after mounting Drive (this is a private repo,
   so use a
   [personal access token](https://github.com/settings/tokens) in the URL):

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive
   !git clone https://<your-github-username>:<your-PAT>@github.com/alisha971/soccer_mi.git
   ```

2. Runtime → Change runtime type → set Hardware accelerator to **GPU**.

3. `cd` into the project and install dependencies. Colab ships with a
   CUDA-matched `torch` preinstalled — check its version first and only
   force-install `torch==2.11.0` if you specifically need parity with the
   original runs; otherwise skip it and install the rest:

   ```python
   %cd /content/drive/MyDrive/soccer_mi/model_training_evaluation
   !python -c "import torch; print(torch.__version__)"
   !pip install numpy pandas pyarrow scikit-learn scipy joblib matplotlib
   ```

4. Open notebooks straight from Drive: colab.research.google.com → File →
   Open notebook → Google Drive → navigate to
   `soccer_mi/model_training_evaluation/notebooks/`.

Notes:
- Drive-mounted I/O is noticeably slower than local disk for reading/writing
  many small files — fine for a single training run, but a full multi-run
  hyperparameter sweep will be slower than running it locally/on a
  dedicated machine.
- The Drive copy is **not** the git repo — edits and new runs made in Colab
  don't reach GitHub automatically. To keep results, either `!git add`,
  `!git commit`, and `!git push` from a Colab cell (using the same PAT), or
  periodically sync the Drive folder back into a real local clone.

## Tests

```bash
cd model_training_evaluation
pytest tests
```

## Adding new experiments / code

- New source code goes in `src/`, with tests added under `tests/`.
- New hyperparameter sweeps/training runs should write into their own
  subfolder under `runs/` (or `benchmarks/<track_name>/runs/` for a new
  benchmark track), following the existing `transformer_<encoding>_<config>_s<seed>/`
  naming convention so they stay comparable to existing runs.
- Commit new runs separately from code changes (as their own commit) so the
  history stays easy to bisect — mirroring how the initial pipeline stages
  were committed.
