# Hyperparameter retune — report

## Objective

Re-tune the optimizer and schedule hyperparameters for the frozen
`T300 / L2-H2-D128-F512 / K16` transformer. The learning rate and weight
decay had previously been selected on the `T100 / L4 / K32` configuration
(notebook 04) and were never re-checked so will be tuning that,
while keeping other explored things fixed.

## Setup

- Model/data: frozen `config/frozen_config.json` base (`T300` encoding,
  `pre_action` possession features, context 16, stride 8, 2 layers,
  2 heads, `d_model` 128, `d_ff` 512).
- Budget: 30 epochs, early stopping patience 5, gradient clip 1.0,
  AdamW `betas=(0.9, 0.999)`.
- Hardware: Tesla T4 (CUDA). All 18 trials are full training runs.
- Validation only: every decision uses the validation split, each event
  scored once. The test split was never touched during selection.

## Selection metric and decision rule

Primary metric: **mean validation NLL over seeds 0–2** (lower is better).
A challenger replaces the incumbent only if it is better **beyond noise**,
i.e. both hold:

1. the 95% match-level bootstrap confidence interval of the per-event NLL
   difference (challenger minus incumbent) lies entirely below 0;
2. every challenger seed beats every incumbent seed (complete separation).

Otherwise the tie rule keeps the lower learning rate, and for equal learning
rates the incumbent weight decay. Accuracy, macro-F1 and ECE are reported
as secondary metrics for every trial.

## Search

| Stage | What | Seed-0 result |
|---|---|---|
| T2-lr | learning rate `{1e-4, 3e-4, 1e-3}`, wd 0.1 | `3e-4` (0.9070) < `1e-3` (0.9073) < `1e-4` (0.9104) |
| T3-wd | weight decay `{0.01, 0.1, 0.3}` at the two best LRs | `0.01` wins at both LRs; best `(1e-3, 0.01)` = 0.9066 |
| T4-seeds | seeds 1–2 for `(3e-4, 0.01)` and `(1e-3, 0.01)` | means 0.9076 / 0.9078 vs incumbent 0.9074 |
| T5-sched | batch size `{128, 256}` × warmup `{0.05, 0.1}` at `(3e-4, 0.1)` | warmup `0.1` wins at both sizes; best `(128, 0.1)` = 0.9061 |
| T5-seeds | seeds 1–2 for `(128, 0.1)` and `(256, 0.1)` | `(128, 0.1)` mean 0.9057 (SD 0.0003); `(256, 0.1)` mean 0.9070 |

Finalist comparisons (challenger minus incumbent, seeds 0–2):

| Comparison | Diff (nats) | 95% CI | Separated | Better beyond noise |
|---|---|---|---|---|
| `(3e-4, 0.01)` vs incumbent | +0.0002 | [-0.0008, +0.0012] | no | no |
| `(1e-3, 0.01)` vs incumbent | +0.0004 | [-0.0013, +0.0022] | no | no |
| `(1e-3, 0.01)` vs `(3e-4, 0.01)` | +0.0002 | [-0.0015, +0.0021] | no | no |
| `(bs128, wu0.1)` vs incumbent | -0.0016 | [-0.0035, +0.0002] | no | no |
| `(bs256, wu0.1)` vs incumbent | -0.0004 | [-0.0014, +0.0006] | no | no |

## Verdict

**Keep the frozen configuration: learning rate 3e-4, weight decay 0.1,
batch size 256, warmup fraction 0.05.** No challenger clears the bar:
seed noise (SD ~0.002) is larger than every observed gain, and the
incumbent's seed 101 (0.9048, from the earlier 8-seed history) beats every
challenger seed outright. The closest signal — warmup `0.1`, best at both
batch sizes with the tightest seed cluster (SD 0.0003 at `bs128`) — is worth
revisiting only with a larger seed budget.

## Reproducibility

- `results/hyperparameter_sweep.csv`: one row per trial (see glossary below).
  Verified cell-for-cell against the run artifacts by
  `tests/test_tuning_table.py`.
- `trials/<trial_id>.json`: exact hyperparameters + run directory per trial.
- `runs/<run_name>/`: `final_results.json`, `history.csv` and
  `val_predictions.npz` (per-event validation logits) per trial.
- `scripts/train_one.py`: builds a trial config and trains it;
  `scripts/score.py`: reports mean validation NLL;
  `scripts/aggregate.py`: seed-mean + bootstrap comparison.

## Glossary

- **seed**: number initializing all random generators. Same seed gives the
  same initial weights and training order, so a run reproduces exactly.
  Different seeds measure training noise.
- **val NLL (nats)**: mean negative log-likelihood on validation events —
  minus the log-probability the model assigns to the true action, averaged
  with each event scored once. Lower is better.
- **val accuracy / macro-F1 / ECE**: top-1 accuracy, macro-averaged F1 over
  the 15 action classes, and expected calibration error. Secondary metrics.
- **best epoch**: epoch with the lowest validation NLL; its checkpoint is
  the reported result. Typical range here: 4–8 of 30.
- **diverged**: whether training hit a non-finite loss (none did).
- **runtime_seconds**: measured wall-clock training time on the T4.
- **weight decay (wd)**: L2 regularization strength in AdamW.
- **warmup fraction (wu)**: share of total steps spent ramping the learning
  rate linearly to its peak before cosine decay.
- **batch size (bs)**: training windows per weight update (256 windows =
  4096 predictions at K=16).
- **decision**: `reference` (incumbent baseline), `finalist` (promoted to
  multi-seed), `supporting`/`supporting-seed` (evidence row),
  `rejected` (worse than reference on seed 0).
- **CI / separated**: the 95% bootstrap interval and seed-separation check
  from the decision rule above.
