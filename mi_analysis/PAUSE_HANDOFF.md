# Notebook rerun handoff — COMPLETED 2026-10-06

This note supersedes the earlier stop-state version. The full 16-notebook
sequence completed: MB01, MB02, MB07R, MB03, MB04, MB05, MB06 ran locally on
CPU; MB07–MB13 ran on a Colab Tesla T4 in full mode. All are promoted into
this directory, and `scripts/finalize_mi_rerun.py` passed, writing
`RERUN_REPORT.md`.

## Completion evidence

- `logs/rerun_status.json`: all 16 notebooks `completed` (MB07 88.14 s,
  MB08 405.70 s, MB09 13.78 s, MB10 44.43 s, MB11 89.13 s, MB11b 34.10 s,
  MB11c 38.14 s, MB12 27.69 s, MB13 31.11 s).
- Verified after delivery: 0 error outputs in any completed notebook,
  full-mode settings (`n_bootstrap=2000`) with CUDA for MB07–MB13, no code
  changes to staged notebooks or `src/`, 79/79 manifest hashes verify,
  16/16 promotion records match, 8/8 MB07 head checkpoints, 24/24
  regression checks pass, 252 same-match controls.
- See [SURGICAL_FIXES.md](SURGICAL_FIXES.md) for repair details and known
  limitations, [RUNNING.md](RUNNING.md) for run setup, and the root
  [MI_ANALYSIS_TASK_HANDOFF.md](../MI_ANALYSIS_TASK_HANDOFF.md) for the full record.

## Environment variant (important)

MB07–MB13 did **not** run in the recorded pinned environment. Colab used a
Tesla T4 with Python 3.12.15 and **torch 2.11.0+cu130 instead of pinned
torch 2.10.0**; remaining package versions were not captured. All
full-model measurements for MB07–MB13 were generated inside that run, so
results are self-consistent, but treat cross-environment numerical
comparisons accordingly. `config/execution_environment.json` still
describes the local CPU baseline. MB12 remains retrospective diagnostics
on a previously accessed test split — no fresh confirmation claim.

## Original task (done)

Apply the surgical repairs to the repaired MI notebooks, rerun them against
the available SEL-4L 4L/2H/K16 benchmark checkpoints (weight decay 0.1,
seeds 100–104), and promote completed notebooks and dependencies into
`mi_analysis/`. Category-adjacency edits stayed disabled (no source codebook).
