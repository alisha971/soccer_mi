# Notebook rerun handoff

## Task

Apply the surgical repairs to the repaired MI notebooks, rerun them against the available SEL-4L 4L/2H/K16 benchmark checkpoints (weight decay 0.1, seeds 100–104), and promote completed notebooks and dependencies into `mi_analysis/`. This is a new benchmark analysis, not a reproduction of the historical weight-decay-0.01 runs. Category-adjacency edits were explicitly left disabled because the source codebook was unavailable.

## Progress

- MB01, MB02, MB07R, MB03, MB04, MB05 and MB06 completed; their notebooks/results and dependencies are in this directory. MB02 prediction parity checks and MB07R repair regression checks passed. MB05 completed with convergence warnings; an earlier slow attempt was interrupted and rerun with thread limits.
- The remaining sequence (MB07, MB08, MB09, MB10, MB11, MB11b, MB11c, MB12, MB13) was started. MB07 again reached its expensive all-event head-ablation cell (cell 23/31) and produced no completed per-head checkpoint before it was stopped. Its prior attempt had also been interrupted in that cell.
- MB07 was amended to use a larger CPU batch and save/resume completed heads with strict analysis/checkpoint/code hashes. No completed head checkpoint exists yet.
- MB10–MB13 repairs and the rerun finalizer are staged under `.mi_analysis_rerun/`; the final review report has not been generated. See [SURGICAL_FIXES.md](SURGICAL_FIXES.md) for repair details and known limitations, and [RUNNING.md](RUNNING.md) for run setup.

## Stop state

At the user's request, the runner (PID 344217) and its Jupyter kernel (PID 344291) were terminated. A process check found no remaining `run_mi_notebooks.py` or `ipykernel_launcher` processes. The run status records MB07 as failed at cell 23/31; subsequent notebooks have not run. The latest execution log is `.mi_analysis_rerun/logs/execution_MB07_MB08_MB09_MB10_MB11_MB11b_MB11c_MB12_MB13.log`.

To resume, rerun the same notebook sequence with `scripts/run_mi_notebooks.py`; completed notebooks are already promoted, and MB07's checkpoint logic will resume only after a head checkpoint has been written.
