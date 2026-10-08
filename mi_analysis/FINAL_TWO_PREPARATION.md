# MB22–MB23 preparation (pending execution)

Prepared 2026-10-08 from historical notebooks; unexecuted notebook copies and
runner live only in `/tmp/opencode/soccer_mi_MB22_MB23.zip`, not published here.
The bundle contains verified MB01–MB21 inputs and the working isolated kernel
configuration (ordinary IPython, inherited extensions disabled).

Changes in bundle copies:
- Honor `MI_ANALYSIS_DIR`, clear historical outputs, label exploratory scope.
- MB22 trains five models at **wd=0.1**, adapting historical wd=0.01 to this
  benchmark: anonymous seeds 0/1/2, identity and single-token controls seed 0;
  maximum 30 epochs, patience 5, unchanged architecture and other settings.
- Build only train/validation windows. Reload checkpoints only with completed
  config/history files and record their hashes on reload. Interrupted partial
  training restarts that model; completed models are reused.
- MB23 uses same-half, different-match donors; historical printed numerical
  comparisons removed. Use regenerated MB14/MB15 outputs for comparisons.

Checks passed: both notebook code cells parse; MB22 setup/data loading and
built-in action-field, causal no-leak and reachability checks ran locally.
No full training or MB23 circuit computation has been run. First-seed circuit
claims remain exploratory, not proof of a unique algorithm. The K32 control
and other training differences do not isolate training-code quality against
SEL-4L K16. No new test access or category-adjacency edits are authorized.

Run `scripts/run_final_two.py MB22`, save progress, then `MB23`. Audit using
`scripts/verify_final_two.py`; return one final project ZIP for verification
before publication. Preserve the MB01–MB13 and MB14–MB21 reports.
