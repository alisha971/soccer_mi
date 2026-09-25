# Evals summary (iters 0-19, full sweep COMPLETE)
Metric: 0.9068 -> 0.9068 incumbent kept (best challenger seed-mean 0.9076, gap within seed noise) | Kept: 15 / Discard: 5 | Trend: flat at optimum
Recommendation: keep frozen_config.json unchanged (lr=3e-4, wd=0.1). Negative result with teeth: LR order re-confirmed on final arch, wd 0.01 not distinguishable from 0.1 beyond noise.
Top changes: (1) T2b exact reproduction of incumbent (stack trust), (2) stale cross-mode cache guard in train_one.py, (3) aggregate.py bootstrap decision artifact.
Cost: 11 full + 1 smoke T4 trials, one session, no auth expiry hit.
