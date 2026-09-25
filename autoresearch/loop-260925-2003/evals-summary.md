# Evals summary T5 (iters 0-9, COMPLETE)
Metric: 0.9070 -> challenger-mean 0.9057 but verdict KEEP (CI crosses 0, no separation) | Kept: 9 / Discard: 1 | Trend: flat at optimum
Recommendation: keep frozen bs=256, warmup=0.05. Carry-forward: wu=0.1 never hurts (3/3 seed0 wins, tightest seed cluster SD 0.0003 at bs128) - revisit if schedule re-opened with bigger seed budget.
Top changes: (1) --bs/--wu + tag-suffix dirs, (2) T5b seed0 0.9061, (3) aggregate T5 verdict logic-proof (separation impossible vs 8-seed incumbent history).
Cost: 7 full T4 trials, no auth issues.
