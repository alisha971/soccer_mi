# MB22–MB23 execution audit

Both final follow-ups completed; all 26 notebook completion hashes, latest dependency snapshot, manifest hashes, external model hashes, unchanged MB01–MB21 baseline and unchanged test-access log verified locally on 2026-10-08. MB22/MB23 code cells executed without error outputs. All five new checkpoints load and produce finite predictions on a small validation batch. Sources match the prepared notebooks plus the authorized namespace amendment. This is execution/provenance verification, not independent scientific replication.

Actual environment: Python 3.12.15, Torch 2.10.0, CUDA 12.8, Tesla T4; package versions preserved in `config/final_two_execution_environment.json`. Delivered ZIP SHA-256: `dcfca23e20ccd3e1b63724a54cb5a2effcdb5a1960f5f85b9e363419385c874f`.

MB22 trained five wd=0.1 models. The initial attempt completed training but overwrote protected historical `src/twotoken.py`; it failed the guards and was not promoted. The original was restored, and new code moved to `src/twotoken_followup.py`. The successful retry reloaded completed models (34.53 seconds, excluding training). MB23 took 3470.27 seconds. Namespace amendment hashes are preserved in `manifests/final_two_namespace_amendment.json`.

## Results and limits

- MB22 single control is 0.0442 nats worse than SEL-4L; two-token identity is 0.0279 nats worse than single and fails the plan's equivalence threshold. Anonymous seed 0 is 0.0189 nats better than two-token identity. These comparisons do not isolate training-code quality or identity removal causally; K32 and other training/design differences matter.
- MB23's exploratory rule labels attention used for all four behaviours, but establishes no MEANINGFUL circuit. Current-situation circuit is PARTIAL; identity, previous-event and interaction circuits are NOT ESTABLISHED. Negative circuit faithfulness is retained, not treated as a verification error.
- Circuit analysis uses the first anonymous seed only; threshold verdicts do not prove a unique algorithm or across-seed circuit replication.
- MB16 legacy scrub remains invalidated; use Section R corrected evidence. MB12 remains retrospective. No new test access or category-adjacency edits were introduced.
