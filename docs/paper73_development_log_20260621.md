# Paper 73 Development Log

Date: 2026-06-21

Paper: `contact_topology_world_models`

Purpose: record allowed development activity before the final frozen run.

## Scope

This pass rebuilt the Paper 73 experiment harness from a v4 archive into an expanded v5 hostile-review protocol.

Development changes were limited to:

- Adding planned benchmark splits, methods, ablations, stress sweeps, aggregate metrics, fixed-risk metrics, and frozen gates.
- Making the runner configurable by explicit CLI arguments.
- Preserving CPU-only and RAM-light execution.
- Repairing plotting and evidence generation so the expanded protocol produces readable artifacts.
- Running one tiny smoke test to catch syntax, shape, file, and decision-path errors.

## Smoke Test

Command:

```powershell
python src\run_experiment.py --seeds 1 --episodes 1 --ablation-episodes 1 --stress-episodes 1 --train-scenes 120 --splits combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress --stress-splits combined_extreme_stress --stress-levels 0.0 1.0 --results-dir results\dev_probe --figures-dir figures\dev_probe --workers 1
```

Outcome:

- Main rows: 30.
- Ablation rows: 24.
- Stress rows: 24.
- Training scenes: 120.
- Required CSV and figure artifacts were produced.
- Terminal decision path executed and returned `KILL_ARCHIVE`.

This smoke test was not used to tune final reported results.

## Repairs Made Before Freeze

- Replaced the old v4 `main()` protocol with an explicit CLI-driven expanded v5 protocol.
- Added `aggregate_metrics.csv`, `aggregate_pairwise_stats.csv`, `fixed_risk_metrics.csv`, and `ablation_aggregate_metrics.csv`.
- Kept stress rows keyed by their real split names instead of collapsing them into `stress_sweep`.
- Changed the maximum-stress gate to compare method averages across frozen max-stress splits.
- Updated negative-case mining to inspect `topology_world_model_v5`.
- Adapted plots for 15 methods and multi-split ablations.

## Freeze Boundary

After this log and `paper73_protocol_freeze_20260621.md` are written, the full run command, methods, splits, seeds, episodes, stress levels, metrics, and gates are frozen. Any later code changes must be bug fixes for recoverable runtime or artifact-generation failures and must be documented.
