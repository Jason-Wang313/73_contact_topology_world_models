# Novelty Boundary Map

## Crowded Territory

- Visuotactile world models for contact-rich manipulation.
- Contact-implicit MPC and hybrid force-position control.
- Distance/contact heuristics for graph prediction.
- State-only dynamics models with induced contact graphs.
- Uncertainty ensembles and risk-aware contact planning.

## Tested Boundary

The tested boundary was whether explicitly predicting contact graph births/deaths, components, and jam modes improves downstream manipulation compared with strong state, distance, uncertainty, and contact-implicit baselines.

## What Falsified The Boundary

The topology model did not beat the strongest non-oracle baseline on combined-stress task success. It improved some graph diagnostics but failed to convert them into better control.

## Remaining Possible Boundary

A future project could still explore learned topology abstractions on real visuotactile datasets or public contact-rich benchmarks, but that is not demonstrated here.
