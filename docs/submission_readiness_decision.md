# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Submission-hardening version: v4.

## Evidence Used

The v4 rebuild replaces the synthetic scaffold with a MuJoCo contact-topology benchmark. It includes implemented baselines, a proposed topology world model, an oracle topology planner, seven seeds, eight evaluation episodes per seed/split, ablations, stress sweeps, uncertainty intervals, paired statistics, figures, raw rollouts, and negative cases.

## Gate Result

On `combined_stress`:

- `topology_world_model`: 0.107 +/- 0.083 success.
- `ensemble_uncertainty_planner`: 0.125 +/- 0.076 success.
- Paired topology-minus-ensemble success: -0.018 +/- 0.035.
- `topology_world_model` edge F1: 0.610.
- `ensemble_uncertainty_planner` edge F1: 0.364.

The topology model improves graph prediction relative to the ensemble baseline but does not improve task success. Strong geometric/state baselines also match topology success while producing better edge F1 or graph edit distance.

## Terminal Action

Archive/kill for ICLR main. Do not submit this paper as an ICLR main paper.

Revival condition: show that contact-topology prediction changes decisions and improves success on real robot or public contact-rich benchmarks, not merely graph diagnostics.
