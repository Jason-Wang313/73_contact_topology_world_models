# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger synthetic baselines, ablations, stress tests, and negative cases.
- Narrowed claims to diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that the existing local artifacts were insufficient for main-track submission.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Real MuJoCo Rebuild

- Replaced the synthetic scaffold with a MuJoCo contact-topology benchmark.
- Implemented persistence, distance-threshold, state-only dynamics, pairwise contact, ensemble uncertainty, contact-implicit MPC, topology world model, and oracle planners.
- Ran seven seeds, eight episodes per seed/split, ablations, stress sweeps, uncertainty intervals, paired comparisons, figures, and negative cases.
- Found that `topology_world_model` does not clear `ensemble_uncertainty_planner` on combined stress: 0.107 +/- 0.083 versus 0.125 +/- 0.076 success.
- Terminal decision remains: KILL_ARCHIVE.
