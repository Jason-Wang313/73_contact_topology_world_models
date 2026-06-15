# Paper 73 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15
Paper: 73 - `contact_topology_world_models`
Target venue posture: ICLR main only if supported by decisive downstream evidence
Current terminal label entering audit: `KILL_ARCHIVE`

## Goal

Rebuild and audit Paper 73 as a real submission candidate rather than a cosmetic manuscript. The audit must decide whether the MuJoCo contact-topology evidence can honestly support an ICLR-main world-model submission, or whether the paper remains a terminal negative result.

## Decision Rule

Upgrade from `KILL_ARCHIVE` only if all of the following are true:

1. `topology_world_model` decisively beats the strongest non-oracle baseline, especially `ensemble_uncertainty_planner`, on combined-stress task success.
2. Graph-prediction improvements translate into downstream control gains, not only better intermediate edge metrics.
3. The paired confidence interval supports a real positive task-success effect.
4. Stress-level results remain favorable under the hardest contact-rich settings.
5. Ablations show topology-specific components are necessary for task success.
6. The evidence is reproducible from checked-in code, raw CSVs, and a clean PDF build.

If any of these gates fail, preserve `KILL_ARCHIVE` and document the exact failure mode.

## Evidence Gates

Run these checks before changing the decision:

1. Code integrity: compile the experiment source with `python -m py_compile src/run_experiment.py`.
2. Result integrity: verify all required CSVs exist, are nonempty, finite, and schema-valid.
3. Scale check: confirm the recorded evidence includes 7 seeds, 2,240 main rollouts, 252 ablation rollouts, and 1,050 stress rollouts.
4. Baseline check: verify `last_contact_persistence`, `distance_threshold_graph`, `state_only_dynamics_model`, `pairwise_contact_classifier`, `ensemble_uncertainty_planner`, `contact_implicit_mpc_baseline`, and `oracle_topology_planner` are present.
5. Stress check: confirm stress-level results are represented and compare the proposed method against the strongest non-oracle stress baseline.
6. Ablation check: confirm whether the full topology world model beats removed-component variants on task success.
7. Paper build: run LaTeX/BibTeX to produce a clean PDF and copy only the numbered PDF to `C:/Users/wangz/Downloads/73.pdf`.
8. Artifact hygiene: confirm no numbered PDF is copied to the visible Desktop.
9. GitHub hygiene: confirm the matching public GitHub repository exists and the local commit is pushed.
10. Root-report hygiene: update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.

## Expected Risk

The existing evidence summary reports that `topology_world_model` reaches 0.107 combined-stress success while `ensemble_uncertainty_planner` reaches 0.125, with paired topology-minus-ensemble success difference -0.018 +/- 0.035. The topology model improves some graph metrics, but the main claim requires task-success improvement. Unless verification contradicts that result, Paper 73 cannot honestly become submission-ready in this pass.

## Execution Order

1. Re-check repository cleanliness and result inventory.
2. Run code and CSV integrity gates.
3. Rebuild the paper PDF and repair recoverable build warnings.
4. Write a terminal audit with exact evidence and rejection rationale.
5. Update child status, local audit docs, and root reports.
6. Commit and push the Paper 73 repository.
7. Verify `Downloads/73.pdf`, no Desktop copy, public GitHub visibility, clean git state, and root report consistency.

