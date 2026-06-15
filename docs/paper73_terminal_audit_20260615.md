# Paper 73 Terminal Audit - 2026-06-15

Paper: `contact_topology_world_models`
Decision: `KILL_ARCHIVE`
ICLR-main ready: no

## Verification Performed

1. Source compile gate passed with `python -m py_compile src/run_experiment.py`.
2. CSV integrity gate passed for all result CSVs: files are present, nonempty, finite, and schema-readable. Blank `stress_level` values are expected only in non-stress rollout tables.
3. Evidence scale matched the reported claims:
   - Main rollouts: 2,240
   - Ablation rollouts: 252
   - Stress rollouts: 1,050
   - Seeds: 0, 1, 2, 3, 4, 5, 6
4. Baselines were present in the main evidence: `last_contact_persistence`, `distance_threshold_graph`, `state_only_dynamics_model`, `pairwise_contact_classifier`, `ensemble_uncertainty_planner`, `contact_implicit_mpc_baseline`, and `oracle_topology_planner`.
5. PDF rebuild completed and `C:/Users/wangz/Downloads/73.pdf` was refreshed.
6. BibTeX sort warnings were repaired by adding stable `key` fields to the local reference entries; one unsafe underscore key was replaced with a LaTeX-safe key.
7. No visible Desktop copy of `73.pdf` was present after the audit.

## Fatal Evidence

The proposed contact-topology world model fails the ICLR-main decision rule because graph-prediction gains do not become downstream control gains. On combined stress, `topology_world_model` reaches 0.107 success while `ensemble_uncertainty_planner` reaches 0.125. The paired topology-minus-ensemble success difference is -0.018 +/- 0.035.

The topology model does improve one intermediate diagnostic: edge F1 is 0.610 versus 0.364 for the ensemble baseline. However, task success is lower, and several non-topology/contact baselines match the topology model's 0.107 success. Some of those baselines also have lower graph edit distance than the proposed method.

The ablation suite does not rescue the mechanism. `topology_full` reaches 0.071 success, while `topology_no_uncertainty_penalty` reaches 0.095 and other removed-component variants are statistically overlapping.

At stress level 1.00, `topology_world_model` reaches 0.000 success. The oracle topology planner reaches 0.086, showing that the task is difficult, but not that the proposed method is submission-ready.

## Decision

Paper 73 remains `KILL_ARCHIVE`. It is a reproducible negative result: topology prediction is not enough without demonstrated downstream control improvement.

## Revival Requirements

To revive this paper, a future version would need a topology representation and planner that decisively improves downstream task success over uncertainty ensembles, contact-implicit MPC, state-only dynamics, distance graph heuristics, and pairwise contact classifiers, ideally with real-robot or public contact-rich benchmark validation.

