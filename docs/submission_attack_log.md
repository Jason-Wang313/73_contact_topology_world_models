# Submission Attack Log

Paper: 73 contact_topology_world_models

This v4 pass applies the ICLR main-conference bar after rebuilding the evidence package.

## 2026-06-15 Continuation Audit
Attack: A topology world model must improve downstream contact-rich control, not merely intermediate graph metrics.

Verdict: Fatal. The verified CSVs contain 2,240 main rollouts, 252 ablation rollouts, and 1,050 stress rollouts across 7 seeds. On combined stress, `topology_world_model` reaches 0.107 success versus 0.125 for `ensemble_uncertainty_planner`, with topology-minus-ensemble paired success difference -0.018 +/- 0.035. The topology model improves edge F1 over the ensemble baseline, 0.610 versus 0.364, but task success is lower. Multiple non-topology/contact baselines match topology success, and at stress level 1.00 the topology method reaches 0.000 success.

Action: Keep KILL_ARCHIVE and preserve the reproducible negative result without claiming ICLR-main readiness.

## Attack 1: Does the proposed method beat the strongest implemented baseline?

Verdict: No. `topology_world_model` reaches 0.107 +/- 0.083 combined-stress success. `ensemble_uncertainty_planner` reaches 0.125 +/- 0.076.

Action: Kill/archive.

## Attack 2: Is the paired comparison favorable?

Verdict: No. Topology-minus-ensemble paired success difference is -0.018 +/- 0.035 over seven seeds.

Action: Kill/archive.

## Attack 3: Do graph metrics translate into control?

Verdict: Not enough. The topology model improves edge F1 over the ensemble baseline, but task success is lower.

Action: Treat graph prediction as an insufficient intermediate result.

## Attack 4: Do ablations support the topology mechanism?

Verdict: No decisive support. Ablations are statistically overlapping and do not prove that birth/death, components, jam/slip, or topology-aware planning is essential.

Action: Kill/archive.

## Attack 5: Are the baselines real enough for a first submission gate?

Verdict: Improved but still insufficient for acceptance. The repo now has implemented controllers and predictors, but lacks real robot and external public benchmark validation.

Action: Keep as a local negative-result archive.

## Attack 6: Does hostile prior work leave enough novelty?

Verdict: Not in the current evidence. Visuotactile world models, contact-implicit MPC, hybrid force-position control, and contact-rich manipulation policies are crowded. A topology representation would need decisive downstream gains.

Action: Kill/archive.

## Attack 7: Could text polishing rescue the paper?

Verdict: No. The central empirical claim fails.

Action: Stop at v4 negative-result package.
