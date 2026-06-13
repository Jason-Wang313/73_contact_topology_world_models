# ICLR Main Gate

Paper: 73 contact_topology_world_models

Hardening version: v4

Gate verdict: KILL_ARCHIVE

Evidence digest: bc72e8ba886f07f4

## Why It Fails

The v4 rebuild produced real local evidence, but the central claim fails:

- `topology_world_model` reaches 0.107 +/- 0.083 success on `combined_stress`.
- The strongest non-oracle baseline, `ensemble_uncertainty_planner`, reaches 0.125 +/- 0.076.
- The paired success difference is -0.018 +/- 0.035 against the topology model.
- `state_only_dynamics_model` and contact-implicit/distance baselines match topology success while producing better graph edit distance or edge F1.
- Ablations do not show a decisive contribution from topology-specific components.

## Remaining Main-Track Blockers

- No real-robot evaluation.
- No public contact-rich benchmark validation.
- The proposed model does not beat the strongest non-oracle baseline on task success.
- Oracle success is low, indicating the local benchmark is hard and not yet a polished benchmark contribution.
- The closest prior-work area is crowded with visuotactile world models, contact-rich policies, contact-implicit MPC, and hybrid force-position control.

The only honest main-conference-safe decision is to archive rather than overclaim.
