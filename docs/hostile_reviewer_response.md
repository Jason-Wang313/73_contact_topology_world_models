# Hostile Reviewer Response

Paper: 73 Contact Topology World Models

## Strongest Technical Threats

- OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation (2026)
- Do World Action Models Generalize Better than VLAs? A Robustness Study (2026)
- PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation Using Tactile-Diffusion Policies (2025)
- Contact-implicit Model Predictive Control: Controlling diverse quadruped motions without pre-planned contact modes or trajectories (2025)
- Force Policy: Learning Hybrid Force-Position Control Policy under Interaction Frame for Contact-Rich Manipulation (2026)
- ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation (2026)
- Fast reprogramming and adaptive reproduction of contact-rich assembly (2026)
- Co-Training Multimodal World Models and Diffusion-Guided Policies for Zero-Shot Contact-Rich Manipulation (n.d.)

## Hostile ICLR Review

A hostile reviewer should reject this as an ICLR-main submission. The v4 rebuild is much stronger than the previous archive because it implements a MuJoCo contact-topology benchmark and multiple baselines. However, the proposed topology world model does not beat the strongest non-oracle baseline on task success.

The most damaging result is that graph-prediction improvement does not translate into control improvement. `topology_world_model` has higher edge F1 than `ensemble_uncertainty_planner`, but lower task success. Meanwhile, state-only and distance/contact-implicit baselines match topology success with better graph edit distance.

The 2026-06-15 continuation audit rechecked the exact numbers from the raw CSVs: on combined stress, `topology_world_model` reaches 0.107 success while `ensemble_uncertainty_planner` reaches 0.125, with topology-minus-ensemble paired success difference -0.018 +/- 0.035. The topology method has higher edge F1 than the ensemble baseline, 0.610 versus 0.364, but this does not become a task-success gain. At stress level 1.00, `topology_world_model` reaches 0.000 success while the oracle topology planner reaches 0.086.

## Honest Action

Mark `KILL_ARCHIVE`. Keep the repository as a negative-result package and do not submit it as an ICLR main paper.

## What Would Be Needed To Revive

- Show that topology prediction improves downstream actions, not just graph metrics.
- Validate on real robot hardware or a public contact-rich benchmark.
- Replace the local hand-engineered topology planner with a learned model/planner trained on larger contact data.
- Beat contact-implicit MPC, state-only dynamics, distance topology, uncertainty ensembles, and relevant visuotactile world-model baselines.
