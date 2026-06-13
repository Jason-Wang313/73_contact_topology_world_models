# Paper 73 Rebuild Plan: Contact Topology World Models

Date: 2026-06-13

## Goal

Rebuild Paper 73 into a real ICLR-main-target robotics submission candidate, or terminate it honestly as `STRONG_REVISE` / `KILL_ARCHIVE` if the evidence does not justify submission. The central question is whether predicting contact-topology changes gives a robot a useful world model beyond state-only dynamics, distance/contact heuristics, uncertainty ensembles, and contact-implicit control baselines.

## Core Claim To Test

Contact-rich manipulation failures often come from wrong beliefs about which bodies are in contact, which contacts will appear/disappear, and whether the contact graph becomes connected, jammed, or separated. A topology world model should explicitly predict contact graph transitions and use those predictions for action selection. It should not win merely by seeing more state features or by using a hand-coded distance threshold.

## High-Fidelity Benchmark

Build a MuJoCo tabletop contact benchmark with a planar pusher, two movable blocks, a wall/fixture, and a target pocket. Each rollout logs object poses, velocities, normal/tangential impulses, pairwise contact graph edges, graph components, contact births/deaths, slip/jam events, and task progress.

Evaluation splits:

- `nominal_push_to_pocket`: pusher moves a block into a target pocket through simple contacts.
- `contact_chain_transfer`: success requires pushing one object through another, so the graph must become pusher-block-block.
- `fixture_topology_shift`: held-out fixture/wall geometry changes which contacts appear first.
- `friction_mass_shift`: held-out mass and friction alter sticking/sliding transitions.
- `combined_stress`: geometry shift, friction/mass shift, noisy contact sensing, actuator limits, and distractor contacts.

## Methods To Implement

- `last_contact_persistence`: predicts the next contact graph from the previous graph.
- `distance_threshold_graph`: predicts contacts from pairwise distances and relative velocities.
- `state_only_dynamics_mlp`: learned state-transition model without explicit graph loss, evaluated by induced contacts.
- `pairwise_contact_classifier`: supervised pairwise contact predictor from state/action features.
- `ensemble_uncertainty_planner`: ensemble of pairwise predictors with conservative action selection.
- `contact_implicit_mpc_baseline`: sampling controller with contact-distance penalties and robust progress objective.
- `topology_world_model`: proposed method; predicts contact births/deaths, graph components, and jam/slip modes, then plans actions to reach useful graph states.
- `oracle_topology_planner`: upper bound with access to true next-step topology from MuJoCo rollouts.

## Metrics

- Task success.
- Next-step contact-edge F1.
- Contact birth/death F1.
- Graph edit distance.
- Connected-component accuracy.
- Jam/slip event F1.
- Planning regret against oracle.
- Safety/collision violation rate.
- Tail failure rate under combined stress.

## Experimental Rigor

- Use seven random seeds unless runtime becomes impossible.
- Use held-out geometries, masses, frictions, and initial object arrangements.
- Report mean, 95 percent confidence intervals, and paired comparisons against the strongest non-oracle baseline.
- Include ablations: no graph-transition loss, no birth/death head, no component head, no jam/slip head, no topology-aware planner, no uncertainty penalty.
- Include stress sweeps over geometry perturbation, friction, mass, contact-sensor noise, actuator limit, and distractor contacts.
- Save raw per-rollout graph traces, per-seed summaries, pairwise statistics, ablation tables, stress tables, and negative cases.

## Submission Gate

The paper can only move above archive if `topology_world_model` beats the best non-oracle baseline on `combined_stress` by a meaningful paired success margin, improves contact birth/death F1, reduces graph edit distance, and does not increase safety violations. If a state-only learned model, contact-implicit MPC baseline, or ensemble planner matches or beats it, the paper remains `KILL_ARCHIVE` or at best `STRONG_REVISE`.

## Deliverables

- Replace the synthetic scaffold with a reproducible MuJoCo contact-topology benchmark runner.
- Generate raw topology traces, metrics, pairwise statistics, ablations, stress sweeps, negative cases, and figures.
- Rewrite README, claims, novelty boundary, hostile review, reproducibility checklist, final audit, and ICLR gate around the actual evidence.
- Rewrite `paper/main.tex` as either a real negative-result paper or a submission-candidate manuscript.
- Compile `paper/main.pdf`, copy exactly to `C:/Users/wangz/Downloads/73.pdf`, and do not copy any PDF to Desktop.
- Commit and push the final Paper 73 repo, then update shared root reports before moving to Paper 74.
