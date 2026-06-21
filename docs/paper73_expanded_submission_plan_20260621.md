# Paper 73 Expanded Submission Plan

Paper: `contact_topology_world_models`

Target venue standard: ICLR main-conference hostile review

Execution mode: CPU-only, RAM-light, plan first, freeze before final run

Terminal policy: do not optimize for pretty results. Optimize for evidence that survives hostile review.

## 1. Core Scientific Question

The paper asks whether an explicit contact-topology world model improves contact-rich manipulation decisions, not merely whether it predicts contact graph edges.

The proposed mechanism is only submission-relevant if the topology representation changes action selection enough to beat strong state, geometric, learned, uncertainty, conformal, and contact-implicit baselines on downstream control.

Graph metrics are diagnostic, not the acceptance criterion. A method that improves edge F1 but loses task success remains `KILL_ARCHIVE`.

## 2. Minimum Expanded Scope

The current v4 archive has 2,240 main rows, five splits, eight methods, one stress split, and a short manuscript. The v5 rebuild must expand it to a legitimate submission-scale negative or revise archive:

- At least 12 evaluation splits.
- At least 15 main methods including strong learned and risk-aware baselines.
- 8 random seeds.
- 6 main episodes per seed/split.
- 4 ablation episodes per seed/split.
- 3 stress episodes per seed/split/level.
- 2,400 CPU-light training scenes.
- A 25+ page manuscript with theory, frozen gates, generated tables, full appendices, bright boxed clickable citations, and no Desktop PDF copy.

Expected frozen evidence scale:

- Main rows: 12 splits x 15 methods x 8 seeds x 6 episodes = 8,640.
- Ablation rows: 4 splits x 12 ablations x 8 seeds x 4 episodes = 1,536.
- Stress rows: 3 stress splits x 12 stress methods x 5 levels x 8 seeds x 3 episodes = 4,320.

## 3. Expanded Benchmark Splits

The final runner must evaluate both graph-prediction and downstream-control failure modes:

- `nominal_push_to_pocket`: simple pusher-block-pocket chain.
- `contact_chain_transfer`: pushing block A through block B into pocket.
- `fixture_topology_shift`: fixture changes contact birth order.
- `friction_mass_shift`: friction and mass alter sticking/sliding.
- `pocket_relocation`: target pocket shifts laterally.
- `fixture_near_wall_jam`: fixture-wall interactions create jammed components.
- `distractor_contact`: irrelevant contacts tempt false graph explanations.
- `contact_sensor_noise_burst`: observed contacts flip in bursts.
- `actuator_limit_chain`: progress requires planning under weak actuation.
- `delayed_topology_transition`: useful contact graph appears late.
- `combined_stress`: geometry, friction, noise, limits, and distractors combined.
- `combined_extreme_stress`: harsher version of combined stress.

## 4. Main Methods

The expanded method set must make the proposed model uncomfortable:

- `last_contact_persistence`: previous graph as next graph.
- `distance_threshold_graph`: contact edges from distances and margins.
- `state_only_dynamics_model`: learned state delta with induced contacts.
- `pairwise_contact_classifier`: supervised pairwise contact classifier.
- `random_forest_topology_planner`: nonparametric learned graph predictor.
- `hist_gradient_topology_planner`: boosted tabular graph predictor.
- `ensemble_uncertainty_planner`: conservative ensemble baseline.
- `conformal_graph_guard`: learned graph predictor with calibrated residual guard.
- `risk_averse_graph_planner`: prioritizes lower fixture/wall/jam risk.
- `robust_contact_mpc`: geometric contact-implicit MPC-style baseline.
- `contact_implicit_mpc_baseline`: v4 contact-implicit baseline.
- `topology_world_model_v4`: frozen prior mechanism replay.
- `topology_world_model_v5`: proposed expanded mechanism.
- `topology_no_memory_ablation`: no graph-transition memory.
- `oracle_topology_planner`: upper-bound diagnostic with true simulated topology.

## 5. Ablation Suite

Ablations must test mechanism necessity rather than decorate the paper:

- `topology_full_v5`
- `ablate_no_birth_death`
- `ablate_no_component_head`
- `ablate_no_jam_slip`
- `ablate_no_topology_planner`
- `ablate_no_uncertainty_penalty`
- `ablate_no_graph_memory`
- `ablate_no_transition_bonus`
- `ablate_no_fixture_guard`
- `ablate_no_tail_risk_objective`
- `topology_world_model_v4`
- `learned_only_topology_replacement`

Full v5 must beat every removed-component ablation by at least 0.020 success or be clearly safer at the same success level. If not, the mechanism is not locally supported.

## 6. Metrics

Primary metrics:

- Task success.
- Final progress.
- Final target lateral error.
- Safety violation rate.
- Fixture contact rate.
- Wall contact rate.
- Jam F1.

Topology diagnostics:

- Contact-edge F1.
- Birth F1.
- Death F1.
- Graph edit distance.
- Connected-component accuracy.

Submission diagnostics:

- Paired seed differences.
- Aggregate hard-regime metrics.
- Aggregate combined/extreme metrics.
- Fixed-risk success at diagnostic-risk budgets 0.05, 0.10, and 0.20.
- Stress-sweep success at maximum stress.
- Negative cases with lessons.

## 7. Frozen Submission Gates

The final decision is computed from CSVs, not hand interpretation.

`topology_world_model_v5` can only avoid `KILL_ARCHIVE` if all gates pass:

- Hard aggregate: v5 beats the strongest non-oracle baseline by at least 0.030 success.
- Paired gate: paired lower bound against the strongest combined/extreme non-oracle baseline is positive.
- Combined/extreme gate: v5 beats the strongest non-oracle baseline by at least 0.030 success.
- Diagnostic gate: v5 cannot trade task success for worse graph edit, fixture contact, wall contact, jam, or safety by more than 0.020.
- Fixed-risk gate: v5 must not be below the best non-oracle method at the 0.10 diagnostic-risk budget.
- Maximum-stress gate: v5 must be within 0.030 of the best non-oracle method at stress level 1.00.
- Ablation gate: all removed components must underperform full v5 by at least 0.020 success or be clearly worse on safety/jam diagnostics.
- Oracle sanity gate: if the oracle cannot achieve meaningful success on hard regimes, the benchmark calibration must be reported as a limitation and cannot rescue the method.

If any gate fails, report the failure honestly and archive or strong-revise accordingly.

## 8. Development Discipline

Development probes may be used only to debug runtime, shape errors, missing files, and obviously broken gate logic. They must not be used to tune final results after looking at the frozen outcome.

Before the full run:

- Record this plan.
- Record a development log.
- Record the exact frozen command.
- Freeze splits, seeds, methods, episodes, stress levels, metrics, and gates.

After the full run:

- Do not rerun with changed gates.
- Do not hide negative results.
- Generate all tables from CSVs.
- Build `C:/Users/wangz/Downloads/73.pdf`.
- Validate row counts, page count, citation boxes, repo URL, and Desktop hygiene.
- Render representative PDF pages for visual QA.
- Commit, push, verify public GitHub, then update root ledgers.

## 9. Expected Manuscript

The manuscript should be at least 25 pages without filler:

- Main text: question, benchmark, method, baselines, protocol, results, fixed-risk analysis, stress sweep, ablations, negative cases, theory, related work, decision.
- Theory: graph-state identifiability, diagnostic-vs-control non-identifiability, topology memory failure, and ablation necessity.
- Appendix: full split-method metrics, aggregate metrics, all seed metrics, paired comparisons, ablations, stress sweep, and negative cases.
- Citations: real verified references; in-text citation boxes must be bright and clickable.

## 10. Success Definition for This Pass

This pass succeeds if the final archive is honest, reproducible, visually checked, pushed publicly, and reflected in root ledgers. It does not require the method to look good.

The most likely scientific outcome is still `KILL_ARCHIVE`: the v4 evidence already shows topology diagnostics failing to become control success. The v5 task is to test that failure much harder, not to decorate it.
