# Paper 73 Protocol Freeze

Date: 2026-06-21

Paper: `contact_topology_world_models`

Frozen protocol status: active before full run.

## Final Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal_push_to_pocket contact_chain_transfer fixture_topology_shift friction_mass_shift pocket_relocation fixture_near_wall_jam distractor_contact contact_sensor_noise_burst actuator_limit_chain delayed_topology_transition combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress fixture_near_wall_jam actuator_limit_chain --stress-splits combined_stress combined_extreme_stress fixture_topology_shift --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

## Frozen Scale

- Main rows: 12 splits x 15 methods x 8 seeds x 6 episodes = 8,640.
- Ablation rows: 4 splits x 12 ablations x 8 seeds x 4 episodes = 1,536.
- Stress rows: 3 stress splits x 12 stress methods x 5 stress levels x 8 seeds x 3 episodes = 4,320.
- Training scenes: 2,400.
- Execution: CPU-only, single process, RAM-light.

## Frozen Main Splits

- `nominal_push_to_pocket`
- `contact_chain_transfer`
- `fixture_topology_shift`
- `friction_mass_shift`
- `pocket_relocation`
- `fixture_near_wall_jam`
- `distractor_contact`
- `contact_sensor_noise_burst`
- `actuator_limit_chain`
- `delayed_topology_transition`
- `combined_stress`
- `combined_extreme_stress`

## Frozen Gates

`topology_world_model_v5` can avoid `KILL_ARCHIVE` only if every gate passes:

- Hard-regime success margin at least 0.030 over the strongest non-oracle baseline.
- Positive paired lower bound against the strongest combined/extreme non-oracle baseline.
- Combined/extreme success margin at least 0.030 over the strongest non-oracle baseline.
- No diagnostic regression above 0.020 in graph edit, safety, fixture contact, or wall contact.
- Fixed-risk success at budget 0.10 not below the best non-oracle method.
- Maximum-stress success within 0.030 of the best non-oracle method averaged over frozen max-stress splits.
- Removed-component ablations must underperform full v5 by at least 0.020 success or be clearly worse on safety.
- Oracle hard-regime sanity must exceed 0.200 success.

## Reporting Rule

The final manuscript must report all predefined results honestly. If any gate fails, the paper is written as a strengthened negative archive or strong-revise artifact, not as a polished success story.
