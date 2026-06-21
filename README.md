# 73 Contact Topology World Models

Submission-hardening version: v5 expanded submission audit

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository contains the expanded Paper 73 rebuild: a CPU-only MuJoCo tabletop contact-topology benchmark with 12 evaluation splits, 15 main methods, 12 ablations, 12 stress methods, fixed-risk metrics, paired seed statistics, generated evidence appendices, and a 36-page ICLR-style archive manuscript.

The evidence does not support submission. On the decisive combined/extreme aggregate, `topology_world_model_v5` reaches 0.02083 +/- 0.02790 task success while the strongest non-oracle baseline, `pairwise_contact_classifier`, reaches 0.05208 +/- 0.04917. On the hard-split aggregate, v5 reaches 0.05871 +/- 0.02304 while `pairwise_contact_classifier` reaches 0.09470 +/- 0.03331. The fixed-risk gate also fails at budget 0.10, and most removed-component ablations match or beat full v5.

## Final Evidence Scale

- Main evaluation rows: 8640.
- Ablation rows: 1536.
- Stress rows: 4320.
- Training scenes: 2400.
- Seeds: 0 through 7.
- Main episodes per seed/split: 6.
- Runtime: 15319.06 seconds.
- Final PDF: `C:/Users/wangz/Downloads/73.pdf`.
- Final PDF pages: 36.
- Final PDF SHA256: `BEAAE8EAD6491D78CBE9C4FB764BA12D8DA653E48D20A3290B03D2E5824A6E9D`.

## Reproduce

```powershell
python src\run_experiment.py --seeds 8 --episodes 6 --ablation-episodes 4 --stress-episodes 3 --train-scenes 2400 --splits nominal_push_to_pocket contact_chain_transfer fixture_topology_shift friction_mass_shift pocket_relocation fixture_near_wall_jam distractor_contact contact_sensor_noise_burst actuator_limit_chain delayed_topology_transition combined_stress combined_extreme_stress --ablation-splits combined_stress combined_extreme_stress fixture_near_wall_jam actuator_limit_chain --stress-splits combined_stress combined_extreme_stress fixture_topology_shift --stress-levels 0.0 0.25 0.5 0.75 1.0 --results-dir results --figures-dir figures --workers 1
```

## Rebuild PDF

```powershell
python scripts\generate_manuscript.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Validate

```powershell
python scripts\validate_submission_artifacts.py
```

No PDF is copied to the visible Desktop.
