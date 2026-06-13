# 73 Contact Topology World Models

Submission-hardening version: v4

Terminal decision: KILL_ARCHIVE for ICLR main conference.

This repository now contains a real Paper 73 rebuild: a MuJoCo tabletop contact-topology benchmark, implemented topology/state/distance/contact-implicit baselines, a proposed topology world model, an oracle topology planner, seven-seed evaluation, uncertainty intervals, ablations, stress sweeps, negative cases, figures, and a rewritten archive manuscript.

The evidence does not support ICLR-main submission. On the decisive `combined_stress` split, `topology_world_model` reaches 0.107 +/- 0.083 task success, while the strongest non-oracle baseline, `ensemble_uncertainty_planner`, reaches 0.125 +/- 0.076. The paired success difference is -0.018 +/- 0.035. The topology model improves edge F1 over the ensemble baseline but does not convert that into task success.

## Main Result

Full run:

- Main evaluation rows: 2240.
- Ablation rows: 252.
- Stress rows: 1050.
- Seeds: 0 through 6.
- Episodes per seed and split: 8.
- Runtime: 3927.52 seconds.

Combined-stress summary:

- `oracle_topology_planner`: 0.250 +/- 0.076 success, edge F1 1.000.
- `ensemble_uncertainty_planner`: 0.125 +/- 0.076 success, edge F1 0.364.
- `topology_world_model`: 0.107 +/- 0.083 success, edge F1 0.610.
- `state_only_dynamics_model`: 0.107 +/- 0.083 success, edge F1 0.904.
- `contact_implicit_mpc_baseline`: 0.107 +/- 0.083 success, edge F1 0.861.

The paper is retained as a reproducible negative-result archive.

## Reproduce

```powershell
python src\run_experiment.py
```

Outputs are written under `results/` and `figures/`.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/73.pdf`

No PDF is copied to the visible Desktop.
