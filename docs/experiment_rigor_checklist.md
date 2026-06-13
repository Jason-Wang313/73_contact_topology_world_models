# Experiment Rigor Checklist

## v4 Evidence

- [x] MuJoCo tabletop contact-topology benchmark.
- [x] Seven random seeds.
- [x] Eight evaluation episodes per seed/split.
- [x] Implemented persistence and distance-threshold baselines.
- [x] Implemented state-only dynamics baseline.
- [x] Implemented pairwise contact classifier.
- [x] Implemented ensemble uncertainty planner.
- [x] Implemented contact-implicit MPC-style baseline.
- [x] Implemented proposed topology world model.
- [x] Implemented oracle topology planner.
- [x] Paired comparisons against the proposed method.
- [x] Topology ablations.
- [x] Stress sweeps.
- [x] Negative-case extraction.
- [x] Paper-specific figures.

## ICLR Main Bar

- [ ] Proposed method beats the strongest non-oracle baseline on task success.
- [ ] Ablations support the central topology mechanism.
- [ ] Real-robot validation.
- [ ] External public benchmark validation.
- [ ] Manual full-paper related-work synthesis deep enough for submission.
- [ ] Qualitative rollouts or videos.

Decision: fail ICLR-main empirical-rigor gate; archive.
