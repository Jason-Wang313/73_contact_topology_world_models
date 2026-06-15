# Plan

Paper 73 was rebuilt as a real MuJoCo contact-topology study before terminal packaging.

1. Implement a tabletop contact benchmark with a pusher, two movable blocks, a pocket, walls, and a fixture.
2. Log contact graph edges, births/deaths, graph edit distance, connected components, jam/safety events, task progress, and rollouts.
3. Compare persistence, distance-threshold, state-dynamics, pairwise classifier, ensemble uncertainty, contact-implicit MPC, topology world model, and oracle topology planners.
4. Run seven-seed main evaluation, topology ablations, stress sweeps, uncertainty intervals, pairwise tests, and negative-case extraction.
5. Decide the ICLR-main gate from evidence.
6. Package the archive manuscript, numbered Downloads PDF, and public GitHub repo.

Outcome: KILL_ARCHIVE. Contact topology prediction does not produce a decisive task-success advantage over the strongest non-oracle baseline.

## 2026-06-15 Continuation Plan

1. Re-audit the real MuJoCo contact-topology evidence before making any submission-readiness claim.
2. Confirm the experiment source compiles and all raw CSVs are present, finite, and at the claimed scale.
3. Rebuild the PDF, repair recoverable LaTeX/BibTeX issues, and copy only `73.pdf` to Downloads.
4. Preserve `KILL_ARCHIVE` unless topology prediction produces a decisive downstream task-success gain over the strongest non-oracle baseline.
5. Update child docs, root reports, and GitHub state before moving to Paper 74.
