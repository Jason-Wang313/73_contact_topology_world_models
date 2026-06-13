# Plan

Paper 73 was rebuilt as a real MuJoCo contact-topology study before terminal packaging.

1. Implement a tabletop contact benchmark with a pusher, two movable blocks, a pocket, walls, and a fixture.
2. Log contact graph edges, births/deaths, graph edit distance, connected components, jam/safety events, task progress, and rollouts.
3. Compare persistence, distance-threshold, state-dynamics, pairwise classifier, ensemble uncertainty, contact-implicit MPC, topology world model, and oracle topology planners.
4. Run seven-seed main evaluation, topology ablations, stress sweeps, uncertainty intervals, pairwise tests, and negative-case extraction.
5. Decide the ICLR-main gate from evidence.
6. Package the archive manuscript, numbered Downloads PDF, and public GitHub repo.

Outcome: KILL_ARCHIVE. Contact topology prediction does not produce a decisive task-success advantage over the strongest non-oracle baseline.
