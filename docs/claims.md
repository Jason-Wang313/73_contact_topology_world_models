# Claims

- Mechanism claim tested: contact-rich world models should predict contact graph topology changes, not only continuous state trajectories.
- Evidence claim: a seven-seed MuJoCo tabletop benchmark compares topology prediction/planning against persistence, distance, state-only dynamics, pairwise contact, ensemble uncertainty, contact-implicit, and oracle baselines.
- Result claim: the current topology world model fails the main gate; it reaches 0.107 combined-stress success versus 0.125 for the strongest non-oracle baseline.
- Diagnostic claim: topology prediction improves edge F1 over the ensemble uncertainty planner, but this does not produce a task-success gain.
- Scope claim: the repository is a reproducible negative-result archive, not an ICLR-main-ready submission.
- Unsupported claim explicitly avoided: no claim of state-of-the-art contact-rich manipulation.
