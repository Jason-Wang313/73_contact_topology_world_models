# Novelty Decision

Chosen thesis: contact-rich world models should predict topology changes in the contact graph rather than only state trajectories.

v4 decision: KILL_ARCHIVE.

The thesis was tested in a real local MuJoCo benchmark. The current topology model improves some graph diagnostics, but it does not improve combined-stress task success over the strongest non-oracle baseline. The result is useful as a negative finding: contact-topology prediction alone is not sufficient unless it changes action selection.
