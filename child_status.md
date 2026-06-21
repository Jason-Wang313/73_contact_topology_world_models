# Child Status 73

Current stage: v5 expanded submission audit terminal
Last update: 2026-06-21 10:41:36 +08:00
PDF: C:/Users/wangz/Downloads/73.pdf
GitHub: https://github.com/Jason-Wang313/73_contact_topology_world_models
Submission-hardening version: v5-expanded
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Evidence: expanded CPU-only MuJoCo contact-topology benchmark with 8640 main rows, 1536 ablation rows, 4320 stress rows, 2400 training scenes, 8 seeds, fixed-risk metrics, paired seed statistics, stress sweeps, negative cases, and generated evidence appendices.

Gate failures: `topology_world_model_v5` does not beat `pairwise_contact_classifier` on hard or combined/extreme success; paired lower bound is not positive; fixed-risk success at budget 0.10 is lower than the best non-oracle baseline; most ablations match or beat full v5; oracle hard-regime success is weak at 0.140.

Final PDF: 36 pages, SHA256 `BEAAE8EAD6491D78CBE9C4FB764BA12D8DA653E48D20A3290B03D2E5824A6E9D`, stored in Downloads only.
