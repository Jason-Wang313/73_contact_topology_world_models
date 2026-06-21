# Final Audit

1. Chosen thesis: explicit contact-topology world models should improve downstream contact-chain control, not merely contact-edge prediction.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v5 expanded submission audit.
4. Evidence: MuJoCo contact-topology benchmark with 8640 main rows, 1536 ablation rows, 4320 stress rows, 2400 training scenes, 8 seeds, 12 splits, 15 main methods, fixed-risk metrics, paired statistics, figures, generated appendix tables, and negative cases.
5. Combined/extreme result: `topology_world_model_v5` reaches 0.02083 +/- 0.02790 success; `pairwise_contact_classifier` reaches 0.05208 +/- 0.04917.
6. Hard-split result: `topology_world_model_v5` reaches 0.05871 +/- 0.02304 success; `pairwise_contact_classifier` reaches 0.09470 +/- 0.03331.
7. Paired gate: paired lower bound against `pairwise_contact_classifier` is not positive.
8. Fixed-risk gate: at diagnostic-risk budget 0.10, v5 reaches 0.000 success while the best non-oracle method reaches 0.013.
9. Ablation gate: most removed-component variants match or beat full v5, so mechanism necessity is not identified.
10. Oracle sanity: hard-regime oracle success is 0.140, too weak to rescue the benchmark claim.
11. Claim-validity status: main-conference claims killed; reproducible negative-result archive retained.
12. Exact Downloads PDF path: `C:/Users/wangz/Downloads/73.pdf`
13. Final PDF pages: 36.
14. Final PDF SHA256: `BEAAE8EAD6491D78CBE9C4FB764BA12D8DA653E48D20A3290B03D2E5824A6E9D`
15. GitHub URL: https://github.com/Jason-Wang313/73_contact_topology_world_models
16. Confirmation: no visible Desktop copy was made.
