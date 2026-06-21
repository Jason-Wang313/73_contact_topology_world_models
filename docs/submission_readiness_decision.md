# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Submission-hardening version: v5 expanded submission audit.

## Evidence Used

The v5 rebuild expands the prior local MuJoCo archive into a hostile-review protocol with 12 splits, 15 main methods, 12 ablations, 12 stress methods, 8 seeds, fixed-risk metrics, paired statistics, generated appendix tables, and a 36-page ICLR-style manuscript.

## Gate Result

The frozen gates fail:

- Hard-regime margin: v5 success 0.05871 vs `pairwise_contact_classifier` 0.09470.
- Combined/extreme margin: v5 success 0.02083 vs `pairwise_contact_classifier` 0.05208.
- Paired lower bound: not positive against the strongest combined/extreme non-oracle baseline.
- Fixed risk: v5 success at budget 0.10 is 0.000 vs best non-oracle 0.013.
- Ablations: most removed-component variants match or beat full v5.
- Oracle sanity: hard-regime oracle success is only 0.140.

## Terminal Action

Archive/kill for ICLR main. Do not submit this paper as an ICLR main paper.

Revival condition: demonstrate that contact-topology prediction improves downstream success over strong non-oracle contact-rich control baselines on calibrated public or real-robot benchmarks, not merely graph diagnostics.
