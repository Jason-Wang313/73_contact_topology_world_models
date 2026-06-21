# Paper 73 Terminal Audit

Date: 2026-06-21

Terminal decision: KILL_ARCHIVE

PDF: `C:/Users/wangz/Downloads/73.pdf`

SHA256: `BEAAE8EAD6491D78CBE9C4FB764BA12D8DA653E48D20A3290B03D2E5824A6E9D`

Pages: 36

## Verification

- Frozen full run completed with empty stderr.
- Row counts matched the protocol: 8640 main, 1536 ablation, 4320 stress, 2400 training scenes.
- `scripts/validate_submission_artifacts.py` passed.
- Final LaTeX hard scan was clean for overfull boxes, undefined citations/references, rerun warnings, table-width warnings, and longtable split errors.
- Visual QA inspected pages 1, 2, 4, 6, 10, 20, and 36.
- Bright green citation boxes and blue URL boxes were visible in the rendered PDF.
- `C:/Users/wangz/Desktop/73.pdf` was absent.

## Scientific Outcome

The expanded audit strengthens the negative conclusion. `topology_world_model_v5` improves neither hard-regime nor combined/extreme downstream success over the strongest non-oracle baseline. Fixed-risk, ablation, paired, and oracle-sanity gates all prevent an ICLR-main submission claim.
