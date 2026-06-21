from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOWNLOADS = Path.home() / "Downloads"
DESKTOP = Path.home() / "Desktop"
PDF = DOWNLOADS / "73.pdf"


EXPECTED_COUNTS = {
    "topology_raw.csv": 8640,
    "topology_rollouts.csv": 8640,
    "raw_seed_metrics.csv": 1440,
    "metrics.csv": 180,
    "topology_metrics.csv": 180,
    "pairwise_stats.csv": 168,
    "topology_pairwise.csv": 168,
    "aggregate_metrics.csv": 45,
    "aggregate_pairwise_stats.csv": 42,
    "fixed_risk_metrics.csv": 45,
    "topology_ablation_raw.csv": 1536,
    "ablation_metrics.csv": 48,
    "topology_ablation.csv": 48,
    "ablation_aggregate_metrics.csv": 12,
    "stress_sweep_raw.csv": 4320,
    "stress_sweep.csv": 180,
    "negative_cases.csv": 12,
    "training_topology_examples.csv": 2400,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            return len(PdfReader(str(path)).pages)
        except Exception:
            result = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    return int(line.split(":", 1)[1].strip())
            raise RuntimeError("could not determine PDF page count")


def main() -> None:
    for name, expected in EXPECTED_COUNTS.items():
        path = RESULTS / name
        require(path.exists(), f"missing {path}")
        rows = read_rows(path)
        require(len(rows) == expected, f"{name} has {len(rows)} rows, expected {expected}")

    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    require("Terminal decision:" in summary, "summary.txt lacks terminal decision")
    require("Main eval rows: 8640" in summary, "summary.txt lacks final main row count")
    require("Ablation rows: 1536" in summary, "summary.txt lacks final ablation row count")
    require("Stress rows: 4320" in summary, "summary.txt lacks final stress row count")

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    require("citebordercolor={0 1 0}" in tex, "bright citation boxes are not configured")
    require("pdfborder={0 0 1.6}" in tex, "PDF border width is not configured")
    require("topology\\_world\\_model\\_v5" in tex or "topo-v5" in tex, "v5 method is absent from manuscript")

    require(PDF.exists(), f"missing Downloads PDF {PDF}")
    require(not (DESKTOP / "73.pdf").exists(), "Desktop copy of 73.pdf exists")
    pages = pdf_page_count(PDF)
    require(pages >= 25, f"PDF has {pages} pages, expected at least 25")

    digest = hashlib.sha256(PDF.read_bytes()).hexdigest().upper()
    print(f"validated Paper 73 artifacts: pages={pages}, sha256={digest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise
