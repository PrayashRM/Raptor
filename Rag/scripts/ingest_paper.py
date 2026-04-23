# scripts/ingest_paper.py
"""
Ingestion entry point.

Usage:
    python scripts/ingest_paper.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pipeline import ingest_paper, ingest_batch
from core.logger import get_logger

logger = get_logger("ingest_script")

# ── Configure paths directly here ─────────────────────────────────────────────

# Option A: Single paper — set path to the paper directory
PAPER_DIR = Path("data/parsed/attention_is_all_you_need_2017")

# Option B: Batch — set path to parent directory containing all papers
# Set PAPER_DIR = None and set INPUT_DIR instead
INPUT_DIR = None

# ──────────────────────────────────────────────────────────────────────────────


def main():
    if PAPER_DIR is not None:
        report  = ingest_paper(PAPER_DIR)
        reports = [report]

    elif INPUT_DIR is not None:
        reports = ingest_batch(INPUT_DIR)

    else:
        print("ERROR: Set either PAPER_DIR or INPUT_DIR at top of script.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for r in reports:
        status_icon = "✅" if r.status == "SUCCESS" else "❌"
        print(
            f"{status_icon} {r.paper_id}: "
            f"{r.total_chunks_created} chunks, "
            f"{r.ingestion_duration_seconds}s"
        )
        if r.errors:
            for err in r.errors:
                print(f"   ERROR: {err}")

    failed = sum(1 for r in reports if r.status == "FAILED")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()