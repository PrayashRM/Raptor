# scripts/ingest_paper.py
"""
CLI entry point for ingestion.

Usage:
    python scripts/ingest_paper.py --input_dir data/parsed/
    python scripts/ingest_paper.py --paper_dir data/parsed/attention_2017/
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_Phase2.pipeline import ingest_paper, ingest_batch
from core.logger import get_logger

logger = get_logger("ingest_script")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest parsed research papers into RAPTOR RAG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--paper_dir",
        type=Path,
        help="Path to single paper directory containing elements.json",
    )
    group.add_argument(
        "--input_dir",
        type=Path,
        help="Path to directory containing multiple paper subdirectories",
    )
    args = parser.parse_args()

    if args.paper_dir:
        report = ingest_paper(args.paper_dir)
        reports = [report]
    else:
        reports = ingest_batch(args.input_dir)

    # Print summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
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