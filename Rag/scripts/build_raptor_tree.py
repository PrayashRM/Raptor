# scripts/build_raptor_tree.py
"""
RAPTOR tree construction entry point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from raptor.pipeline import build_raptor_tree
from core.logger import get_logger
import config

logger = get_logger("raptor_script")

# ── Configure directly here ────────────────────────────────────────────────────

# Option A: Single paper — set paper_id string
PAPER_ID = "attention"

# Set True to force complete rebuild even if tree already exists
FORCE_REBUILD = True

# Option B: All papers — set PAPER_ID = None
# PAPER_ID = None

# ──────────────────────────────────────────────────────────────────────────────


def main():
    if PAPER_ID is not None:
        paper_ids = [PAPER_ID]
    else:
        parsed_dir = config.PARSED_DIR
        paper_ids  = [
            d.name for d in parsed_dir.iterdir()
            if d.is_dir() and (d / "elements.json").exists()
        ]
        if not paper_ids:
            print(f"No papers found in {parsed_dir}")
            sys.exit(1)

    print(f"\nBuilding RAPTOR tree for {len(paper_ids)} paper(s)...\n")

    reports = []
    for paper_id in paper_ids:
        try:
            report = build_raptor_tree(
                paper_id,
                force_rebuild=FORCE_REBUILD,    #passing force_rebuild to build_raptor_tree function
            )
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed for '{paper_id}': {e}")
            reports.append({
                "paper_id": paper_id,
                "status":   "FAILED",
                "error":    str(e),
            })

    print("\n" + "=" * 60)
    print("RAPTOR TREE BUILD SUMMARY")
    print("=" * 60)
    for r in reports:
        status_icon = "✅" if r.get("status") == "SUCCESS" else "❌"
        print(f"{status_icon} {r['paper_id']}")
        if r.get("status") == "SUCCESS":
            print(
                f"   L0:{r['leaf_chunks']} → "
                f"L1:{r['l1_nodes']} → "
                f"L2:{r['l2_nodes']} → "
                f"L3:1  |  {r['duration_seconds']}s"
            )
        else:
            print(f"   ERROR: {r.get('error', 'unknown')}")

    failed = sum(1 for r in reports if r.get("status") != "SUCCESS")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()