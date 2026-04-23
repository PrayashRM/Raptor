# scripts/inspect_qdrant.py
"""
Inspect everything stored in Qdrant.
Run from project root:
    python scripts/inspect_qdrant.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.storage import QdrantStorage
from qdrant_client.models import Filter, FieldCondition, MatchValue
import config

storage = QdrantStorage()
client  = storage.client

# ── 1. Collection overview ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("QDRANT COLLECTION OVERVIEW")
print("="*60)

collections = client.get_collections().collections
for col in collections:
    info = client.get_collection(col.name)
    count = client.count(col.name).count
    print(f"\nCollection: {col.name}")
    print(f"  Total points:  {count}")
    print(f"  Vector config: {info.config.params.vectors}")

# ── 2. Per-paper breakdown ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("PER-PAPER BREAKDOWN")
print("="*60)

# Get all unique paper_ids
all_points, _ = client.scroll(
    collection_name = config.QDRANT_COLLECTION,
    with_payload    = ["paper_id", "level", "chunk_type",
                       "section_title", "token_count",
                       "chunk_id"],
    with_vectors    = False,
    limit           = 10_000,
)

# Group by paper_id
from collections import defaultdict
papers = defaultdict(lambda: defaultdict(list))

for point in all_points:
    p        = point.payload
    paper_id = p.get("paper_id", "unknown")
    level    = p.get("level", -1)
    papers[paper_id][level].append(p)

for paper_id, levels in papers.items():
    print(f"\nPaper: {paper_id}")
    total = sum(len(v) for v in levels.values())
    print(f"  Total nodes: {total}")

    for level in sorted(levels.keys()):
        nodes     = levels[level]
        level_name = {
            0: "L0 Leaf chunks",
            1: "L1 Section summaries",
            2: "L2 Group summaries",
            3: "L3 Root",
        }.get(level, f"L{level} Unknown")

        print(f"\n  {level_name}: {len(nodes)} nodes")

        for node in nodes[:3]:  # show first 3 of each level
            print(
                f"    [{node.get('chunk_id', '?')}] "
                f"section='{node.get('section_title', '?')[:40]}' "
                f"tokens={node.get('token_count', '?')} "
                f"type={node.get('chunk_type', '?')}"
            )
        if len(nodes) > 3:
            print(f"    ... and {len(nodes) - 3} more")

# ── 3. Detailed view of specific level ────────────────────────────────────────
print("\n" + "="*60)
print("L1 SUMMARY NODES (full text preview)")
print("="*60)

l1_nodes, _ = client.scroll(
    collection_name = config.QDRANT_COLLECTION,
    scroll_filter   = Filter(
        must=[
            FieldCondition(
                key   = "level",
                match = MatchValue(value=1),
            )
        ]
    ),
    with_payload = True,
    with_vectors = False,
    limit        = 100,
)

for node in l1_nodes:
    p = node.payload
    print(f"\n  chunk_id:  {p.get('chunk_id')}")
    print(f"  section:   {p.get('section_title')}")
    print(f"  tokens:    {p.get('token_count')}")
    print(f"  children:  {p.get('children_ids', [])}")
    print(f"  parents:   {p.get('parent_ids', [])}")
    text = p.get('text_for_embedding', '')
    print(f"  summary:   {text[:200]}...")

# ── 4. L3 Root ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("L3 ROOT NODE")
print("="*60)

l3_nodes, _ = client.scroll(
    collection_name = config.QDRANT_COLLECTION,
    scroll_filter   = Filter(
        must=[
            FieldCondition(
                key   = "level",
                match = MatchValue(value=3),
            )
        ]
    ),
    with_payload = True,
    with_vectors = False,
    limit        = 10,
)

for node in l3_nodes:
    p = node.payload
    print(f"\n  chunk_id:  {p.get('chunk_id')}")
    print(f"  children:  {p.get('children_ids')}")
    print(f"  tokens:    {p.get('token_count')}")
    text = p.get('text_for_embedding', '')
    print(f"  root summary:\n")
    print(f"  {text}")

# ── 5. Tree link integrity check ──────────────────────────────────────────────
print("\n" + "="*60)
print("TREE LINK INTEGRITY")
print("="*60)

# Build id → payload map
id_map = {
    p.payload.get("chunk_id"): p.payload
    for p in all_points
    if p.payload.get("chunk_id")
}

broken_links  = []
orphan_leaves = []

for point in all_points:
    p        = point.payload
    chunk_id = p.get("chunk_id")
    level    = p.get("level", -1)

    # Check children exist
    for child_id in p.get("children_ids", []):
        if child_id not in id_map:
            broken_links.append(
                f"{chunk_id} → missing child {child_id}"
            )

    # Check leaf nodes have parents
    if level == 0 and not p.get("parent_ids"):
        orphan_leaves.append(chunk_id)

if broken_links:
    print(f"  BROKEN LINKS: {len(broken_links)}")
    for link in broken_links[:5]:
        print(f"    {link}")
else:
    print(f"  Broken links:  0 ✅")

if orphan_leaves:
    print(f"  Orphan leaves: {len(orphan_leaves)}")
    for oid in orphan_leaves[:5]:
        print(f"    {oid}")
else:
    print(f"  Orphan leaves: 0 ✅")

print(f"\n  Total nodes checked: {len(all_points)}")
print(f"  Unique papers:       {len(papers)}")
print()