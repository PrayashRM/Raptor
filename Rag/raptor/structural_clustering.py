# raptor/structural_clustering.py
"""
Layer 1A: Section-header-based structural clustering.

Groups leaf chunks by section hierarchy.
This is deterministic, zero-ML, and gives us k
for FCM without needing BIC or elbow method.

Rules:
- Group by top-level section title
- Merge groups with < 3 chunks into nearest neighbor
- Split groups with > 12 chunks at subsection boundary
  or at equal halves if no subsection exists
- Output: list of StructuralCluster objects
"""

from __future__ import annotations
from dataclasses import dataclass, field
from core.logger import get_logger
from core.utils import normalize_section_title

logger = get_logger(__name__)

MIN_CLUSTER_SIZE = 3
MAX_CLUSTER_SIZE = 12


@dataclass
class StructuralCluster:
    cluster_id:    int
    section_title: str
    section_hierarchy: list[str]
    chunk_ids:     list[str]
    chunk_indices: list[int]   # indices into the master chunks list
    is_split:      bool = False
    split_part:    int  = 0


def build_structural_clusters(
    chunks: list[dict],
) -> list[StructuralCluster]:
    """
    Group chunks into structural clusters based on section hierarchy.

    Args:
        chunks: List of Qdrant scroll results (payload dicts)

    Returns:
        List of StructuralCluster, ordered by reading order.
        Length of this list = k for FCM.
    """
    if not chunks:
        raise ValueError("Cannot cluster empty chunk list")

    # ── Step 1: Group by top-level section ────────────────────────────────
    section_groups: dict[str, list[tuple[int, dict]]] = {}

    for idx, chunk in enumerate(chunks):
        payload   = chunk.payload
        hierarchy = payload.get("section_hierarchy", [])
        top_level = hierarchy[0] if hierarchy else payload.get(
            "section_title", "unknown"
        )
        normalized = normalize_section_title(top_level)

        if normalized not in section_groups:
            section_groups[normalized] = []
        section_groups[normalized].append((idx, payload))

    logger.info(
        f"Initial grouping: {len(section_groups)} section groups "
        f"from {len(chunks)} chunks"
    )

    # ── Step 2: Apply size rules ───────────────────────────────────────────
    # Preserve reading order by sorting groups by first chunk's order
    ordered_groups = sorted(
        section_groups.items(),
        key=lambda x: x[1][0][1].get("order_range", [0])[0]
        if x[1] else 0
    )

    clusters: list[StructuralCluster] = []
    cluster_id = 0
    pending_merge: list[tuple[int, dict]] = []
    pending_section: str = ""

    for section_norm, chunk_pairs in ordered_groups:

        # Merge pending small group into current
        if pending_merge:
            chunk_pairs    = pending_merge + chunk_pairs
            pending_merge  = []
            pending_section = ""

        # ── Too small: hold for merge with next ───────────────────────
        if len(chunk_pairs) < MIN_CLUSTER_SIZE:
            if clusters:
                # Merge backward into last cluster
                last = clusters[-1]
                for idx, payload in chunk_pairs:
                    last.chunk_ids.append(payload["chunk_id"])
                    last.chunk_indices.append(idx)
                logger.debug(
                    f"Merged small group '{section_norm}' "
                    f"({len(chunk_pairs)} chunks) into "
                    f"'{last.section_title}'"
                )
            else:
                # No previous cluster, hold for next
                pending_merge   = chunk_pairs
                pending_section = section_norm
            continue

        # ── Too large: split ───────────────────────────────────────────
        if len(chunk_pairs) > MAX_CLUSTER_SIZE:
            sub_clusters = _split_large_group(
                chunk_pairs, section_norm, cluster_id
            )
            clusters.extend(sub_clusters)
            cluster_id += len(sub_clusters)
            continue

        # ── Normal size: create cluster ────────────────────────────────
        first_payload = chunk_pairs[0][1]
        cluster = StructuralCluster(
            cluster_id        = cluster_id,
            section_title     = first_payload.get("section_title", section_norm),
            section_hierarchy = first_payload.get("section_hierarchy", [section_norm]),
            chunk_ids         = [p["chunk_id"] for _, p in chunk_pairs],
            chunk_indices     = [i for i, _ in chunk_pairs],
        )
        clusters.append(cluster)
        cluster_id += 1

    # ── Handle remaining pending merge ────────────────────────────────────
    if pending_merge and clusters:
        last = clusters[-1]
        for idx, payload in pending_merge:
            last.chunk_ids.append(payload["chunk_id"])
            last.chunk_indices.append(idx)
    elif pending_merge:
        # Edge case: entire document is one tiny section
        first_payload = pending_merge[0][1]
        cluster = StructuralCluster(
            cluster_id        = cluster_id,
            section_title     = first_payload.get("section_title", "document"),
            section_hierarchy = first_payload.get("section_hierarchy", ["document"]),
            chunk_ids         = [p["chunk_id"] for _, p in pending_merge],
            chunk_indices     = [i for i, _ in pending_merge],
        )
        clusters.append(cluster)

    logger.info(
        f"Structural clustering complete: "
        f"{len(clusters)} clusters, k={len(clusters)}"
    )
    for c in clusters:
        logger.debug(
            f"  Cluster {c.cluster_id}: '{c.section_title}' "
            f"({len(c.chunk_ids)} chunks)"
        )

    return clusters


def _split_large_group(
    chunk_pairs: list[tuple[int, dict]],
    section_norm: str,
    start_cluster_id: int,
) -> list[StructuralCluster]:
    """
    Split a large section group into sub-clusters.
    Tries subsection boundaries first, falls back to equal halves.
    """
    # Try subsection split
    subsection_groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, payload in chunk_pairs:
        hierarchy   = payload.get("section_hierarchy", [])
        sub_section = hierarchy[1] if len(hierarchy) > 1 else None
        key         = normalize_section_title(sub_section) \
            if sub_section else "main"
        if key not in subsection_groups:
            subsection_groups[key] = []
        subsection_groups[key].append((idx, payload))

    has_subsections = len(subsection_groups) > 1

    if has_subsections:
        sub_clusters = []
        cid = start_cluster_id
        for sub_norm, sub_pairs in subsection_groups.items():
            if not sub_pairs:
                continue
            first_payload = sub_pairs[0][1]
            sub_clusters.append(StructuralCluster(
                cluster_id        = cid,
                section_title     = first_payload.get(
                    "section_title", sub_norm
                ),
                section_hierarchy = first_payload.get(
                    "section_hierarchy", [sub_norm]
                ),
                chunk_ids         = [p["chunk_id"] for _, p in sub_pairs],
                chunk_indices     = [i for i, _ in sub_pairs],
                is_split          = True,
            ))
            cid += 1
        logger.debug(
            f"Split '{section_norm}' by subsection into "
            f"{len(sub_clusters)} sub-clusters"
        )
        return sub_clusters

    # No subsections: split into equal halves
    mid   = len(chunk_pairs) // 2
    parts = [chunk_pairs[:mid], chunk_pairs[mid:]]
    sub_clusters = []

    for part_idx, part in enumerate(parts):
        if not part:
            continue
        first_payload = part[0][1]
        sub_clusters.append(StructuralCluster(
            cluster_id        = start_cluster_id + part_idx,
            section_title     = first_payload.get("section_title", section_norm),
            section_hierarchy = first_payload.get("section_hierarchy", []),
            chunk_ids         = [p["chunk_id"] for _, p in part],
            chunk_indices     = [i for i, _ in part],
            is_split          = True,
            split_part        = part_idx + 1,
        ))

    logger.debug(
        f"Split '{section_norm}' into equal halves: "
        f"{[len(p.chunk_ids) for p in sub_clusters]}"
    )
    return sub_clusters