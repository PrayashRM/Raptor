# raptor/deduplicator.py
"""
Cosine similarity deduplication of summary nodes.
Only runs on L1/L2/L3 nodes, never on leaf nodes.
Threshold: 0.90 (configured in config.py).
"""

from __future__ import annotations
import numpy as np
from core.models import ChunkMetadata, Chunk
from core.logger import get_logger
import config

logger = get_logger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def deduplicate_summary_nodes(
    embedded_nodes: list[Chunk],
) -> list[Chunk]:
    """
    Remove near-duplicate summary nodes before storage.
    Merges duplicate pairs: keeps longer summary,
    unions children_ids and parent_ids.

    Args:
        embedded_nodes: List of embedded Chunk objects (L1/L2/L3 only)

    Returns:
        Deduplicated list of Chunk objects.
    """
    if len(embedded_nodes) <= 1:
        return embedded_nodes

    threshold = config.SEMANTIC_DEDUP_THRESHOLD
    to_remove = set()
    merged    = {}  # index → merged Chunk

    for i in range(len(embedded_nodes)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(embedded_nodes)):
            if j in to_remove:
                continue

            sim = cosine_similarity(
                embedded_nodes[i].dense_vector,
                embedded_nodes[j].dense_vector,
            )

            if sim >= threshold:
                logger.info(
                    f"Dedup: merging {embedded_nodes[j].metadata.chunk_id} "
                    f"into {embedded_nodes[i].metadata.chunk_id} "
                    f"(similarity={sim:.4f})"
                )

                # Keep the longer summary
                if (
                    embedded_nodes[j].metadata.token_count
                    > embedded_nodes[i].metadata.token_count
                ):
                    keep, drop = j, i
                else:
                    keep, drop = i, j

                # Merge metadata links
                keep_meta = embedded_nodes[keep].metadata
                drop_meta = embedded_nodes[drop].metadata

                keep_meta.children_ids = sorted(set(
                    keep_meta.children_ids + drop_meta.children_ids
                ))
                keep_meta.parent_ids = sorted(set(
                    keep_meta.parent_ids + drop_meta.parent_ids
                ))
                keep_meta.summary_of = sorted(set(
                    keep_meta.summary_of + drop_meta.summary_of
                ))

                merged[keep] = embedded_nodes[keep]
                to_remove.add(drop)

    result = [
        merged.get(i, embedded_nodes[i])
        for i in range(len(embedded_nodes))
        if i not in to_remove
    ]

    removed = len(embedded_nodes) - len(result)
    if removed > 0:
        logger.info(
            f"Deduplication removed {removed} redundant summary nodes"
        )
    else:
        logger.info("No duplicate summary nodes found")

    return result