# raptor/pipeline.py
"""
RAPTOR tree construction orchestrator.

Reads leaf chunks from Qdrant → clusters → summarizes →
validates → embeds → deduplicates → stores back to Qdrant.

Also writes full debug output to data/raptor/{paper_id}/
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from core.logger import get_logger
from core.exceptions import (
    ClusteringError, SummarizationError, StorageError
)
from ingestion.embedder import Embedder
from ingestion.storage import QdrantStorage
from raptor.structural_clustering import build_structural_clusters
from raptor.fcm_clustering import run_fcm_clustering
from raptor.summarizer import check_llm_available
from raptor.tree_builder import build_l1_nodes, build_l2_nodes, build_l3_root
from raptor.deduplicator import deduplicate_summary_nodes
import config

logger = get_logger(__name__)


def _write_debug(paper_id: str, filename: str, data: object):
    """Write a debug JSON file to data/raptor/{paper_id}/"""
    raptor_dir = config.RAPTOR_DIR / paper_id
    raptor_dir.mkdir(parents=True, exist_ok=True)
    path = raptor_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.debug(f"Debug written: {path}")


#Checkpoint logic used in build_raptor_tree to check existing state in Qdrant and decide which steps can be skipped on resume.
def _get_existing_raptor_state(
    storage: QdrantStorage,
    paper_id: str,
) -> dict:
    """
    Check what RAPTOR nodes already exist in Qdrant for this paper.
    Used to determine which steps can be skipped on resume.

    Returns dict with counts per level.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    state = {"l0": 0, "l1": 0, "l2": 0, "l3": 0}

    for level, key in [(0, "l0"), (1, "l1"), (2, "l2"), (3, "l3")]:
        try:
            results, _ = storage.client.scroll(
                collection_name = storage.collection,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = "paper_id",
                            match = MatchValue(value=paper_id),
                        ),
                        FieldCondition(
                            key   = "level",
                            match = MatchValue(value=level),
                        ),
                    ]
                ),
                with_payload = False,
                with_vectors = False,
                limit        = 1000,
            )
            state[key] = len(results)
        except Exception:
            pass

    return state

#Checkpoint cleanup function to delete existing RAPTOR summary nodes (L1/L2/L3) for a paper before rebuilding the tree. Leaf chunks (L0) are preserved to avoid re-embedding.
def _delete_raptor_summary_nodes(
    storage: QdrantStorage,
    paper_id: str,
):
    """
    Delete ONLY L1/L2/L3 summary nodes for a paper.
    Leaf chunks (L0) are preserved — no re-embedding needed.
    Called before rebuilding the RAPTOR tree.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from qdrant_client.models import Range

    try:
        storage.client.delete(
            collection_name = storage.collection,
            points_selector = Filter(
                must=[
                    FieldCondition(
                        key   = "paper_id",
                        match = MatchValue(value=paper_id),
                    ),
                    FieldCondition(
                        key   = "level",
                        range = Range(gte=1),  # level >= 1 only
                    ),
                ]
            ),
        )
        logger.info(
            f"Cleared L1/L2/L3 summary nodes for '{paper_id}'. "
            f"L0 leaf chunks preserved."
        )
    except Exception as e:
        logger.warning(
            f"Could not clear summary nodes for '{paper_id}': {e}"
        )


def _load_nodes_by_level(
    storage: QdrantStorage,
    paper_id: str,
    level: int,
) -> list:
    """
    Load existing ChunkMetadata nodes from Qdrant by level.
    Used when resuming a partial build.
    Returns list of ChunkMetadata objects reconstructed from payload.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from core.models import ChunkMetadata

    results, _ = storage.client.scroll(
        collection_name = storage.collection,
        scroll_filter   = Filter(
            must=[
                FieldCondition(
                    key   = "paper_id",
                    match = MatchValue(value=paper_id),
                ),
                FieldCondition(
                    key   = "level",
                    match = MatchValue(value=level),
                ),
            ]
        ),
        with_payload = True,
        with_vectors = False,
        limit        = 1000,
    )

    nodes = []
    for r in results:
        try:
            node = ChunkMetadata.model_validate(r.payload)
            nodes.append(node)
        except Exception as e:
            logger.warning(
                f"Could not reconstruct node from payload: {e}"
            )

    logger.info(
        f"Loaded {len(nodes)} existing L{level} nodes for '{paper_id}'"
    )
    return nodes



def build_raptor_tree(paper_id: str, force_rebuild: bool = False) -> dict:

    start_time = time.time()
    logger.info(f"{'='*60}")
    logger.info(f"RAPTOR Tree Build: {paper_id}")
    logger.info(f"{'='*60}")

    # ── Step 0: Check LLM ─────────────────────────────────────────────────
    logger.info("Step 0: Checking LLM availability...")
    check_llm_available()

    # ── Step 1: Load leaf chunks ───────────────────────────────────────────
    logger.info("Step 1: Loading leaf chunks from Qdrant...")
    storage = QdrantStorage()
    chunks  = storage.get_leaf_chunks(paper_id)

    if not chunks:
        raise StorageError(
            f"No leaf chunks found for paper '{paper_id}'. "
            f"Run ingestion pipeline first."
        )

    logger.info(f"Loaded {len(chunks)} leaf chunks")

    paper_title = chunks[0].payload.get("paper_title", paper_id)
    language    = chunks[0].payload.get("language", "en")

    # ── Checkpoint: check existing state ──────────────────────────────────
    existing = _get_existing_raptor_state(storage, paper_id)
    logger.info(
        f"Existing RAPTOR state: "
        f"L0={existing['l0']} "
        f"L1={existing['l1']} "
        f"L2={existing['l2']} "
        f"L3={existing['l3']}"
    )

    # ── Scenario: complete tree exists ────────────────────────────────────
    if existing["l3"] == 1 and not force_rebuild:
        logger.info(
            f"Complete RAPTOR tree already exists for '{paper_id}'. "
            f"Use force_rebuild=True to rebuild."
        )
        return {
            "paper_id":         paper_id,
            "leaf_chunks":      existing["l0"],
            "l1_nodes":         existing["l1"],
            "l2_nodes":         existing["l2"],
            "l3_root":          1,
            "total_nodes":      sum(existing.values()),
            "nodes_upserted":   0,
            "duration_seconds": 0,
            "status":           "SKIPPED_ALREADY_EXISTS",
            "built_at":         datetime.now(timezone.utc).isoformat(),
        }

    # ── Scenario: force rebuild — clear summary nodes only ────────────────
    if force_rebuild:
        logger.info(
            "Force rebuild: clearing L1/L2/L3 nodes. "
            "L0 leaf chunks preserved."
        )
        _delete_raptor_summary_nodes(storage, paper_id)
        # Reset existing state after deletion
        existing = {"l0": existing["l0"], "l1": 0, "l2": 0, "l3": 0}

    # ── Determine which steps to run ──────────────────────────────────────
    # Each level is only built if it does not already exist.
    # Existing valid nodes are reused as-is.
    need_l1 = existing["l1"] == 0
    need_l2 = existing["l2"] == 0
    need_l3 = existing["l3"] == 0

    logger.info(
        f"Build plan: "
        f"L1={'BUILD' if need_l1 else 'REUSE'} "
        f"L2={'BUILD' if need_l2 else 'REUSE'} "
        f"L3={'BUILD' if need_l3 else 'REUSE'}"
    )

    # ── Extract dense vectors for clustering ──────────────────────────────
    dense_vectors = []
    for chunk in chunks:
        vec = chunk.vector
        if isinstance(vec, dict):
            vec = vec.get("dense", [])
        dense_vectors.append(vec if vec else [0.0] * config.EMBEDDING_DIM)

    chunk_id_to_index = {
        chunk.payload["chunk_id"]: idx
        for idx, chunk in enumerate(chunks)
    }

    # ── Step 2: Structural Clustering ─────────────────────────────────────
    # Always runs — needed to know k and cluster assignments
    # even when L1 already exists
    logger.info("Step 2: Structural clustering (Layer 1A)...")
    try:
        structural_clusters = build_structural_clusters(chunks)
    except Exception as e:
        raise ClusteringError(f"Structural clustering failed: {e}") from e

    k = len(structural_clusters)
    logger.info(f"Structural clusters: {k}")

    _write_debug(paper_id, "structural_clusters.json", [
        {
            "cluster_id":    c.cluster_id,
            "section_title": c.section_title,
            "chunk_count":   len(c.chunk_ids),
            "chunk_ids":     c.chunk_ids,
            "is_split":      c.is_split,
        }
        for c in structural_clusters
    ])

    # ── Step 3: FCM Clustering ─────────────────────────────────────────────
    # Always runs — needed for enrichment even when L1 exists
    logger.info("Step 3: FCM soft clustering (Layer 1B)...")
    try:
        fcm_memberships = run_fcm_clustering(
            chunks        = chunks,
            k             = k,
            dense_vectors = dense_vectors,
        )
    except ClusteringError as e:
        logger.warning(
            f"FCM failed: {e}. Using structural clusters only."
        )
        fcm_memberships = {
            chunk.payload["chunk_id"]: {}
            for chunk in chunks
        }

    _write_debug(paper_id, "fcm_clusters.json", fcm_memberships)

    # ── Step 4: L1 nodes ──────────────────────────────────────────────────
    if need_l1:
        logger.info("Step 4: Building L1 summary nodes...")
        l1_nodes = build_l1_nodes(
            paper_id            = paper_id,
            paper_title         = paper_title,
            language            = language,
            structural_clusters = structural_clusters,
            fcm_memberships     = fcm_memberships,
            all_chunks          = chunks,
            chunk_id_to_index   = chunk_id_to_index,
            l1_start_index      = 0,
        )
        logger.info(f"Built {len(l1_nodes)} L1 nodes")

        # Embed and store L1 immediately
        embedder       = Embedder()
        embedded_l1    = embedder.embed_chunks(l1_nodes)
        storage.upsert_chunks(embedded_l1)
        logger.info(f"L1 nodes stored in Qdrant")

        _write_debug(paper_id, "L1_summaries.json", [
            {
                "chunk_id":    n.chunk_id,
                "section":     n.section_title,
                "token_count": n.token_count,
                "children":    n.children_ids,
                "text":        n.text_for_embedding,
            }
            for n in l1_nodes
        ])

    else:
        # Load existing L1 nodes from Qdrant
        logger.info(
            f"Step 4: Reusing {existing['l1']} existing L1 nodes"
        )
        l1_nodes = _load_nodes_by_level(storage, paper_id, level=1)

    # ── Step 5: L2 nodes ──────────────────────────────────────────────────
    if need_l2:
        logger.info("Step 5: Building L2 summary nodes...")
        l2_nodes = build_l2_nodes(
            paper_id       = paper_id,
            paper_title    = paper_title,
            language       = language,
            l1_nodes       = l1_nodes,
            l2_start_index = 0,
        )
        logger.info(f"Built {len(l2_nodes)} L2 nodes")

        # Embed and store L2 immediately
        embedder    = Embedder()
        embedded_l2 = embedder.embed_chunks(l2_nodes)
        storage.upsert_chunks(embedded_l2)
        logger.info(f"L2 nodes stored in Qdrant")

        _write_debug(paper_id, "L2_summaries.json", [
            {
                "chunk_id":    n.chunk_id,
                "section":     n.section_title,
                "token_count": n.token_count,
                "children":    n.children_ids,
                "text":        n.text_for_embedding,
            }
            for n in l2_nodes
        ])

    else:
        logger.info(
            f"Step 5: Reusing {existing['l2']} existing L2 nodes"
        )
        l2_nodes = _load_nodes_by_level(storage, paper_id, level=2)

    # ── Step 6: L3 root ───────────────────────────────────────────────────
    if need_l3:
        logger.info("Step 6: Building L3 root summary...")
        l3_root = build_l3_root(
            paper_id    = paper_id,
            paper_title = paper_title,
            language    = language,
            l2_nodes    = l2_nodes,
            l1_nodes    = l1_nodes,
            root_index  = 0,
        )

        # Embed and store L3 immediately
        embedder    = Embedder()
        embedded_l3 = embedder.embed_chunks([l3_root])
        storage.upsert_chunks(embedded_l3)
        logger.info(f"L3 root stored in Qdrant")

        _write_debug(paper_id, "L3_root.json", {
            "chunk_id":    l3_root.chunk_id,
            "token_count": l3_root.token_count,
            "children":    l3_root.children_ids,
            "text":        l3_root.text_for_embedding,
        })

    else:
        logger.info("Step 6: Reusing existing L3 root")
        l3_nodes = _load_nodes_by_level(storage, paper_id, level=3)
        l3_root  = l3_nodes[0] if l3_nodes else None

    # ── Step 7: Backfill leaf links ────────────────────────────────────────
    # Always runs — ensures leaf chunks have correct parent_ids
    logger.info("Step 7: Backfilling RAPTOR links on leaf chunks...")
    _backfill_leaf_links(
        storage             = storage,
        structural_clusters = structural_clusters,
        fcm_memberships     = fcm_memberships,
        l1_nodes            = l1_nodes,
        chunks              = chunks,
    )

    # ── Step 8: Write tree graph ───────────────────────────────────────────
    all_summary_nodes = l1_nodes + l2_nodes + (
        [l3_root] if l3_root else []
    )
    tree_graph = _build_tree_graph(chunks, l1_nodes, l2_nodes, l3_root)
    _write_debug(paper_id, "tree_graph.json", tree_graph)

    duration = time.time() - start_time
    report = {
        "paper_id":         paper_id,
        "leaf_chunks":      len(chunks),
        "l1_nodes":         len(l1_nodes),
        "l2_nodes":         len(l2_nodes),
        "l3_root":          1 if l3_root else 0,
        "total_nodes":      len(chunks) + len(all_summary_nodes),
        "nodes_upserted":   (
            (len(l1_nodes) if need_l1 else 0) +
            (len(l2_nodes) if need_l2 else 0) +
            (1 if need_l3 else 0)
        ),
        "duration_seconds": round(duration, 2),
        "status":           "SUCCESS",
        "built_at":         datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"{'='*60}")
    logger.info(f"RAPTOR tree complete for '{paper_id}'")
    logger.info(f"  L0: {len(chunks)}")
    logger.info(f"  L1: {len(l1_nodes)}")
    logger.info(f"  L2: {len(l2_nodes)}")
    logger.info(f"  L3: 1")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"{'='*60}")



    #getting the sessions cost report for summaries
    from raptor.summarizer import get_session_cost_report

    cost_report = get_session_cost_report()
    logger.info(
        f"Summarizer session stats:\n"
        f"  Total LLM calls:      {cost_report['total_calls']}\n"
        f"  Fallback calls:       {cost_report['fallback_calls']}\n"
        f"  Extractive fallbacks: {cost_report['extractive_calls']}\n"
        f"  Failed calls:         {cost_report['failed_calls']}\n"
        f"  Input tokens:         {cost_report['total_input_tokens']}\n"
        f"  Output tokens:        {cost_report['total_output_tokens']}"
    )
    _write_debug(paper_id, "cost_report.json", cost_report)


    _write_debug(paper_id, "build_report.json", report)
    return report


def _backfill_leaf_links(
    storage: QdrantStorage,
    structural_clusters,
    fcm_memberships: dict,
    l1_nodes,
    chunks: list,
):
    """
    Update parent_ids, cluster_ids, cluster_probabilities,
    sibling_ids on existing leaf chunks in Qdrant.
    """
    # Build lookup: chunk_id → L1 node that covers it
    chunk_to_l1: dict[str, str] = {}
    chunk_to_cluster: dict[str, int] = {}

    for cluster in structural_clusters:
        for chunk_id in cluster.chunk_ids:
            chunk_to_cluster[chunk_id] = cluster.cluster_id

    for l1_node in l1_nodes:
        for chunk_id in l1_node.children_ids:
            chunk_to_l1[chunk_id] = l1_node.chunk_id

    for chunk in chunks:
        chunk_id    = chunk.payload["chunk_id"]
        parent_id   = chunk_to_l1.get(chunk_id)
        cluster_id  = chunk_to_cluster.get(chunk_id, -1)
        memberships = fcm_memberships.get(chunk_id, {})

        # Sibling ids: other chunks in same structural cluster
        sibling_ids = []
        if cluster_id >= 0:
            for sc in structural_clusters:
                if sc.cluster_id == cluster_id:
                    sibling_ids = [
                        cid for cid in sc.chunk_ids
                        if cid != chunk_id
                    ]
                    break

        storage.update_chunk_raptor_links(
            chunk_id_field        = "chunk_id",
            chunk_id_value        = chunk_id,
            parent_ids            = [parent_id] if parent_id else [],
            cluster_ids           = list(memberships.keys()),
            cluster_probabilities = {
                str(k): v for k, v in memberships.items()
            },
            sibling_ids           = sibling_ids,
        )


def _build_tree_graph(
    leaf_chunks, l1_nodes, l2_nodes, l3_root
) -> dict:
    """Build a tree graph dict for visualization and debugging."""
    nodes = []
    edges = []

    for chunk in leaf_chunks:
        p = chunk.payload
        nodes.append({
            "id":      p["chunk_id"],
            "level":   0,
            "section": p.get("section_title", ""),
            "tokens":  p.get("token_count", 0),
        })

    for node in l1_nodes + l2_nodes + [l3_root]:
        nodes.append({
            "id":      node.chunk_id,
            "level":   node.level,
            "section": node.section_title,
            "tokens":  node.token_count,
        })
        for child_id in node.children_ids:
            edges.append({
                "parent": node.chunk_id,
                "child":  child_id,
            })

    return {"nodes": nodes, "edges": edges}