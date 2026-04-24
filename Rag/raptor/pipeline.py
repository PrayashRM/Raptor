# raptor/pipeline.py
"""
RAPTOR tree construction orchestrator.
Supports async parallel cluster summarization.
Full checkpoint/resume logic.
Correct parent_ids backfill using point UUIDs.
"""

from __future__ import annotations
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone

from core.logger import get_logger
from core.exceptions import ClusteringError, StorageError
from core.models import ChunkMetadata
from ingestion.embedder import Embedder
from ingestion.storage import QdrantStorage
from raptor.structural_clustering import (
    build_structural_clusters, StructuralCluster
)
from raptor.fcm_clustering import run_fcm_clustering
from raptor.summarizer import (
    check_llm_available,
    summarize_clusters_parallel,
    summarize_l1_section,
    summarize_l2_group,
    summarize_l3_root,
    extractive_fallback,
    get_session_stats,
)
from raptor.validator import validate_summary, extractive_fallback
from raptor.tree_builder import (
    build_l1_nodes, build_l2_nodes, build_l3_root
)
from raptor.deduplicator import deduplicate_summary_nodes
import config

logger = get_logger(__name__)

MAX_SUMMARY_ATTEMPTS = 3


# ── Debug output helpers ───────────────────────────────────────────────────────

def _write_debug(paper_id: str, filename: str, data: object):
    """Write a debug JSON file to data/raptor/{paper_id}/"""
    raptor_dir = config.RAPTOR_DIR / paper_id
    raptor_dir.mkdir(parents=True, exist_ok=True)
    path = raptor_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.debug(f"Debug written: {path}")


# ── Qdrant state helpers ───────────────────────────────────────────────────────

def _get_existing_raptor_state(
    storage: QdrantStorage,
    paper_id: str,
) -> dict:
    """Check existing RAPTOR node counts per level."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    state = {"l0": 0, "l1": 0, "l2": 0, "l3": 0}
    for level, key in [(0,"l0"),(1,"l1"),(2,"l2"),(3,"l3")]:
        try:
            results, _ = storage.client.scroll(
                collection_name = storage.collection,
                scroll_filter   = Filter(must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id)
                    ),
                    FieldCondition(
                        key="level",
                        match=MatchValue(value=level)
                    ),
                ]),
                with_payload = False,
                with_vectors = False,
                limit        = 10_000,
            )
            state[key] = len(results)
        except Exception:
            pass
    return state


def _delete_raptor_summary_nodes(
    storage: QdrantStorage,
    paper_id: str,
):
    """Delete L1/L2/L3 nodes only. L0 leaves preserved."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, Range
    )
    try:
        storage.client.delete(
            collection_name = storage.collection,
            points_selector = Filter(must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id)
                ),
                FieldCondition(
                    key="level",
                    range=Range(gte=1)
                ),
            ]),
        )
        logger.info(
            f"Cleared L1/L2/L3 nodes for '{paper_id}'. "
            f"L0 preserved."
        )
    except Exception as e:
        logger.warning(f"Could not clear summary nodes: {e}")


def _delete_nodes_by_level(
    storage: QdrantStorage,
    paper_id: str,
    level: int,
):
    """Delete all nodes of a specific level for one paper."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue
    )
    try:
        storage.client.delete(
            collection_name = storage.collection,
            points_selector = Filter(must=[
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id)
                ),
                FieldCondition(
                    key="level",
                    match=MatchValue(value=level)
                ),
            ]),
        )
        logger.info(f"Cleared L{level} nodes for '{paper_id}'")
    except Exception as e:
        logger.warning(f"Could not clear L{level} nodes: {e}")


def _load_nodes_by_level(
    storage: QdrantStorage,
    paper_id: str,
    level: int,
) -> list[ChunkMetadata]:
    """Load existing ChunkMetadata nodes from Qdrant by level."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue
    )
    results, _ = storage.client.scroll(
        collection_name = storage.collection,
        scroll_filter   = Filter(must=[
            FieldCondition(
                key="paper_id",
                match=MatchValue(value=paper_id)
            ),
            FieldCondition(
                key="level",
                match=MatchValue(value=level)
            ),
        ]),
        with_payload = True,
        with_vectors = False,
        limit        = 10_000,
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
        f"Loaded {len(nodes)} existing L{level} nodes "
        f"for '{paper_id}'"
    )
    return nodes


# ── Parent backfill ────────────────────────────────────────────────────────────

def _backfill_parent_ids(
    storage: QdrantStorage,
    parent_node: ChunkMetadata,
):
    """
    Backfill parent_ids on all children of a node.
    Uses point UUID (not Filter) — works in local + server mode.
    Critical fix for orphan leaf chunks.
    """
    for child_id in parent_node.children_ids:
        try:
            storage.update_chunk_raptor_links(
                chunk_id_field        = "chunk_id",
                chunk_id_value        = child_id,
                parent_ids            = [parent_node.chunk_id],
                cluster_ids           = None,
                cluster_probabilities = None,
                sibling_ids           = None,
            )
        except Exception as e:
            logger.warning(
                f"Could not backfill parent for {child_id}: {e}"
            )


def _backfill_leaf_fcm_links(
    storage: QdrantStorage,
    structural_clusters: list[StructuralCluster],
    fcm_memberships: dict[str, dict],
    l1_nodes: list[ChunkMetadata],
    chunks: list,
):
    """
    Backfill cluster_ids, cluster_probabilities, sibling_ids
    on L0 leaf chunks.
    Uses UUID-based set_payload (local Qdrant compatible).
    """
    chunk_to_l1: dict[str, str] = {}
    chunk_to_cluster: dict[str, int] = {}

    for cluster in structural_clusters:
        for chunk_id in cluster.chunk_ids:
            chunk_to_cluster[chunk_id] = cluster.cluster_id

    for l1_node in l1_nodes:
        for chunk_id in l1_node.children_ids:
            chunk_to_l1[chunk_id] = l1_node.chunk_id

    for chunk in chunks:
        chunk_id   = chunk.payload["chunk_id"]
        cluster_id = chunk_to_cluster.get(chunk_id, -1)
        memberships = fcm_memberships.get(chunk_id, {})

        sibling_ids = []
        if cluster_id >= 0:
            for sc in structural_clusters:
                if sc.cluster_id == cluster_id:
                    sibling_ids = [
                        cid for cid in sc.chunk_ids
                        if cid != chunk_id
                    ]
                    break

        try:
            storage.update_chunk_raptor_links(
                chunk_id_field        = "chunk_id",
                chunk_id_value        = chunk_id,
                parent_ids            = [chunk_to_l1[chunk_id]]
                                        if chunk_id in chunk_to_l1
                                        else [],
                cluster_ids           = list(memberships.keys()),
                cluster_probabilities = {
                    str(k): v for k, v in memberships.items()
                },
                sibling_ids           = sibling_ids,
            )
        except Exception as e:
            logger.warning(
                f"Could not backfill leaf links for "
                f"{chunk_id}: {e}"
            )


# ── L1 async build ─────────────────────────────────────────────────────────────

async def _build_l1_nodes_async(
    paper_id: str,
    paper_title: str,
    language: str,
    structural_clusters: list[StructuralCluster],
    fcm_memberships: dict[str, dict],
    all_chunks: list,
    chunk_id_to_index: dict[str, int],
    l1_start_index: int,
) -> list[ChunkMetadata]:
    """
    Build L1 nodes with async parallel summarization.
    Each cluster is summarized concurrently where possible.
    """
    from raptor.tree_builder import _make_summary_chunk_id
    from raptor.tree_builder import _attempt_summary_with_validation
    from core.utils import count_tokens, hash_text
    from datetime import datetime, timezone

    valid_clusters = [
        c for c in structural_clusters
        if len(c.chunk_ids) >= 3
    ]

    if not valid_clusters:
        logger.warning("No clusters with >= 3 chunks for L1 building")
        return []

    # Prepare all summarization tasks
    tasks = []
    for cluster in valid_clusters:
        primary_ids = set(cluster.chunk_ids)

        # FCM enrichment chunks
        fcm_cluster_id = cluster.cluster_id
        enrichment_ids = {
            cid for cid, memberships in fcm_memberships.items()
            if cid not in primary_ids
            and fcm_cluster_id in memberships
            and memberships[fcm_cluster_id]
            >= config.FCM_MEMBERSHIP_THRESHOLD
        }

        # Collect texts
        primary_texts    = []
        enrichment_texts = []

        for cid in cluster.chunk_ids:
            idx = chunk_id_to_index.get(cid)
            if idx is not None:
                text = all_chunks[idx].payload.get(
                    "text_for_embedding", ""
                )
                if text:
                    primary_texts.append(text)

        for cid in enrichment_ids:
            idx = chunk_id_to_index.get(cid)
            if idx is not None:
                text = all_chunks[idx].payload.get(
                    "text_for_embedding", ""
                )
                if text:
                    enrichment_texts.append(text)

        all_texts  = primary_texts + enrichment_texts
        combined   = "\n\n".join(all_texts)
        word_limit = max(
            100, min(350, int(len(combined.split()) * 0.30))
        )

        tasks.append({
            "cluster":         cluster,
            "section_title":   cluster.section_title,
            "section_hierarchy": cluster.section_hierarchy,
            "paper_title":     paper_title,
            "chunk_texts":     all_texts,
            "word_limit":      word_limit,
            "primary_ids":     list(primary_ids),
            "enrichment_ids":  list(enrichment_ids),
            "all_source_ids":  list(primary_ids | enrichment_ids),
        })

    logger.info(
        f"Building {len(tasks)} L1 nodes "
        f"(async={config.SUMMARIZER_USE_ASYNC})"
    )

    # Run summarization
    if config.SUMMARIZER_USE_ASYNC and len(tasks) > 1:
        summaries = await summarize_clusters_parallel(tasks)
    else:
        # Sequential fallback
        summaries = []
        for task in tasks:
            try:
                from raptor.summarizer import _call_llm, _build_l1_prompt
                prompt = _build_l1_prompt(
                    task["section_title"],
                    task["paper_title"],
                    "\n\n".join(task["chunk_texts"]),
                    task["word_limit"],
                )
                summaries.append(_call_llm(prompt))
            except Exception as e:
                logger.warning(
                    f"L1 summarization failed for "
                    f"'{task['section_title']}': {e}"
                )
                summaries.append(
                    extractive_fallback(task["chunk_texts"])
                )

    # Build ChunkMetadata nodes from summaries
    l1_nodes = []
    all_l1_ids = []

    for task_idx, (task, summary_text) in enumerate(
        zip(tasks, summaries)
    ):
        cluster  = task["cluster"]
        node_id  = _make_summary_chunk_id(
            paper_id, 1, l1_start_index + cluster.cluster_id
        )
        all_l1_ids.append(node_id)

        # Validate summary
        source_combined = "\n\n".join(task["chunk_texts"])
        valid, reason   = validate_summary(
            summary_text, source_combined, node_id
        )
        if not valid:
            logger.warning(
                f"L1 summary validation failed for {node_id}: "
                f"{reason}. Using extractive fallback."
            )
            summary_text = extractive_fallback(task["chunk_texts"])

        first_chunk_payload = all_chunks[
            chunk_id_to_index[cluster.chunk_ids[0]]
        ].payload if cluster.chunk_ids else {}

        node = ChunkMetadata(
            chunk_id        = node_id,
            paper_id        = paper_id,
            paper_title     = paper_title,
            content_hash    = hash_text(summary_text),
            embedding_model = config.EMBEDDING_MODEL,
            embedding_dim   = config.EMBEDDING_DIM,
            ingested_at     = datetime.now(timezone.utc).isoformat(),
            language        = language,
            section_title     = cluster.section_title,
            section_hierarchy = cluster.section_hierarchy,
            is_key_section    = first_chunk_payload.get(
                "is_key_section", False
            ),
            page_range        = [
                first_chunk_payload.get("page_range", [0,0])[0],
                all_chunks[
                    chunk_id_to_index[cluster.chunk_ids[-1]]
                ].payload.get("page_range", [0,0])[-1]
                if cluster.chunk_ids else 0
            ],
            order_range       = [
                first_chunk_payload.get("order_range", [0,0])[0],
                all_chunks[
                    chunk_id_to_index[cluster.chunk_ids[-1]]
                ].payload.get("order_range", [0,0])[-1]
                if cluster.chunk_ids else 0
            ],
            chunk_sequence_index = l1_start_index + cluster.cluster_id,
            position_in_paper    = first_chunk_payload.get(
                "position_in_paper", 0.5
            ),
            level         = 1,
            depth         = 1,
            chunk_type    = "text",
            modalities    = ["text"],
            has_table     = False,
            has_equation  = False,
            has_figure    = False,
            table_image_paths    = [],
            equation_image_paths = [],
            figure_image_paths   = [],
            equation_latex       = [],
            text_for_embedding = summary_text,
            text_for_display   = summary_text,
            token_count        = count_tokens(summary_text),
            parent_ids    = [],
            children_ids  = cluster.chunk_ids,
            summary_of    = task["all_source_ids"],
            cluster_ids   = [cluster.cluster_id],
            cluster_probabilities = {
                str(cluster.cluster_id): 1.0
            },
            sibling_ids   = [],
            query_affinities  = ["broad"],
            section_importance = first_chunk_payload.get(
                "section_importance", "medium"
            ),
            retrieval_boost   = 1.10,
            cites_tables    = [],
            cites_figures   = [],
            cites_equations = [],
            referenced_by_elements = [],
            soft_split         = False,
            source_element_ids = task["all_source_ids"],
        )
        l1_nodes.append(node)
        logger.info(
            f"L1 node created: {node_id} "
            f"({node.token_count} tokens)"
        )

    # Fill sibling_ids
    for node in l1_nodes:
        node.sibling_ids = [
            n.chunk_id for n in l1_nodes
            if n.chunk_id != node.chunk_id
        ]

    return l1_nodes


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build_raptor_tree(
    paper_id: str,
    force_rebuild: bool = False,
) -> dict:
    """
    Full RAPTOR tree construction with:
    - Async parallel L1 summarization
    - Checkpoint/resume at each level
    - Correct parent_ids backfill (UUID-based)
    - Full error recovery
    """
    return asyncio.run(
        _build_raptor_tree_async(paper_id, force_rebuild)
    )


async def _build_raptor_tree_async(
    paper_id: str,
    force_rebuild: bool = False,
) -> dict:
    """Async implementation of RAPTOR tree build."""
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
            f"No leaf chunks for '{paper_id}'. "
            f"Run ingestion first."
        )

    logger.info(f"Loaded {len(chunks)} leaf chunks")

    paper_title = chunks[0].payload.get("paper_title", paper_id)
    language    = chunks[0].payload.get("language", "en")
    n_chunks    = len(chunks)

    dense_vectors = []
    for chunk in chunks:
        vec = chunk.vector
        if isinstance(vec, dict):
            vec = vec.get("dense", [])
        dense_vectors.append(
            vec if vec else [0.0] * config.EMBEDDING_DIM
        )

    chunk_id_to_index = {
        chunk.payload["chunk_id"]: idx
        for idx, chunk in enumerate(chunks)
    }

    # ── Checkpoint check ───────────────────────────────────────────────────
    existing = _get_existing_raptor_state(storage, paper_id)
    logger.info(
        f"Existing state: "
        f"L0={existing['l0']} L1={existing['l1']} "
        f"L2={existing['l2']} L3={existing['l3']}"
    )

    # Complete tree exists
    if existing["l3"] == 1 and not force_rebuild:
        logger.info(
            f"Complete tree exists for '{paper_id}'. "
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

    # Force rebuild: clear all summary nodes
    if force_rebuild:
        logger.info("Force rebuild: clearing L1/L2/L3...")
        _delete_raptor_summary_nodes(storage, paper_id)
        existing = {"l0": existing["l0"], "l1": 0, "l2": 0, "l3": 0}

    # ── Step 2: Structural clustering ──────────────────────────────────────
    logger.info("Step 2: Structural clustering...")
    try:
        structural_clusters = build_structural_clusters(chunks)
    except Exception as e:
        raise ClusteringError(
            f"Structural clustering failed: {e}"
        ) from e

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

    # ── Step 3: FCM clustering ─────────────────────────────────────────────
    logger.info("Step 3: FCM clustering...")
    try:
        fcm_memberships = run_fcm_clustering(
            chunks        = chunks,
            k             = k,
            dense_vectors = dense_vectors,
        )
    except ClusteringError as e:
        logger.warning(f"FCM failed: {e}. Structural only.")
        fcm_memberships = {
            chunk.payload["chunk_id"]: {}
            for chunk in chunks
        }

    _write_debug(paper_id, "fcm_clusters.json", fcm_memberships)

    # ── Determine build plan ───────────────────────────────────────────────
    expected_l1 = len([
        c for c in structural_clusters
        if len(c.chunk_ids) >= 3
    ])
    expected_l2 = max(2, len(structural_clusters) // 3)

    need_l1 = existing["l1"] < expected_l1
    need_l2 = existing["l2"] < expected_l2
    need_l3 = existing["l3"] == 0

    # Cascade: incomplete L1 forces L2 and L3 rebuild
    if need_l1:
        need_l2 = True
        need_l3 = True
        if existing["l1"] > 0:
            logger.info(
                f"Incomplete L1: {existing['l1']}/{expected_l1}. "
                f"Clearing and rebuilding."
            )
            _delete_nodes_by_level(storage, paper_id, 1)

    if need_l2 and existing["l2"] > 0:
        logger.info("Incomplete L2. Clearing and rebuilding.")
        _delete_nodes_by_level(storage, paper_id, 2)

    logger.info(
        f"Build plan: "
        f"L1={'BUILD' if need_l1 else 'REUSE'} "
        f"L2={'BUILD' if need_l2 else 'REUSE'} "
        f"L3={'BUILD' if need_l3 else 'REUSE'}"
    )

    # ── Step 4: L1 nodes ───────────────────────────────────────────────────
    if need_l1:
        logger.info("Step 4: Building L1 nodes (async)...")
        l1_nodes = await _build_l1_nodes_async(
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

        # Embed and store L1
        embedder    = Embedder()
        embedded_l1 = embedder.embed_chunks(l1_nodes)
        deduped_l1  = deduplicate_summary_nodes(embedded_l1)
        storage.upsert_chunks(deduped_l1)
        logger.info("L1 nodes stored")

        # Backfill leaf parent_ids immediately after L1 stored
        logger.info("Backfilling L0 parent_ids...")
        _backfill_leaf_fcm_links(
            storage, structural_clusters,
            fcm_memberships, l1_nodes, chunks
        )

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
        logger.info(f"Step 4: Reusing {existing['l1']} L1 nodes")
        l1_nodes = _load_nodes_by_level(storage, paper_id, 1)

    # ── Step 5: L2 nodes ───────────────────────────────────────────────────
    if need_l2:
        logger.info("Step 5: Building L2 nodes...")
        l2_nodes = build_l2_nodes(
            paper_id       = paper_id,
            paper_title    = paper_title,
            language       = language,
            l1_nodes       = l1_nodes,
            l2_start_index = 0,
        )
        logger.info(f"Built {len(l2_nodes)} L2 nodes")

        embedder    = Embedder()
        embedded_l2 = embedder.embed_chunks(l2_nodes)
        storage.upsert_chunks(embedded_l2)
        logger.info("L2 nodes stored")

        # Backfill L1 parent_ids
        logger.info("Backfilling L1 parent_ids...")
        for l2_node in l2_nodes:
            _backfill_parent_ids(storage, l2_node)

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
        logger.info(f"Step 5: Reusing {existing['l2']} L2 nodes")
        l2_nodes = _load_nodes_by_level(storage, paper_id, 2)

    # ── Step 6: L3 root ────────────────────────────────────────────────────
    if need_l3:
        logger.info("Step 6: Building L3 root...")
        l3_root = build_l3_root(
            paper_id    = paper_id,
            paper_title = paper_title,
            language    = language,
            l2_nodes    = l2_nodes,
            l1_nodes    = l1_nodes,
            root_index  = 0,
        )

        embedder    = Embedder()
        embedded_l3 = embedder.embed_chunks([l3_root])
        storage.upsert_chunks(embedded_l3)
        logger.info("L3 root stored")

        # Backfill L2 parent_ids
        logger.info("Backfilling L2 parent_ids...")
        _backfill_parent_ids(storage, l3_root)

        _write_debug(paper_id, "L3_root.json", {
            "chunk_id":    l3_root.chunk_id,
            "token_count": l3_root.token_count,
            "children":    l3_root.children_ids,
            "text":        l3_root.text_for_embedding,
        })

    else:
        logger.info("Step 6: Reusing existing L3 root")
        l3_list = _load_nodes_by_level(storage, paper_id, 3)
        l3_root = l3_list[0] if l3_list else None

    # ── Step 7: Write tree graph ────────────────────────────────────────────
    all_summary = l1_nodes + l2_nodes + (
        [l3_root] if l3_root else []
    )
    tree_graph = _build_tree_graph(chunks, l1_nodes, l2_nodes, l3_root)
    _write_debug(paper_id, "tree_graph.json", tree_graph)

    # ── Step 8: Session stats ──────────────────────────────────────────────
    stats = get_session_stats()
    logger.info(
        f"Summarizer stats: "
        f"calls={stats['total_calls']} "
        f"success={stats['successful_calls']} "
        f"fallback={stats['fallback_calls']} "
        f"extractive={stats['extractive_calls']} "
        f"input_tokens={stats['total_input_tokens']} "
        f"wait={stats['total_wait_seconds']:.1f}s"
    )
    _write_debug(paper_id, "session_stats.json", stats)

    duration = time.time() - start_time
    report = {
        "paper_id":         paper_id,
        "leaf_chunks":      len(chunks),
        "l1_nodes":         len(l1_nodes),
        "l2_nodes":         len(l2_nodes),
        "l3_root":          1 if l3_root else 0,
        "total_nodes":      len(chunks) + len(all_summary),
        "nodes_upserted":   (
            (len(l1_nodes) if need_l1 else 0) +
            (len(l2_nodes) if need_l2 else 0) +
            (1 if need_l3 else 0)
        ),
        "duration_seconds": round(duration, 2),
        "status":           "SUCCESS",
        "built_at":         datetime.now(timezone.utc).isoformat(),
        "summarizer_stats": stats,
    }

    logger.info(f"{'='*60}")
    logger.info(f"RAPTOR complete: {paper_id}")
    logger.info(f"  L0: {len(chunks)}")
    logger.info(f"  L1: {len(l1_nodes)}")
    logger.info(f"  L2: {len(l2_nodes)}")
    logger.info(f"  L3: 1")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"{'='*60}")

    _write_debug(paper_id, "build_report.json", report)
    return report


def _build_tree_graph(
    leaf_chunks, l1_nodes, l2_nodes, l3_root
) -> dict:
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

    for node in l1_nodes + l2_nodes + (
        [l3_root] if l3_root else []
    ):
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