# raptor/tree_builder.py
"""
Constructs L1, L2, L3 ChunkMetadata summary nodes.
Links parent_ids ↔ children_ids across the tree.
Calls summarizer + validator for each node.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone

from core.models import ChunkMetadata
from core.logger import get_logger
from core.utils import count_tokens, hash_text
from core.exceptions import SummarizationError
from raptor.structural_clustering import StructuralCluster
from raptor.summarizer import (
    summarize_l1_section,
    summarize_l2_group,
    summarize_l3_root,
)
from raptor.validator import validate_summary, extractive_fallback
import config

logger = get_logger(__name__)

MAX_RETRIES = 3


def _make_summary_chunk_id(
    paper_id: str,
    level: int,
    index: int,
) -> str:
    paper_short = paper_id.replace(" ", "_")[:30]
    return f"{paper_short}_L{level}_{index:04d}"


def _attempt_summary_with_validation(
    summarize_fn,
    source_texts: list[str],
    chunk_id_for_log: str,
    **kwargs,
) -> str:
    """
    Try to generate a valid summary up to MAX_RETRIES times.
    Falls back to extractive summary on repeated failure.
    """
    source_combined = "\n\n".join(source_texts)

    for attempt in range(MAX_RETRIES):
        try:
            summary = summarize_fn(**kwargs)
        except SummarizationError as e:
            logger.warning(
                f"Summarization attempt {attempt+1} failed "
                f"for {chunk_id_for_log}: {e}"
            )
            continue

        valid, reason = validate_summary(
            summary, source_combined, chunk_id_for_log
        )

        if valid:
            logger.debug(
                f"Summary validated for {chunk_id_for_log} "
                f"(attempt {attempt+1})"
            )
            return summary

        logger.warning(
            f"Summary validation failed for {chunk_id_for_log}: "
            f"{reason} (attempt {attempt+1})"
        )

    # All attempts failed: extractive fallback
    logger.warning(
        f"All {MAX_RETRIES} attempts failed for {chunk_id_for_log}. "
        f"Using extractive fallback."
    )
    return extractive_fallback(source_texts)


def build_l1_nodes(
    paper_id: str,
    paper_title: str,
    language: str,
    structural_clusters: list[StructuralCluster],
    fcm_memberships: dict[str, dict[int, float]],
    all_chunks: list[dict],
    chunk_id_to_index: dict[str, int],
    l1_start_index: int = 0,
) -> list[ChunkMetadata]:
    """
    Build one L1 summary node per structural cluster.
    FCM memberships enrich the summary with cross-section chunks.

    Returns list of L1 ChunkMetadata nodes (without vectors yet).
    """
    l1_nodes      = []
    total_chunks  = len(all_chunks)

    for cluster in structural_clusters:
        node_id = _make_summary_chunk_id(
            paper_id, 1, l1_start_index + cluster.cluster_id
        )

        # Primary chunks: all chunks in this structural cluster
        primary_chunk_ids = set(cluster.chunk_ids)

        # Enrichment chunks: chunks from OTHER clusters with
        # FCM membership in THIS cluster above threshold
        fcm_cluster_id = cluster.cluster_id
        enrichment_chunk_ids = set()

        for chunk_id, memberships in fcm_memberships.items():
            if (
                chunk_id not in primary_chunk_ids
                and fcm_cluster_id in memberships
                and memberships[fcm_cluster_id] >= config.FCM_MEMBERSHIP_THRESHOLD
            ):
                enrichment_chunk_ids.add(chunk_id)

        all_source_ids = list(primary_chunk_ids) + list(enrichment_chunk_ids)

        # Get texts for summarization (primary first, enrichment after)
        primary_texts    = []
        enrichment_texts = []

        for chunk_id in cluster.chunk_ids:
            idx = chunk_id_to_index.get(chunk_id)
            if idx is not None:
                text = all_chunks[idx].payload.get(
                    "text_for_embedding", ""
                )
                if text:
                    primary_texts.append(text)

        for chunk_id in enrichment_chunk_ids:
            idx = chunk_id_to_index.get(chunk_id)
            if idx is not None:
                text = all_chunks[idx].payload.get(
                    "text_for_embedding", ""
                )
                if text:
                    enrichment_texts.append(text)

        all_texts = primary_texts + enrichment_texts

        if not all_texts:
            logger.warning(
                f"No texts for cluster {cluster.cluster_id}, skipping"
            )
            continue

        if len(cluster.chunk_ids) < 3:
            logger.warning(
                f"Cluster {cluster.cluster_id} has only "
                f"{len(cluster.chunk_ids)} chunks, skipping L1 summary "
                f"(minimum is 3)"
            )
            continue

        logger.info(
            f"Generating L1 summary for cluster {cluster.cluster_id}: "
            f"'{cluster.section_title}' "
            f"({len(primary_texts)} primary + "
            f"{len(enrichment_texts)} enrichment chunks)"
        )

        summary_text = _attempt_summary_with_validation(
            summarize_fn  = summarize_l1_section,
            source_texts  = all_texts,
            chunk_id_for_log = node_id,
            section_title    = cluster.section_title,
            section_hierarchy = cluster.section_hierarchy,
            paper_title      = paper_title,
            chunk_texts      = all_texts,
        )

        # Get position info from first primary chunk
        first_chunk_payload = all_chunks[
            chunk_id_to_index[cluster.chunk_ids[0]]
        ].payload if cluster.chunk_ids else {}

        node = ChunkMetadata(
            # IDENTITY
            chunk_id        = node_id,
            paper_id        = paper_id,
            paper_title     = paper_title,
            content_hash    = hash_text(summary_text),
            embedding_model = config.EMBEDDING_MODEL,
            embedding_dim   = config.EMBEDDING_DIM,
            ingested_at     = datetime.now(timezone.utc).isoformat(),
            language        = language,

            # POSITION
            section_title     = cluster.section_title,
            section_hierarchy = cluster.section_hierarchy,
            is_key_section    = first_chunk_payload.get(
                "is_key_section", False
            ),
            page_range        = [
                first_chunk_payload.get("page_range", [0, 0])[0],
                all_chunks[
                    chunk_id_to_index[cluster.chunk_ids[-1]]
                ].payload.get("page_range", [0, 0])[-1]
                if cluster.chunk_ids else 0
            ],
            order_range       = [
                first_chunk_payload.get("order_range", [0, 0])[0],
                all_chunks[
                    chunk_id_to_index[cluster.chunk_ids[-1]]
                ].payload.get("order_range", [0, 0])[-1]
                if cluster.chunk_ids else 0
            ],
            chunk_sequence_index = l1_start_index + cluster.cluster_id,
            position_in_paper    = first_chunk_payload.get(
                "position_in_paper", 0.5
            ),

            # CONTENT TYPE
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

            # CONTENT
            text_for_embedding = summary_text,
            text_for_display   = summary_text,
            token_count        = count_tokens(summary_text),

            # RAPTOR TREE LINKS
            parent_ids    = [],  # filled when L2 nodes are built
            children_ids  = cluster.chunk_ids,
            summary_of    = all_source_ids,
            cluster_ids   = [cluster.cluster_id],
            cluster_probabilities = {
                str(cluster.cluster_id): 1.0
            },
            sibling_ids   = [],  # filled after all L1s are built

            # RETRIEVAL SIGNALS
            query_affinities  = ["broad"],
            section_importance = first_chunk_payload.get(
                "section_importance", "medium"
            ),
            retrieval_boost   = 1.10,

            # CROSS REFERENCES
            cites_tables    = [],
            cites_figures   = [],
            cites_equations = [],
            referenced_by_elements = [],

            soft_split         = False,
            source_element_ids = all_source_ids,
        )

        l1_nodes.append(node)
        logger.info(
            f"L1 node created: {node_id} "
            f"({node.token_count} tokens)"
        )

    # Fill sibling_ids for L1 nodes
    all_l1_ids = [n.chunk_id for n in l1_nodes]
    for node in l1_nodes:
        node.sibling_ids = [
            nid for nid in all_l1_ids
            if nid != node.chunk_id
        ]

    return l1_nodes


def build_l2_nodes(
    paper_id: str,
    paper_title: str,
    language: str,
    l1_nodes: list[ChunkMetadata],
    l2_start_index: int = 0,
) -> list[ChunkMetadata]:
    """
    Build L2 summary nodes by grouping related L1 nodes.
    Groups L1 nodes into 2-3 thematic groups by section proximity.
    """
    if len(l1_nodes) < 2:
        logger.info(
            f"Only {len(l1_nodes)} L1 nodes, skipping L2 build"
        )
        return []

    # Group L1 nodes into 2-3 groups based on count
    n_groups = min(3, max(2, len(l1_nodes) // 3))
    group_size = math.ceil(len(l1_nodes) / n_groups)

    groups = [
        l1_nodes[i:i + group_size]
        for i in range(0, len(l1_nodes), group_size)
    ]
    # Filter empty groups
    groups = [g for g in groups if g]

    logger.info(
        f"Building L2 nodes: {len(l1_nodes)} L1 nodes → "
        f"{len(groups)} L2 groups"
    )

    l2_nodes = []

    for group_idx, group in enumerate(groups):
        node_id = _make_summary_chunk_id(
            paper_id, 2, l2_start_index + group_idx
        )

        l1_texts         = [n.text_for_embedding for n in group]
        l1_section_titles = [n.section_title for n in group]
        children_ids     = [n.chunk_id for n in group]
        all_source_ids   = []
        for n in group:
            all_source_ids.extend(n.summary_of)

        logger.info(
            f"Generating L2 summary {group_idx + 1}/{len(groups)}: "
            f"sections {l1_section_titles}"
        )

        summary_text = _attempt_summary_with_validation(
            summarize_fn     = summarize_l2_group,
            source_texts     = l1_texts,
            chunk_id_for_log = node_id,
            paper_title      = paper_title,
            l1_summary_texts = l1_texts,
            section_titles   = l1_section_titles,
        )

        # Use position from first L1 in group
        first_l1 = group[0]
        last_l1  = group[-1]

        node = ChunkMetadata(
            chunk_id        = node_id,
            paper_id        = paper_id,
            paper_title     = paper_title,
            content_hash    = hash_text(summary_text),
            embedding_model = config.EMBEDDING_MODEL,
            embedding_dim   = config.EMBEDDING_DIM,
            ingested_at     = datetime.now(timezone.utc).isoformat(),
            language        = language,

            section_title     = f"Group: {l1_section_titles[0]} ... {l1_section_titles[-1]}",
            section_hierarchy = first_l1.section_hierarchy[:1],
            is_key_section    = False,
            page_range        = [
                first_l1.page_range[0],
                last_l1.page_range[-1],
            ],
            order_range       = [
                first_l1.order_range[0],
                last_l1.order_range[-1],
            ],
            chunk_sequence_index = l2_start_index + group_idx,
            position_in_paper    = first_l1.position_in_paper,

            level         = 2,
            depth         = 2,
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

            parent_ids    = [],  # filled when L3 is built
            children_ids  = children_ids,
            summary_of    = all_source_ids,
            cluster_ids   = [],
            cluster_probabilities = {},
            sibling_ids   = [],

            query_affinities  = ["broad"],
            section_importance = "high",
            retrieval_boost   = 1.05,

            cites_tables    = [],
            cites_figures   = [],
            cites_equations = [],
            referenced_by_elements = [],

            soft_split         = False,
            source_element_ids = all_source_ids,
        )

        # Backfill parent_ids on L1 children
        for l1_node in group:
            l1_node.parent_ids = [node_id]

        l2_nodes.append(node)
        logger.info(f"L2 node created: {node_id} ({node.token_count} tokens)")

    # Fill sibling_ids for L2 nodes
    all_l2_ids = [n.chunk_id for n in l2_nodes]
    for node in l2_nodes:
        node.sibling_ids = [
            nid for nid in all_l2_ids
            if nid != node.chunk_id
        ]

    return l2_nodes


def build_l3_root(
    paper_id: str,
    paper_title: str,
    language: str,
    l2_nodes: list[ChunkMetadata],
    l1_nodes: list[ChunkMetadata],
    root_index: int = 0,
) -> ChunkMetadata:
    """
    Build single L3 root summary node.
    Always built. Never skipped.
    Input: L2 summaries (or L1 if no L2 nodes exist).
    """
    source_nodes  = l2_nodes if l2_nodes else l1_nodes
    source_texts  = [n.text_for_embedding for n in source_nodes]
    children_ids  = [n.chunk_id for n in source_nodes]
    all_source_ids = []
    for n in source_nodes:
        all_source_ids.extend(n.summary_of)

    node_id = _make_summary_chunk_id(paper_id, 3, root_index)

    logger.info(
        f"Generating L3 root summary from "
        f"{len(source_nodes)} source nodes"
    )

    summary_text = _attempt_summary_with_validation(
        summarize_fn     = summarize_l3_root,
        source_texts     = source_texts,
        chunk_id_for_log = node_id,
        paper_title      = paper_title,
        l2_summary_texts = source_texts,
    )

    node = ChunkMetadata(
        chunk_id        = node_id,
        paper_id        = paper_id,
        paper_title     = paper_title,
        content_hash    = hash_text(summary_text),
        embedding_model = config.EMBEDDING_MODEL,
        embedding_dim   = config.EMBEDDING_DIM,
        ingested_at     = datetime.now(timezone.utc).isoformat(),
        language        = language,

        section_title     = f"ROOT: {paper_title}",
        section_hierarchy = ["ROOT"],
        is_key_section    = True,
        page_range        = [1, 999],
        order_range       = [0, 999],
        chunk_sequence_index = root_index,
        position_in_paper    = 0.5,

        level         = 3,
        depth         = 3,
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
        children_ids  = children_ids,
        summary_of    = all_source_ids,
        cluster_ids   = [],
        cluster_probabilities = {},
        sibling_ids   = [],

        query_affinities  = ["broad"],
        section_importance = "high",
        retrieval_boost   = 1.08,

        cites_tables    = [],
        cites_figures   = [],
        cites_equations = [],
        referenced_by_elements = [],

        soft_split         = False,
        source_element_ids = all_source_ids,
    )

    # Backfill parent_ids on L2/L1 children
    for child_node in source_nodes:
        child_node.parent_ids = [node_id]

    logger.info(f"L3 root created: {node_id} ({node.token_count} tokens)")
    return node