# retrieval/pipeline.py
"""
Main retrieval orchestrator.
Runs the full retrieval pipeline for one query:
signal extraction → mode routing → hybrid search →
reranking → bidirectional tree traversal

Returns ordered context chunks ready for Phase 6.
"""

from __future__ import annotations
from dataclasses import dataclass
from qdrant_client.models import Filter, FieldCondition, MatchValue

from core.logger import get_logger
from core.exceptions import RetrievalError
from retrieval.signal_extractor import extract_signals, QuerySignals
from retrieval.mode_router import route_query, RetrievalConfig, RetrievalMode
from retrieval.searcher import HybridSearcher, search_for_mode_b_children
from retrieval.reranker import Reranker
from ingestion.storage import QdrantStorage
import config

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Complete retrieval result for one query."""
    query:            str
    paper_id:         str
    mode:             str
    signals:          QuerySignals
    final_chunks:     list[dict]     # ordered by chunk_sequence_index
    total_candidates: int
    after_rerank:     int
    after_traversal:  int


class RetrievalPipeline:
    """
    Full retrieval pipeline.
    Initialize once, call retrieve() for each query.
    Embedder and reranker are loaded once and cached.
    """

    def __init__(self):
        logger.info("Initializing retrieval pipeline...")
        self.searcher  = HybridSearcher()
        self.reranker  = Reranker()
        self.storage   = self.searcher.storage
        logger.info("Retrieval pipeline ready")

    def retrieve(
        self,
        query: str,
        paper_id: str,
    ) -> RetrievalResult:
        """
        Full retrieval for one query against one paper.

        Args:
            query:    User query string
            paper_id: Which paper to search

        Returns:
            RetrievalResult with final_chunks ready for generation
        """
        logger.info(f"Retrieving: '{query[:80]}...' paper='{paper_id}'")

        # ── Step 1: Extract signals ────────────────────────────────────────
        signals = extract_signals(query)

        # ── Step 2: Route to mode ──────────────────────────────────────────
        retrieval_config = route_query(signals)

        logger.info(
            f"Mode: {retrieval_config.mode.name} | "
            f"dense={retrieval_config.dense_weight} "
            f"sparse={retrieval_config.sparse_weight} | "
            f"top_k={retrieval_config.top_k}"
        )

        # ── Step 3: Execute search based on mode ───────────────────────────
        if retrieval_config.mode == RetrievalMode.B:
            candidates = self._search_mode_b(
                query, paper_id, retrieval_config
            )
        else:
            candidates = self.searcher.search(
                query            = query,
                paper_id         = paper_id,
                retrieval_config = retrieval_config,
            )

        total_candidates = len(candidates)

        if not candidates:
            logger.warning(
                f"No candidates found for query: '{query[:60]}'"
            )
            return RetrievalResult(
                query            = query,
                paper_id         = paper_id,
                mode             = retrieval_config.mode.name,
                signals          = signals,
                final_chunks     = [],
                total_candidates = 0,
                after_rerank     = 0,
                after_traversal  = 0,
            )

        # ── Step 4: Rerank ─────────────────────────────────────────────────
        reranked = self.reranker.rerank(
            query      = query,
            candidates = candidates,
            top_n      = retrieval_config.rerank_top_n,
        )
        after_rerank = len(reranked)

        # ── Step 5: Bidirectional tree traversal ───────────────────────────
        context_set = self._bidirectional_traversal(reranked)
        after_traversal = len(context_set)

        # ── Step 6: Sort by reading order ─────────────────────────────────
        final_chunks = sorted(
            context_set.values(),
            key=lambda x: x.get("chunk_sequence_index", 0),
        )

        logger.info(
            f"Retrieval complete: "
            f"candidates={total_candidates} → "
            f"reranked={after_rerank} → "
            f"final={after_traversal}"
        )

        return RetrievalResult(
            query            = query,
            paper_id         = paper_id,
            mode             = retrieval_config.mode.name,
            signals          = signals,
            final_chunks     = final_chunks,
            total_candidates = total_candidates,
            after_rerank     = after_rerank,
            after_traversal  = after_traversal,
        )

    def _search_mode_b(
        self,
        query: str,
        paper_id: str,
        retrieval_config: RetrievalConfig,
    ) -> list[dict]:
        """
        Mode B: Search summaries first, then fetch children.
        Returns combined list of summaries + their children.
        """
        # Search summary nodes only
        summary_candidates = self.searcher.search(
            query            = query,
            paper_id         = paper_id,
            retrieval_config = retrieval_config,
        )

        if not summary_candidates:
            # Fallback: search all levels if no summaries found
            logger.warning(
                "Mode B: no summaries found, falling back to all levels"
            )
            from retrieval.mode_router import RetrievalConfig, RetrievalMode
            fallback_config = RetrievalConfig(
                mode             = RetrievalMode.C,
                dense_weight     = 0.50,
                sparse_weight    = 0.50,
                top_k            = 8,
                level_filter     = None,
                modality_filter  = None,
                boost_key_sections = True,
                rerank_top_n     = retrieval_config.rerank_top_n,
            )
            return self.searcher.search(
                query            = query,
                paper_id         = paper_id,
                retrieval_config = fallback_config,
            )

        # Fetch top children for each summary
        children = search_for_mode_b_children(
            storage                 = self.storage,
            summary_payloads        = summary_candidates,
            top_children_per_summary = 3,
        )

        # Combine: summaries + children, deduplicated
        combined  = {c["_chunk_id"]: c for c in summary_candidates}
        for child in children:
            cid = child.get("_chunk_id") or child.get("chunk_id")
            if cid and cid not in combined:
                combined[cid] = child

        return list(combined.values())

    def _bidirectional_traversal(
        self,
        reranked_chunks: list[dict],
    ) -> dict[str, dict]:
        """
        Bidirectional tree traversal after reranking.

        For leaf nodes (level=0):
            Fetch parent summary → adds thematic context
            Direction: UP the tree

        For summary nodes (level>=1):
            Fetch top 2 children → adds specific evidence
            Direction: DOWN the tree
            Child priority: cluster_probability × rerank_score

        Returns:
            Dict of chunk_id → payload, deduplicated
        """
        context_set: dict[str, dict] = {}

        for chunk in reranked_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("_chunk_id", "")
            if chunk_id:
                context_set[chunk_id] = chunk

            level        = chunk.get("level", 0)
            rerank_score = chunk.get("_rerank_score", 0.5)

            if level == 0:
                # LEAF: fetch parent for thematic context
                parent_ids = chunk.get("parent_ids", [])
                for parent_id in parent_ids[:1]:  # max 1 parent
                    parent = self._fetch_chunk_by_id(parent_id)
                    if parent and parent_id not in context_set:
                        parent["_fetched_as"]    = "traversal_parent"
                        parent["_rerank_score"]  = rerank_score * 0.8
                        context_set[parent_id]   = parent

            elif level >= 1:
                # SUMMARY: fetch top 2 children for specific evidence
                children_ids   = chunk.get("children_ids", [])
                cluster_probs  = chunk.get("cluster_probabilities", {})

                # Score children: membership × parent_rerank_score
                def child_priority(cid: str) -> float:
                    membership = cluster_probs.get(cid, 0.1)
                    return membership * rerank_score

                top_children = sorted(
                    children_ids,
                    key=child_priority,
                    reverse=True,
                )[:2]

                for child_id in top_children:
                    if child_id not in context_set:
                        child = self._fetch_chunk_by_id(child_id)
                        if child:
                            child["_fetched_as"]   = "traversal_child"
                            child["_rerank_score"] = child_priority(child_id)
                            context_set[child_id]  = child

        logger.debug(
            f"Traversal: {len(reranked_chunks)} reranked → "
            f"{len(context_set)} context chunks"
        )
        return context_set

    def _fetch_chunk_by_id(self, chunk_id: str) -> dict | None:
        """
        Fetch a single chunk from Qdrant by chunk_id.
        Returns payload dict or None if not found.
        """
        try:
            results, _ = self.storage.client.scroll(
                collection_name = config.QDRANT_COLLECTION,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = "chunk_id",
                            match = MatchValue(value=chunk_id),
                        )
                    ]
                ),
                with_payload = True,
                with_vectors = False,
                limit        = 1,
            )
            if results:
                payload = results[0].payload.copy()
                payload["_chunk_id"] = chunk_id
                return payload
            return None
        except Exception as e:
            logger.warning(f"Could not fetch chunk '{chunk_id}': {e}")
            return None


# ── Module-level singleton ─────────────────────────────────────────────────────
# Loaded once per process, reused for all queries.
# Avoids reloading embedders and reranker on every request.
_pipeline_instance: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    """Get or create the singleton retrieval pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RetrievalPipeline()
    return _pipeline_instance


def retrieve(
    query: str,
    paper_id: str,
) -> RetrievalResult:
    """
    Main entry point for retrieval.
    Call this from Phase 6 generation and API routes.

    Args:
        query:    User query string
        paper_id: Paper to search within

    Returns:
        RetrievalResult with final_chunks sorted by reading order
    """
    pipeline = get_pipeline()
    return pipeline.retrieve(query=query, paper_id=paper_id)