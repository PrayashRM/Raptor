# retrieval/searcher.py
"""
Qdrant hybrid search execution.
Combines dense (nomic) and sparse (SPLADE) search
with metadata filtering and score boosting.
"""

from __future__ import annotations
import numpy as np
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    SparseVector, NamedVector, NamedSparseVector,
    SearchRequest, QueryResponse,
)
from core.logger import get_logger
from core.exceptions import RetrievalError
from ingestion.embedder import DenseEmbedder, SparseEmbedder
from ingestion.storage import QdrantStorage
from retrieval.mode_router import RetrievalConfig, RetrievalMode
import config

logger = get_logger(__name__)

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"


class HybridSearcher:
    """
    Executes hybrid dense + sparse search against Qdrant.
    Embedder instances are loaded once and reused.
    """

    def __init__(self):
        logger.info("Initializing hybrid searcher...")
        self.dense_embedder  = DenseEmbedder()
        self.sparse_embedder = SparseEmbedder()
        self.storage         = QdrantStorage()
        logger.info("Hybrid searcher ready")

    def search(
        self,
        query: str,
        paper_id: str,
        retrieval_config: RetrievalConfig,
    ) -> list[dict]:
        """
        Execute hybrid search for a query.

        Args:
            query:            User query string
            paper_id:         Paper to search within
            retrieval_config: Mode-specific search config

        Returns:
            List of payload dicts, sorted by combined score.
            Each dict has full chunk metadata + _score field.
        """
        logger.info(
            f"Hybrid search: mode={retrieval_config.mode.name} "
            f"paper='{paper_id}' top_k={retrieval_config.top_k}"
        )

        # ── Embed query ────────────────────────────────────────────────────
        query_prefix  = config.EMBEDDING_PREFIX + query
        dense_vector  = self.dense_embedder.embed([query])[0]
        sparse_idx, sparse_val = self.sparse_embedder.embed_single(
            query
        )

        # ── Build Qdrant filter ────────────────────────────────────────────
        must_conditions = [
            FieldCondition(
                key   = "paper_id",
                match = MatchValue(value=paper_id),
            )
        ]

        # Level filter
        if retrieval_config.level_filter is not None:
            if len(retrieval_config.level_filter) == 1:
                must_conditions.append(
                    FieldCondition(
                        key   = "level",
                        match = MatchValue(
                            value=retrieval_config.level_filter[0]
                        ),
                    )
                )
            else:
                # Multiple levels: use should (OR) condition
                from qdrant_client.models import Filter as QFilter
                level_conditions = [
                    FieldCondition(
                        key   = "level",
                        match = MatchValue(value=lvl),
                    )
                    for lvl in retrieval_config.level_filter
                ]
                must_conditions.append(
                    QFilter(should=level_conditions)
                )

        # Modality filter
        if retrieval_config.modality_filter:
            for field, value in retrieval_config.modality_filter.items():
                must_conditions.append(
                    FieldCondition(
                        key   = field,
                        match = MatchValue(value=value),
                    )
                )

        search_filter = Filter(must=must_conditions)

        # ── Dense search ───────────────────────────────────────────────────
        dense_results = self.storage.client.search(
            collection_name = config.QDRANT_COLLECTION,
            query_vector    = NamedVector(
                name   = DENSE_VECTOR_NAME,
                vector = dense_vector,
            ),
            query_filter    = search_filter,
            limit           = retrieval_config.top_k,
            with_payload    = True,
            with_vectors    = False,
            score_threshold = 0.0,
        )

        # ── Sparse search ──────────────────────────────────────────────────
        sparse_results = self.storage.client.search(
            collection_name = config.QDRANT_COLLECTION,
            query_vector    = NamedSparseVector(
                name   = SPARSE_VECTOR_NAME,
                vector = SparseVector(
                    indices = sparse_idx,
                    values  = sparse_val,
                ),
            ),
            query_filter    = search_filter,
            limit           = retrieval_config.top_k,
            with_payload    = True,
            with_vectors    = False,
        )

        # ── Combine scores (Reciprocal Rank Fusion) ────────────────────────
        combined = self._reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results,
            dense_weight   = retrieval_config.dense_weight,
            sparse_weight  = retrieval_config.sparse_weight,
        )

        # ── Apply score boosts ─────────────────────────────────────────────
        if retrieval_config.boost_key_sections:
            combined = self._apply_boosts(combined)

        # ── Sort and return top_k ──────────────────────────────────────────
        combined.sort(key=lambda x: x["_score"], reverse=True)
        top_results = combined[:retrieval_config.top_k]

        logger.info(
            f"Search returned {len(top_results)} candidates"
        )
        return top_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        dense_weight:  float,
        sparse_weight: float,
        k: int = 60,
    ) -> list[dict]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion.
        RRF is more robust than simple score interpolation because
        it handles different score scales between dense and sparse.

        Formula: score(d) = sum(weight / (k + rank(d)))
        """
        scores: dict[str, float]  = {}
        payloads: dict[str, dict] = {}

        # Dense rankings
        for rank, result in enumerate(dense_results):
            chunk_id = result.payload.get("chunk_id", str(result.id))
            rrf_score = dense_weight / (k + rank + 1)
            scores[chunk_id]   = scores.get(chunk_id, 0) + rrf_score
            payloads[chunk_id] = result.payload

        # Sparse rankings
        for rank, result in enumerate(sparse_results):
            chunk_id = result.payload.get("chunk_id", str(result.id))
            rrf_score = sparse_weight / (k + rank + 1)
            scores[chunk_id]   = scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in payloads:
                payloads[chunk_id] = result.payload

        # Build combined result list
        combined = []
        for chunk_id, score in scores.items():
            payload = payloads[chunk_id].copy()
            payload["_score"]    = score
            payload["_chunk_id"] = chunk_id
            combined.append(payload)

        return combined

    def _apply_boosts(self, results: list[dict]) -> list[dict]:
        """
        Apply score multipliers based on metadata signals.

        Boosts:
        - L1 summaries: ×1.10 (specific enough, context-rich)
        - L2 summaries: ×1.05
        - L3 root:      ×1.08
        - Key sections: ×1.05 (abstract, conclusion, results)
        - High importance: ×1.05
        """
        for result in results:
            multiplier = 1.0
            level = result.get("level", 0)

            # Level boosts
            level_boosts = {1: 1.10, 2: 1.05, 3: 1.08}
            multiplier *= level_boosts.get(level, 1.0)

            # Key section boost
            if result.get("is_key_section", False):
                multiplier *= 1.05

            # Section importance boost
            importance = result.get("section_importance", "medium")
            if importance == "high":
                multiplier *= 1.05
            elif importance == "low":
                multiplier *= 0.95

            result["_score"] *= multiplier

        return results


def search_for_mode_b_children(
    storage: QdrantStorage,
    summary_payloads: list[dict],
    top_children_per_summary: int = 3,
) -> list[dict]:
    """
    Mode B: After retrieving summary nodes,
    fetch their top children by cluster_probability.

    Args:
        storage:                  QdrantStorage instance
        summary_payloads:         List of L1/L2 payload dicts
        top_children_per_summary: Max children to fetch per summary

    Returns:
        List of child chunk payload dicts
    """
    child_payloads = []
    seen_ids       = set()

    for summary in summary_payloads:
        children_ids = summary.get("children_ids", [])
        cluster_probs = summary.get("cluster_probabilities", {})

        # Sort children by cluster probability (most representative first)
        sorted_children = sorted(
            children_ids,
            key=lambda cid: cluster_probs.get(cid, 0.0),
            reverse=True,
        )[:top_children_per_summary]

        for child_id in sorted_children:
            if child_id in seen_ids:
                continue
            seen_ids.add(child_id)

            # Fetch child from Qdrant
            results, _ = storage.client.scroll(
                collection_name = config.QDRANT_COLLECTION,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = "chunk_id",
                            match = MatchValue(value=child_id),
                        )
                    ]
                ),
                with_payload = True,
                with_vectors = False,
                limit        = 1,
            )

            if results:
                payload = results[0].payload.copy()
                payload["_score"]    = 0.5  # default score for fetched children
                payload["_chunk_id"] = child_id
                payload["_fetched_as"] = "mode_b_child"
                child_payloads.append(payload)

    logger.debug(
        f"Mode B fetched {len(child_payloads)} children "
        f"from {len(summary_payloads)} summaries"
    )
    return child_payloads