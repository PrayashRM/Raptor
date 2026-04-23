# retrieval/reranker.py
"""
Cross-encoder reranking using BAAI/bge-reranker-v2-m3.
Scores true relevance of retrieved candidates.
Drops chunks below minimum score threshold.
"""

from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from core.logger import get_logger
from core.exceptions import RetrievalError
import config

logger = get_logger(__name__)

RERANKER_MIN_SCORE = getattr(config, 'RERANKER_MIN_SCORE', 0.25)
RERANKER_MODEL     = getattr(config, 'RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')


class Reranker:
    """
    Cross-encoder reranker.
    Loaded once, reused for all queries.
    """

    def __init__(self):
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
            self.model     = AutoModelForSequenceClassification.from_pretrained(
                RERANKER_MODEL
            )
            self.device    = "cuda" if torch.cuda.is_available() else "cpu"
            self.model     = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Reranker loaded on {self.device}")
        except Exception as e:
            raise RetrievalError(
                f"Failed to load reranker '{RERANKER_MODEL}': {e}"
            ) from e

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Rerank candidates by true relevance to query.
        Uses text_for_display (full markdown, not stripped text).
        Drops candidates below RERANKER_MIN_SCORE.

        Args:
            query:      Original user query
            candidates: List of payload dicts from searcher
            top_n:      Maximum results to return

        Returns:
            Reranked and filtered list, best first.
            May return fewer than top_n if scores are low.
        """
        if not candidates:
            return []

        logger.info(
            f"Reranking {len(candidates)} candidates "
            f"(top_n={top_n})"
        )

        # Build (query, document) pairs
        # Use text_for_display — reranker sees full markdown
        pairs = [
            [query, c.get("text_for_display", c.get("text_for_embedding", ""))]
            for c in candidates
        ]

        # Score all pairs
        try:
            scores = self._score_pairs(pairs)
        except Exception as e:
            logger.warning(
                f"Reranker failed: {e}. "
                f"Returning candidates sorted by vector score."
            )
            # Fallback: return top_n by original vector score
            sorted_by_vector = sorted(
                candidates,
                key=lambda x: x.get("_score", 0),
                reverse=True,
            )
            return sorted_by_vector[:top_n]

        # Attach reranker scores
        for candidate, score in zip(candidates, scores):
            candidate["_rerank_score"] = float(score)

        # Filter by minimum score threshold
        above_threshold = [
            c for c in candidates
            if c["_rerank_score"] >= RERANKER_MIN_SCORE
        ]

        if not above_threshold:
            logger.warning(
                f"All {len(candidates)} candidates below "
                f"reranker threshold {RERANKER_MIN_SCORE}. "
                f"Returning top 1 regardless."
            )
            # Return at least 1 — better than empty context
            candidates.sort(
                key=lambda x: x["_rerank_score"], reverse=True
            )
            return candidates[:1]

        # Sort by reranker score and return top_n
        above_threshold.sort(
            key=lambda x: x["_rerank_score"], reverse=True
        )
        final = above_threshold[:top_n]

        logger.info(
            f"Reranking complete: {len(final)} results "
            f"(filtered from {len(candidates)}, "
            f"top score={final[0]['_rerank_score']:.4f})"
        )
        return final

    def _score_pairs(self, pairs: list[list[str]]) -> list[float]:
        """
        Score (query, document) pairs with cross-encoder.
        Processes in batches to avoid OOM on CPU.
        """
        all_scores = []
        batch_size = 8  # conservative for CPU

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                padding        = True,
                truncation     = True,
                max_length     = 512,
                return_tensors = "pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores  = outputs.logits.squeeze(-1)

                # Convert to sigmoid for probability-like scores
                scores = torch.sigmoid(scores)
                all_scores.extend(scores.cpu().tolist())

        return all_scores