# ingestion/embedder.py
"""
Produces dense (nomic) and sparse (SPLADE) vectors for each chunk.
Runs on CPU by default. GPU is automatically used if available
(PyTorch detects CUDA without any code change needed).
"""

from __future__ import annotations
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from core.models import ChunkMetadata, Chunk
from core.logger import get_logger
from core.exceptions import EmbeddingError
import config

logger = get_logger(__name__)


class DenseEmbedder:
    """
    Wraps nomic-embed-text-v1.5 via sentence-transformers.
    Produces 512-dim Matryoshka embeddings.
    """

    def __init__(self):
        logger.info(
            f"Loading dense embedding model: {config.EMBEDDING_MODEL}"
        )
        try:
            self.model = SentenceTransformer(
                config.EMBEDDING_MODEL,
                trust_remote_code=True,   # required for nomic
            )
            self.model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
            logger.info(f"Dense embedder loaded on {device}")
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load dense embedding model "
                f"'{config.EMBEDDING_MODEL}': {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prefixed = [config.EMBEDDING_PREFIX + t for t in texts]

        try:
            import numpy as np

            # Get full embeddings WITHOUT normalization first
            embeddings = self.model.encode(
                prefixed,
                batch_size           = 32,
                show_progress_bar    = False,
                normalize_embeddings = False,  # normalize AFTER truncation
            )

            # Truncate to Matryoshka target dimension
            embeddings = embeddings[:, :config.EMBEDDING_DIM]

            # Re-normalize AFTER truncation
            # This is the correct order for Matryoshka embeddings
            norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms      = np.where(norms == 0, 1, norms)  # avoid div by zero
            embeddings = embeddings / norms

            return embeddings.tolist()

        except Exception as e:
            raise EmbeddingError(
                f"Dense embedding failed for batch of "
                f"{len(texts)} texts: {e}"
            ) from e


class SparseEmbedder:
    """
    Wraps SPLADE via HuggingFace transformers.
    Produces sparse vectors (indices + values) for each text.
    Runs on CPU (SPLADE on CPU is slow but acceptable at ingest time).
    """

    def __init__(self):
        logger.info(
            f"Loading sparse embedding model: {config.SPARSE_MODEL}"
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.SPARSE_MODEL
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                config.SPARSE_MODEL
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Sparse embedder (SPLADE) loaded on {self.device}")
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load SPLADE model "
                f"'{config.SPARSE_MODEL}': {e}"
            ) from e

    def embed_single(self, text: str) -> tuple[list[int], list[float]]:
        """
        Produce sparse vector for a single text.
        Returns (indices, values) tuple.

        SPLADE is run one-at-a-time due to memory constraints on CPU.
        Batching SPLADE on CPU causes OOM on large texts.
        """
        try:
            tokens = self.tokenizer(
                text,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 512,
                padding        = False,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with torch.no_grad():
                output = self.model(**tokens)

            # SPLADE aggregation: max over sequence, then ReLU + log
            logits     = output.logits          # (1, seq_len, vocab_size)
            aggregated = torch.max(
                torch.log(1 + torch.relu(logits)),
                dim=1
            ).values.squeeze()                  # (vocab_size,)

            # Extract non-zero entries
            nonzero_mask   = aggregated > 0
            indices        = nonzero_mask.nonzero(as_tuple=True)[0].tolist()
            values         = aggregated[nonzero_mask].tolist()

            return indices, values

        except Exception as e:
            raise EmbeddingError(
                f"SPLADE embedding failed: {e}"
            ) from e

    def embed_batch(
        self, texts: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        """
        Embed a list of texts one at a time.
        Returns list of (indices, values) tuples.
        """
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                logger.debug(
                    f"SPLADE progress: {i}/{len(texts)} texts embedded"
                )
            result = self.embed_single(text)
            results.append(result)
        return results


class Embedder:
    """
    Unified embedder that runs both dense and sparse.
    Single entry point for the ingestion pipeline.
    """

    def __init__(self):
        self.dense  = DenseEmbedder()
        self.sparse = SparseEmbedder()

    def embed_chunks(self, chunks: list[ChunkMetadata]) -> list[Chunk]:
        """
        Embed all chunks with both dense and sparse models.
        Returns list of Chunk objects with vectors populated.
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks")

        texts = [c.text_for_embedding for c in chunks]

        # Dense: batch embed all at once
        logger.info("Running dense embedding (nomic)...")
        dense_vectors = self.dense.embed(texts)

        # Sparse: one at a time (SPLADE CPU constraint)
        logger.info("Running sparse embedding (SPLADE)...")
        sparse_results = self.sparse.embed_batch(texts)

        embedded_chunks = []
        for i, chunk_meta in enumerate(chunks):
            dense_vec              = dense_vectors[i]
            sparse_indices, sparse_values = sparse_results[i]

            embedded_chunks.append(Chunk(
                metadata       = chunk_meta,
                dense_vector   = dense_vec,
                sparse_indices = sparse_indices,
                sparse_values  = sparse_values,
            ))

        logger.info(f"Embedding complete for {len(embedded_chunks)} chunks")
        return embedded_chunks