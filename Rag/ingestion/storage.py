# ingestion/storage.py
"""
Handles all Qdrant operations:
- Collection creation
- Content-hash deduplication
- Semantic deduplication of summary nodes
- Upsert with full metadata payload
"""

from __future__ import annotations
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    PointStruct, SparseVector, Filter, FieldCondition,
    MatchValue, PayloadSchemaType
)
from qdrant_client.http.models import UpdateStatus

from core.models import Chunk, ChunkMetadata
from core.logger import get_logger
from core.exceptions import StorageError, QdrantConnectionError
import config

logger = get_logger(__name__)

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantStorage:
    """
    Manages all interactions with the local Qdrant instance.
    """

    def __init__(self):
        self.client     = self._connect()
        self.collection = config.QDRANT_COLLECTION
        self._ensure_collection()
        self._ensure_payload_indexes()

    def _connect(self) -> QdrantClient:
        """
        Connect to Qdrant.
        Local mode:  uses path (development, no server needed)
        Server mode: uses host + port (production, supports concurrency)
        Controlled via config.QDRANT_USE_SERVER boolean.
        """
        try:
            import logging
            logging.getLogger("qdrant_client").setLevel(logging.ERROR)

            if config.QDRANT_USE_SERVER:
                # Production: server mode
                # Supports concurrent access, payload indexes work
                client = QdrantClient(
                    host   = config.QDRANT_HOST,
                    port   = config.QDRANT_PORT,
                )
                mode = f"server ({config.QDRANT_HOST}:{config.QDRANT_PORT})"
            else:
                # Development: local embedded mode
                # No server needed, single instance only
                config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                client = QdrantClient(
                    path = str(config.STORAGE_DIR),
                )
                mode = f"local ({config.STORAGE_DIR})"

            client.get_collections()
            logger.info(f"Connected to Qdrant [{mode}]")
            return client

        except Exception as e:
            raise QdrantConnectionError(
                f"Cannot connect to Qdrant. Error: {e}\n"
                f"QDRANT_USE_SERVER={config.QDRANT_USE_SERVER}\n"
                f"If local: check STORAGE_DIR path.\n"
                f"If server: check QDRANT_HOST and QDRANT_PORT."
            ) from e

    def _ensure_collection(self):
        """Create collection if it does not already exist."""
        existing = [
            c.name for c in self.client.get_collections().collections
        ]
        if self.collection in existing:
            logger.info(
                f"Collection '{self.collection}' already exists, skipping creation"
            )
            return

        logger.info(f"Creating collection '{self.collection}'")
        try:
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config  = {
                    DENSE_VECTOR_NAME: VectorParams(
                        size     = config.EMBEDDING_DIM,
                        distance = Distance.COSINE,
                    )
                },
                sparse_vectors_config = {
                    SPARSE_VECTOR_NAME: SparseVectorParams()
                },
            )
            logger.info(f"Collection '{self.collection}' created successfully")
        except Exception as e:
            raise StorageError(
                f"Failed to create Qdrant collection "
                f"'{self.collection}': {e}"
            ) from e

    def _ensure_payload_indexes(self):
        """Create payload indexes for fast filtered search."""
        indexed_fields = {
            "paper_id":           PayloadSchemaType.KEYWORD,
            "level":              PayloadSchemaType.INTEGER,
            "chunk_type":         PayloadSchemaType.KEYWORD,
            "has_table":          PayloadSchemaType.BOOL,
            "has_equation":       PayloadSchemaType.BOOL,
            "has_figure":         PayloadSchemaType.BOOL,
            "section_title":      PayloadSchemaType.KEYWORD,
            "is_key_section":     PayloadSchemaType.BOOL,
            "section_importance": PayloadSchemaType.KEYWORD,
            "language":           PayloadSchemaType.KEYWORD,
        }

        existing_indexes = set()
        try:
            collection_info = self.client.get_collection(self.collection)
            if collection_info.payload_schema:
                existing_indexes = set(
                    collection_info.payload_schema.keys()
                )
        except Exception:
            pass  # Collection might be empty, indexes don't exist yet

        for field_name, schema_type in indexed_fields.items():
            if field_name not in existing_indexes:
                try:
                    self.client.create_payload_index(
                        collection_name = self.collection,
                        field_name      = field_name,
                        field_schema    = schema_type,
                    )
                    logger.debug(f"Created payload index: {field_name}")
                except Exception as e:
                    logger.warning(
                        f"Could not create index for '{field_name}': {e}"
                    )

    def get_existing_hashes(self, paper_id: str) -> set[str]:
        """
        Retrieve all content_hash values already stored for a paper.
        Used for exact deduplication before upsert.
        """
        try:
            results, _ = self.client.scroll(
                collection_name = self.collection,
                scroll_filter   = Filter(
                    must=[
                        FieldCondition(
                            key   = "paper_id",
                            match = MatchValue(value=paper_id),
                        )
                    ]
                ),
                with_payload = ["content_hash"],
                with_vectors = False,
                limit        = 10_000,
            )
            return {
                r.payload["content_hash"]
                for r in results
                if r.payload and "content_hash" in r.payload
            }
        except Exception as e:
            logger.warning(
                f"Could not fetch existing hashes for '{paper_id}': {e}. "
                f"Proceeding without deduplication."
            )
            return set()

    def delete_paper(self, paper_id: str):
        """
        Delete all points for a given paper.
        Used when reingesting a paper with changed content.
        """
        try:
            self.client.delete(
                collection_name = self.collection,
                points_selector = Filter(
                    must=[
                        FieldCondition(
                            key   = "paper_id",
                            match = MatchValue(value=paper_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted all points for paper '{paper_id}'")
        except Exception as e:
            raise StorageError(
                f"Failed to delete points for paper '{paper_id}': {e}"
            ) from e

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Upsert chunks into Qdrant.
        Performs content-hash deduplication before upserting.

        Returns number of points actually upserted.
        """
        if not chunks:
            return 0

        paper_id = chunks[0].metadata.paper_id

        # Content-hash deduplication
        existing_hashes = self.get_existing_hashes(paper_id)
        new_chunks = [
            c for c in chunks
            if c.metadata.content_hash not in existing_hashes
        ]

        skipped = len(chunks) - len(new_chunks)
        if skipped > 0:
            logger.info(
                f"Skipping {skipped} chunks already in index "
                f"(content-hash match)"
            )

        if not new_chunks:
            logger.info("All chunks already indexed, nothing to upsert")
            return 0

        # Build Qdrant points
        points = []
        for chunk in new_chunks:
            meta = chunk.metadata

            # Payload: full metadata dict minus large text fields
            # (text fields are stored but not indexed for search)
            payload = meta.model_dump()

            point = PointStruct(
                id      = str(uuid.uuid4()),
                vector  = {
                    DENSE_VECTOR_NAME: chunk.dense_vector,
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices = chunk.sparse_indices,
                        values  = chunk.sparse_values,
                    ),
                },
                payload = payload,
            )
            points.append(point)

        # Batch upsert in groups of 100
        batch_size  = 100
        total_upserted = 0

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                result = self.client.upsert(
                    collection_name = self.collection,
                    points          = batch,
                    wait            = True,
                )
                if result.status == UpdateStatus.COMPLETED:
                    total_upserted += len(batch)
                    logger.debug(
                        f"Upserted batch {i // batch_size + 1}: "
                        f"{len(batch)} points"
                    )
                else:
                    logger.warning(
                        f"Upsert batch {i // batch_size + 1} "
                        f"returned status: {result.status}"
                    )
            except Exception as e:
                raise StorageError(
                    f"Qdrant upsert failed for batch starting at "
                    f"index {i}: {e}"
                ) from e

        logger.info(
            f"Upserted {total_upserted} new points for paper '{paper_id}'"
        )
        return total_upserted
    

    def get_leaf_chunks(self, paper_id: str) -> list[dict]:
        """
        Retrieve all level=0 leaf chunks for a paper from Qdrant.
        Returns list of dicts containing payload + dense vector.
        Used exclusively by Phase 3 RAPTOR tree builder.
        Phase 2 write methods are completely untouched.
        """
        try:
            results = []
            offset  = None

            while True:
                batch, next_offset = self.client.scroll(
                    collection_name = self.collection,
                    scroll_filter   = Filter(
                        must=[
                            FieldCondition(
                                key   = "paper_id",
                                match = MatchValue(value=paper_id),
                            ),
                            FieldCondition(
                                key   = "level",
                                match = MatchValue(value=0),
                            ),
                        ]
                    ),
                    with_payload = True,
                    with_vectors = [DENSE_VECTOR_NAME],
                    limit        = 100,
                    offset       = offset,
                )

                results.extend(batch)

                if next_offset is None:
                    break
                offset = next_offset

            logger.info(
                f"Retrieved {len(results)} leaf chunks "
                f"for paper '{paper_id}'"
            )
            return results

        except Exception as e:
            raise StorageError(
                f"Failed to retrieve leaf chunks for '{paper_id}': {e}"
            ) from e


    def update_chunk_raptor_links(
        self,
        chunk_id_field: str,
        chunk_id_value: str,
        parent_ids: list[str],
        cluster_ids: list[int],
        cluster_probabilities: dict[str, float],
        sibling_ids: list[str],
    ):
        """
        Update RAPTOR tree link fields on existing leaf chunks.
        Called by Phase 3 after clustering to backfill parent_ids,
        cluster_ids, cluster_probabilities, sibling_ids.
        Uses Qdrant set_payload to update without re-embedding.
        """
        try:
            self.client.set_payload(
                collection_name = self.collection,
                payload         = {
                    "parent_ids":            parent_ids,
                    "cluster_ids":           cluster_ids,
                    "cluster_probabilities": cluster_probabilities,
                    "sibling_ids":           sibling_ids,
                },
                points = Filter(
                    must=[
                        FieldCondition(
                            key   = "chunk_id",
                            match = MatchValue(value=chunk_id_value),
                        )
                    ]
                ),
            )
        except Exception as e:
            raise StorageError(
                f"Failed to update RAPTOR links for chunk "
                f"'{chunk_id_value}': {e}"
            ) from e