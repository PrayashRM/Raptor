# ingestion_Phase2/__init__.py
from ingestion_Phase2.pipeline import ingest_paper, ingest_batch

__all__ = ["ingest_paper", "ingest_batch"]