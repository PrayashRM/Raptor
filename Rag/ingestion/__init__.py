# ingestion_Phase2/__init__.py
from ingestion.pipeline import ingest_paper, ingest_batch

__all__ = ["ingest_paper", "ingest_batch"]