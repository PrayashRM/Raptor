# ingestion/pipeline.py
"""
Main ingestion orchestrator.
Reads parsed JSON → chunks → embeds → stores.
Writes debug dumps and ingestion report.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from core.models import ParsedDocument, DocumentMeta, IngestionReport
from core.logger import get_logger
from core.exceptions import IngestionError, InputFormatError
from ingestion_Phase2.chunker import build_chunks
from ingestion_Phase2.embedder import Embedder
from ingestion_Phase2.storage import QdrantStorage
import config

logger = get_logger(__name__)


def _load_parsed_document(elements_path: Path) -> ParsedDocument:
    """
    Load and validate the parser JSON output.
    Fails fast with a clear error if schema does not match.
    """
    if not elements_path.exists():
        raise InputFormatError(
            f"elements.json not found at: {elements_path}"
        )

    try:
        with open(elements_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise InputFormatError(
            f"Invalid JSON in {elements_path}: {e}"
        ) from e

    try:
        document = ParsedDocument.model_validate(raw)
    except Exception as e:
        raise InputFormatError(
            f"elements.json does not match expected schema: {e}"
        ) from e

    logger.info(
        f"Loaded document '{document.document.paper_id}' "
        f"with {len(document.elements)} elements"
    )
    return document


def _write_document_meta(document: ParsedDocument, paper_dir: Path):
    """Extract and write document_meta.json from elements.json."""
    meta_path = paper_dir / "document_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            document.document.model_dump(),
            f, indent=2, ensure_ascii=False
        )
    logger.debug(f"Written document_meta.json to {meta_path}")


def _write_chunks_debug(
    chunks_data: list[dict],
    paper_id: str,
    report: dict,
):
    """Write chunks.json and chunks_report.json for debugging."""
    chunks_dir = config.CHUNKS_DIR / paper_id
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = chunks_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Debug chunks written to {chunks_path}")

    report_path = chunks_dir / "chunks_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Chunks report written to {report_path}")


def _write_references_meta(
    references_list: list[str],
    paper_id: str,
    parsed_dir: Path,
):
    """Write references to parsed directory as document-level metadata."""
    ref_path = parsed_dir / paper_id / "references.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(
            {"paper_id": paper_id, "references": references_list},
            f, indent=2, ensure_ascii=False
        )
    logger.debug(f"References metadata written to {ref_path}")


def ingest_paper(paper_dir_path: Path) -> IngestionReport:
    """
    Ingest a single paper from its parsed directory.

    Args:
        paper_dir_path: Path to directory containing elements.json
                        e.g., data/parsed/attention_is_all_you_need_2017/

    Returns:
        IngestionReport with full stats and status.
    """
    start_time = time.time()
    errors     = []

    logger.info(f"{'='*60}")
    logger.info(f"Starting ingestion: {paper_dir_path}")
    logger.info(f"{'='*60}")

    # ── Step 1: Load parsed document ──────────────────────────────────────
    elements_path = paper_dir_path / "elements.json"
    try:
        document = _load_parsed_document(elements_path)
    except InputFormatError as e:
        logger.error(str(e))
        return IngestionReport(
            paper_id                 = paper_dir_path.name,
            total_elements_input     = 0,
            elements_skipped         = 0,
            elements_processed       = 0,
            total_chunks_created     = 0,
            soft_splits              = 0,
            self_contained_chunks    = 0,
            merged_chunks            = 0,
            min_chunk_tokens         = 0,
            max_chunk_tokens         = 0,
            avg_chunk_tokens         = 0.0,
            sections_processed       = [],
            sections_skipped         = [],
            ingestion_duration_seconds = time.time() - start_time,
            status                   = "FAILED",
            errors                   = [str(e)],
        )

    paper_id    = document.document.paper_id
    total_elems = len(document.elements)

    # ── Step 2: Write document_meta.json ──────────────────────────────────
    _write_document_meta(document, paper_dir_path)

    # ── Step 3: Build chunks ───────────────────────────────────────────────
    logger.info("Phase 2A: Chunking elements...")
    try:
        chunks_meta, references_list = build_chunks(document)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        errors.append(f"Chunking error: {e}")
        return IngestionReport(
            paper_id                   = paper_id,
            total_elements_input       = total_elems,
            elements_skipped           = 0,
            elements_processed         = 0,
            total_chunks_created       = 0,
            soft_splits                = 0,
            self_contained_chunks      = 0,
            merged_chunks              = 0,
            min_chunk_tokens           = 0,
            max_chunk_tokens           = 0,
            avg_chunk_tokens           = 0.0,
            sections_processed         = [],
            sections_skipped           = [],
            ingestion_duration_seconds = time.time() - start_time,
            status                     = "FAILED",
            errors                     = errors,
        )

    # ── Step 4: Write references metadata ─────────────────────────────────
    if references_list:
        _write_references_meta(references_list, paper_id, config.PARSED_DIR)

    # ── Step 5: Write chunks debug dump ───────────────────────────────────
    token_counts = [c.token_count for c in chunks_meta]
    sections_seen = list({c.section_title for c in chunks_meta})

    chunk_report = {
        "paper_id":           paper_id,
        "total_chunks":       len(chunks_meta),
        "total_elements_in":  total_elems,
        "soft_splits":        sum(1 for c in chunks_meta if c.soft_split),
        "self_contained":     sum(
            1 for c in chunks_meta
            if c.chunk_type in ("table", "equation", "figure")
            and len(c.source_element_ids) == 1
        ),
        "min_tokens":         min(token_counts) if token_counts else 0,
        "max_tokens":         max(token_counts) if token_counts else 0,
        "avg_tokens":         round(
            sum(token_counts) / len(token_counts), 1
        ) if token_counts else 0.0,
        "sections":           sections_seen,
    }

    _write_chunks_debug(
        [c.model_dump() for c in chunks_meta],
        paper_id,
        chunk_report,
    )

    # ── Step 6: Embed chunks ───────────────────────────────────────────────
    logger.info("Phase 2B: Embedding chunks...")
    try:
        embedder      = Embedder()
        embedded_chunks = embedder.embed_chunks(chunks_meta)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        errors.append(f"Embedding error: {e}")
        return IngestionReport(
            paper_id                   = paper_id,
            total_elements_input       = total_elems,
            elements_skipped           = 0,
            elements_processed         = len(chunks_meta),
            total_chunks_created       = len(chunks_meta),
            soft_splits                = chunk_report["soft_splits"],
            self_contained_chunks      = chunk_report["self_contained"],
            merged_chunks              = len(chunks_meta),
            min_chunk_tokens           = chunk_report["min_tokens"],
            max_chunk_tokens           = chunk_report["max_tokens"],
            avg_chunk_tokens           = chunk_report["avg_tokens"],
            sections_processed         = sections_seen,
            sections_skipped           = [],
            ingestion_duration_seconds = time.time() - start_time,
            status                     = "FAILED",
            errors                     = errors,
        )

    # ── Step 7: Store in Qdrant ────────────────────────────────────────────
    logger.info("Phase 2C: Storing in Qdrant...")
    try:
        storage       = QdrantStorage()
        upserted_count = storage.upsert_chunks(embedded_chunks)
    except Exception as e:
        logger.error(f"Storage failed: {e}")
        errors.append(f"Storage error: {e}")
        status = "FAILED"
    else:
        status = "SUCCESS"

    duration = time.time() - start_time

    report = IngestionReport(
        paper_id                   = paper_id,
        total_elements_input       = total_elems,
        elements_skipped           = total_elems - sum(
            len(c.source_element_ids) for c in chunks_meta
        ),
        elements_processed         = sum(
            len(c.source_element_ids) for c in chunks_meta
        ),
        total_chunks_created       = len(chunks_meta),
        soft_splits                = chunk_report["soft_splits"],
        self_contained_chunks      = chunk_report["self_contained"],
        merged_chunks              = len(chunks_meta) - chunk_report["self_contained"],
        min_chunk_tokens           = chunk_report["min_tokens"],
        max_chunk_tokens           = chunk_report["max_tokens"],
        avg_chunk_tokens           = chunk_report["avg_tokens"],
        sections_processed         = sections_seen,
        sections_skipped           = list(config.SECTIONS_TO_SKIP),
        ingestion_duration_seconds = round(duration, 2),
        status                     = status,
        errors                     = errors,
    )

    logger.info(f"{'='*60}")
    logger.info(f"Ingestion complete: {paper_id}")
    logger.info(f"Status:   {status}")
    logger.info(f"Chunks:   {len(chunks_meta)}")
    logger.info(f"Upserted: {upserted_count}")
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"{'='*60}")

    return report


def ingest_batch(input_dir: Path) -> list[IngestionReport]:
    """
    Ingest all papers found in input_dir.
    Each subdirectory containing elements.json is treated as one paper.

    Args:
        input_dir: Path to directory containing paper subdirectories
                   e.g., data/parsed/

    Returns:
        List of IngestionReport, one per paper.
    """
    paper_dirs = [
        d for d in input_dir.iterdir()
        if d.is_dir() and (d / "elements.json").exists()
    ]

    if not paper_dirs:
        logger.warning(
            f"No paper directories with elements.json found in {input_dir}"
        )
        return []

    logger.info(f"Found {len(paper_dirs)} papers to ingest")

    reports = []
    for i, paper_dir in enumerate(sorted(paper_dirs)):
        logger.info(
            f"Processing paper {i+1}/{len(paper_dirs)}: {paper_dir.name}"
        )
        report = ingest_paper(paper_dir)
        reports.append(report)

    # Summary
    succeeded = sum(1 for r in reports if r.status == "SUCCESS")
    failed    = sum(1 for r in reports if r.status == "FAILED")
    logger.info(
        f"Batch ingestion complete: "
        f"{succeeded} succeeded, {failed} failed"
    )

    return reports