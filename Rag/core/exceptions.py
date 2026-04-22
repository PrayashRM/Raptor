# core/exceptions.py


class RAPTORRAGError(Exception):
    """Base exception for all RAPTOR RAG errors."""
    pass


class IngestionError(RAPTORRAGError):
    """Raised when ingestion pipeline fails."""
    pass


class ChunkingError(IngestionError):
    """Raised when element merging or chunking fails."""
    pass


class EmbeddingError(IngestionError):
    """Raised when dense or sparse embedding fails."""
    pass


class StorageError(IngestionError):
    """Raised when Qdrant upsert or setup fails."""
    pass


class OllamaNotAvailableError(RAPTORRAGError):
    """Raised when Ollama is not running at startup check."""
    pass


class ClusteringError(RAPTORRAGError):
    """Raised when structural or FCM clustering fails."""
    pass


class SummarizationError(RAPTORRAGError):
    """Raised when LLM summarization fails."""
    pass


class ValidationError(RAPTORRAGError):
    """Raised when summary validation fails after retries."""
    pass


class RetrievalError(RAPTORRAGError):
    """Raised when retrieval pipeline fails."""
    pass


class GenerationError(RAPTORRAGError):
    """Raised when final answer generation fails."""
    pass


class QdrantConnectionError(RAPTORRAGError):
    """Raised when Qdrant connection cannot be established."""
    pass


class InputFormatError(IngestionError):
    """Raised when input JSON does not match expected schema."""
    pass