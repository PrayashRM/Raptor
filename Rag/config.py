# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
ASSETS_DIR = DATA_DIR / "assets"
CHUNKS_DIR = DATA_DIR / "chunks"
RAPTOR_DIR = DATA_DIR / "raptor"
STORAGE_DIR = BASE_DIR / "storage" / "qdrant_data"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM    = 512
EMBEDDING_PREFIX = "search_document: "
SPARSE_MODEL     = "prithivida/Splade_PP_En_V1"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MIN_TOKENS    = 100
CHUNK_TARGET_TOKENS = 300
CHUNK_MAX_TOKENS    = 500
CHUNK_HARD_LIMIT    = 500

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = "research_papers"
QDRANT_USE_SERVER = (
    os.getenv("QDRANT_USE_SERVER", "false").lower() == "true"
)

# ── RAPTOR / FCM ──────────────────────────────────────────────────────────────
FCM_FUZZINESS              = 2.0
FCM_MAX_ITER               = 150
FCM_TOLERANCE              = 1e-4
FCM_RANDOM_STATE           = 42
FCM_MEMBERSHIP_THRESHOLD   = 0.25
FCM_MAX_CLUSTERS_PER_CHUNK = 2

# ── Summary Validation ────────────────────────────────────────────────────────
SUMMARY_COMPRESSION_MAX  = 0.6
SEMANTIC_DEDUP_THRESHOLD = 0.90

# ── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL     = "BAAI/bge-reranker-v2-m3"
RERANKER_MIN_SCORE = 0.25

# ── Phase 3: Summarizer ───────────────────────────────────────────────────────
# Used ONLY at ingest time (offline, not at query time).

# Models
SUMMARIZER_PRIMARY_MODEL   = os.getenv(
    "SUMMARIZER_PRIMARY_MODEL", "gemini/gemini-2.5-flash"
)
SUMMARIZER_FALLBACK_MODELS = [
    m for m in [
        os.getenv("SUMMARIZER_FALLBACK_1", "gemini/gemini-2.0-flash"),
        os.getenv("SUMMARIZER_FALLBACK_2", "groq/llama-3.1-8b-instant"),
        os.getenv("SUMMARIZER_FALLBACK_3", "groq/gemma2-9b-it"),
    ] if m
]

# API keys — separate from generation keys
# Summarizer and generation share same physical keys but
# are tracked separately to avoid cross-contamination
SUMMARIZER_GEMINI_KEYS = [
    k for k in [
        os.getenv("SUMMARIZER_GEMINI_KEY_1",
            os.getenv("SUMMARIZER_GEMINI_API_KEY",
                os.getenv("GEMINI_API_KEY", ""))),
        os.getenv("SUMMARIZER_GEMINI_KEY_2", ""),
        os.getenv("SUMMARIZER_GEMINI_KEY_3", ""),
    ] if k
]

SUMMARIZER_GROQ_KEYS = [
    k for k in [
        os.getenv("SUMMARIZER_GROQ_KEY_1",
            os.getenv("SUMMARIZER_GROQ_API_KEY",
                os.getenv("GROQ_API_KEY", ""))),
        os.getenv("SUMMARIZER_GROQ_KEY_2", ""),
    ] if k
]

SUMMARIZER_OPENROUTER_KEY = os.getenv(
    "SUMMARIZER_OPENROUTER_KEY",
    os.getenv("OPENROUTER_API_KEY", "")
)

# Async + concurrency config
SUMMARIZER_USE_ASYNC = (
    os.getenv("SUMMARIZER_USE_ASYNC", "true").lower() == "true"
)
SUMMARIZER_MIN_CALL_GAP_SECONDS = float(
    os.getenv("SUMMARIZER_MIN_CALL_GAP_SECONDS", "3")
)
SUMMARIZER_MAX_CONCURRENT_SAME_PROVIDER = int(
    os.getenv("SUMMARIZER_MAX_CONCURRENT_SAME_PROVIDER", "1")
)
SUMMARIZER_MAX_CONCURRENT_TOTAL = int(
    os.getenv("SUMMARIZER_MAX_CONCURRENT_TOTAL", "1")
)

# Generation parameters
SUMMARIZER_TEMPERATURE = 0.1
SUMMARIZER_MAX_TOKENS  = 600

# Retry config
SUMMARIZER_MAX_RETRIES      = 5
SUMMARIZER_RETRY_BASE_DELAY = 5      # seconds, base for exponential backoff
SUMMARIZER_RETRY_MAX_DELAY  = 120    # seconds, cap on wait time
SUMMARIZER_REQUEST_TIMEOUT  = 120    # seconds, hard timeout per request

# Input size limits
SUMMARIZER_MAX_INPUT_TOKENS = 6000   # max tokens per LLM call input
SUMMARIZER_MAX_PART_TOKENS  = 2500   # max tokens per split part

# ── Phase 6: Generation ───────────────────────────────────────────────────────
# Used at query time for every user question.

# BYOK — user provides their own model + key via API request
GENERATION_BYOK_MODEL   = os.getenv("GENERATION_BYOK_MODEL", "")
GENERATION_BYOK_API_KEY = os.getenv("GENERATION_BYOK_API_KEY", "")

# System default models
GENERATION_PRIMARY_MODEL = os.getenv(
    "GENERATION_PRIMARY_MODEL",
    "gemini/gemini-2.0-flash"
)
GENERATION_FALLBACK_MODELS = [
    m for m in [
        os.getenv("GENERATION_FALLBACK_1", "gemini/gemini-2.5-flash"),
        os.getenv("GENERATION_FALLBACK_2", "groq/llama-3.1-70b-versatile"),
        os.getenv("GENERATION_FALLBACK_3", "groq/llama-3.1-8b-instant"),
        os.getenv("GENERATION_FALLBACK_4",
                  "openrouter/google/gemini-2.0-flash-exp:free"),
    ] if m
]

# API keys with multi-key rotation
GEMINI_API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY_1",
            os.getenv("GEMINI_API_KEY", "")),
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
    ] if k
]

GROQ_API_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY_1",
            os.getenv("GROQ_API_KEY", "")),
        os.getenv("GROQ_API_KEY_2", ""),
    ] if k
]

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Generation parameters
GENERATION_TEMPERATURE     = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))
GENERATION_MAX_TOKENS      = int(os.getenv("GENERATION_MAX_TOKENS", "2048"))
GENERATION_REQUEST_TIMEOUT = 120

# Context window limits (65% of actual model limits)
GENERATION_CONTEXT_RESERVE    = 1024
GENERATION_MAX_CONTEXT_TOKENS = {
    "gemini/gemini-2.0-flash":        int(1_048_576 * 0.65),
    "gemini/gemini-2.5-flash":        int(1_048_576 * 0.65),
    "groq/llama-3.1-70b-versatile":   int(131_072  * 0.65),
    "groq/llama-3.1-8b-instant":      int(131_072  * 0.65),
    "gpt-4o":                         int(128_000  * 0.65),
    "gpt-4o-mini":                    int(128_000  * 0.65),
    "claude-3-5-sonnet-20241022":     int(200_000  * 0.65),
    "default":                        int(32_000   * 0.65),
}

# Retry config
GENERATION_MAX_RETRIES = 5
GENERATION_RETRY_BASE  = 5
GENERATION_RETRY_MAX   = 120

# ── Sections ──────────────────────────────────────────────────────────────────
SECTIONS_TO_SKIP = {
    "references", "bibliography",
    "acknowledgements", "acknowledgments",
    "author contributions", "credit author statement",
    "conflict of interest", "conflict of interest statement",
    "funding", "funding sources",
    "neurips checklist", "iclr checklist",
    "reproducibility checklist",
    "licenses", "legal notices",
}

SECTIONS_METADATA_ONLY = {
    "references", "bibliography",
}

SECTIONS_HIGH = {
    "abstract",
    "introduction", "motivation",
    "conclusion", "conclusions",
    "contributions", "summary of contributions",
    "problem statement", "problem formulation",
    "method", "methods", "methodology",
    "proposed method", "our approach", "approach",
    "model", "architecture", "framework",
    "algorithm",
    "experiments", "experimental setup",
    "experimental results", "results",
    "evaluation", "evaluation setup",
    "baselines", "datasets", "data",
    "implementation details", "training details",
    "training", "training setup",
    "ablation study", "ablation studies",
    "analysis", "quantitative results",
    "discussion",
    "limitations", "limitations and future work",
}

SECTIONS_LOW = {
    "reproducibility statement",
    "iclr reproducibility statement",
    "qualitative examples",
}