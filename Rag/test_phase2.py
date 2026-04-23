# test_ingestion_quick.py
"""
Quick end-to-end test for the ingestion pipeline.
Run from project root:
    python test_ingestion_quick.py

Expects:
    data/parsed/attention_is_all_you_need_2017/elements.json
"""

import json
import sys
from pathlib import Path

# ── 0. PATH SETUP ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(label: str, condition: bool, detail: str = ""):
    icon = "✅" if condition else "❌"
    msg  = f"  {icon} {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    if not condition:
        print(f"\n  STOPPING: Fix this before continuing.\n")
        sys.exit(1)


# ── 1. IMPORT TEST ─────────────────────────────────────────────────────────────
separator("TEST 1: Imports")

try:
    import config
    print("  ✅ config.py")
except Exception as e:
    print(f"  ❌ config.py: {e}")
    sys.exit(1)

try:
    from core.models import ParsedDocument, ChunkMetadata, Chunk
    print("  ✅ core.models")
except Exception as e:
    print(f"  ❌ core.models: {e}")
    sys.exit(1)

try:
    from core.utils import (
        count_tokens, hash_text, clean_text,
        extract_numbers, find_sentence_boundary,
        normalize_section_title
    )
    print("  ✅ core.utils")
except Exception as e:
    print(f"  ❌ core.utils: {e}")
    sys.exit(1)

try:
    from core.exceptions import (
        IngestionError, ChunkingError, EmbeddingError,
        StorageError, QdrantConnectionError, InputFormatError
    )
    print("  ✅ core.exceptions")
except Exception as e:
    print(f"  ❌ core.exceptions: {e}")
    sys.exit(1)

try:
    from ingestion.section_filter import (
        get_section_treatment, should_embed,
        get_importance_from_treatment, get_boost_from_treatment,
        get_query_affinities
    )
    print("  ✅ ingestion_Phase2    .section_filter")
except Exception as e:
    print(f"  ❌ ingestion_Phase2.section_filter: {e}")
    sys.exit(1)

try:
    from ingestion.chunker import build_chunks
    print("  ✅ ingestion.chunker")
except Exception as e:
    print(f"  ❌ ingestion_Phase2.chunker: {e}")
    sys.exit(1)

try:
    from ingestion.storage import QdrantStorage
    print("  ✅ ingestion_Phase2.storage")
except Exception as e:
    print(f"  ❌ ingestion_Phase2.storage: {e}")
    sys.exit(1)


# ── 2. UTILITY FUNCTIONS TEST ──────────────────────────────────────────────────
separator("TEST 2: Utility Functions")

# count_tokens
t = count_tokens("hello world this is a test sentence")
check("count_tokens returns int > 0", isinstance(t, int) and t > 0, f"got {t}")

t_empty = count_tokens("")
check("count_tokens empty string returns 0", t_empty == 0, f"got {t_empty}")

# hash_text
h1 = hash_text("hello")
h2 = hash_text("hello")
h3 = hash_text("world")
check("hash_text same input = same hash",   h1 == h2, f"{h1} vs {h2}")
check("hash_text diff input = diff hash",   h1 != h3, f"{h1} vs {h3}")
check("hash_text is 64 chars (sha256)",     len(h1) == 64, f"len={len(h1)}")

# clean_text
cleaned = clean_text("hello   world\n\n\n\ntest")
check(
    "clean_text normalizes whitespace",
    "   " not in cleaned and "\n\n\n" not in cleaned,
    f"got: '{cleaned}'"
)

# extract_numbers
nums = extract_numbers("The model achieved 28.4 BLEU with 6 layers")
check(
    "extract_numbers finds 28.4 and 6",
    "28.4" in nums and "6" in nums,
    f"got: {nums}"
)

# normalize_section_title
n1 = normalize_section_title("6.2 Model Variations")
n2 = normalize_section_title("3. Methods")
n3 = normalize_section_title("Abstract")
check("normalize strips number prefix", n1 == "model variations", f"got: '{n1}'")
check("normalize strips dot number",    n2 == "methods",          f"got: '{n2}'")
check("normalize lowercases",           n3 == "abstract",         f"got: '{n3}'")

# find_sentence_boundary
long_text = (
    "This is the first sentence. "
    "This is the second sentence. "
    "This is the third sentence. "
    "This is the fourth sentence. "
    "This is the fifth sentence."
)
boundary = find_sentence_boundary(long_text, 10)
check(
    "find_sentence_boundary returns valid index",
    0 < boundary <= len(long_text),
    f"got index {boundary} for text of length {len(long_text)}"
)


# ── 3. SECTION FILTER TEST ─────────────────────────────────────────────────────
separator("TEST 3: Section Filter")

from core.models import (
    ParsedElement, ElementLocation, ElementSection,
    ElementContent, ElementPhase2Ready, ElementAdjacency
)

def make_test_element(section_title: str, elem_type: str = "text") -> ParsedElement:
    return ParsedElement(
        element_id = "test_elem_001",
        type       = elem_type,
        location   = ElementLocation(page_no=1, order=1),
        section    = ElementSection(
            title            = section_title,
            hierarchy        = [section_title],
            is_key_section   = False,
            section_index    = 1,
            section_importance = "medium",
            retrieval_boost  = 1.0,
        ),
        content    = ElementContent(raw="Test content here.", token_count=4),
        phase2_ready = ElementPhase2Ready(
            action             = "merge_with_adjacent_text",
            text_for_embedding = "Test content here.",
            text_for_display   = "Test content here.",
        ),
        adjacency  = ElementAdjacency(),
    )

# Test skip sections
ref_elem  = make_test_element("References")
ack_elem  = make_test_element("Acknowledgements")
check(
    "References → metadata_only",
    get_section_treatment(ref_elem) == "metadata_only",
    f"got: {get_section_treatment(ref_elem)}"
)
check(
    "Acknowledgements → skip",
    get_section_treatment(ack_elem) == "skip",
    f"got: {get_section_treatment(ack_elem)}"
)

# Test high sections
for section in ["Abstract", "Introduction", "Methods", "Results", "Conclusion"]:
    elem = make_test_element(section)
    treatment = get_section_treatment(elem)
    check(
        f"'{section}' → process_high",
        treatment == "process_high",
        f"got: '{treatment}'"
    )

# Test numbered section stripping
elem_numbered = make_test_element("6.2 Model Variations")
t = get_section_treatment(elem_numbered)
check(
    "Numbered section normalized correctly",
    t in ("process_high", "process_medium"),
    f"'6.2 Model Variations' got: '{t}'"
)

# Test inline acknowledgement detection
inline_ack = make_test_element("7 Conclusion")
inline_ack.content.raw = "Acknowledgements We are grateful to XYZ."
t = get_section_treatment(inline_ack)
check(
    "Inline acknowledgement in Conclusion → skip",
    t == "skip",
    f"got: '{t}'"
)

# Test should_embed
check("should_embed process_high = True",   should_embed("process_high"),   "")
check("should_embed skip = False",          not should_embed("skip"),        "")
check("should_embed metadata_only = False", not should_embed("metadata_only"), "")

# Test boost values
check(
    "process_high boost = 1.1",
    get_boost_from_treatment("process_high") == 1.10,
    f"got: {get_boost_from_treatment('process_high')}"
)
check(
    "skip boost = 0.0",
    get_boost_from_treatment("skip") == 0.0,
    f"got: {get_boost_from_treatment('skip')}"
)


# ── 4. CHUNKER TEST ────────────────────────────────────────────────────────────
separator("TEST 4: Chunker — Load Real elements.json")

PAPER_DIR    = config.PARSED_DIR / "attention_is_all_you_need_2017"
ELEMENTS_PATH = PAPER_DIR / "elements.json"

check(
    f"elements.json exists at {ELEMENTS_PATH}",
    ELEMENTS_PATH.exists(),
    f"Put your elements.json at: {ELEMENTS_PATH}"
)

with open(ELEMENTS_PATH, "r", encoding="utf-8") as f:
    raw_json = json.load(f)

try:
    document = ParsedDocument.model_validate(raw_json)
    check(
        "elements.json validates against ParsedDocument schema",
        True,
        f"{len(document.elements)} elements loaded"
    )
except Exception as e:
    check("elements.json validates against ParsedDocument schema", False, str(e))

# Run chunker
try:
    chunks, references = build_chunks(document)
    check("build_chunks runs without error", True, f"{len(chunks)} chunks produced")
except Exception as e:
    check("build_chunks runs without error", False, str(e))

# Validate chunk outputs
check("At least 1 chunk produced",        len(chunks) > 0,  f"got {len(chunks)}")
check("References list is a list",        isinstance(references, list), "")

# Validate each chunk
bad_chunks = []
for c in chunks:
    issues = []
    if not c.chunk_id:
        issues.append("missing chunk_id")
    if not c.text_for_embedding:
        issues.append("empty text_for_embedding")
    if not c.text_for_display:
        issues.append("empty text_for_display")
    if c.token_count <= 0:
        issues.append(f"bad token_count: {c.token_count}")
    if not c.content_hash or len(c.content_hash) != 64:
        issues.append("bad content_hash")
    if not c.section_title:
        issues.append("missing section_title")

    # Hard limit only applies to text chunks
    # Tables, equations, figures are NEVER split
    # Their naturalized text can legitimately exceed 500 tokens
    is_self_contained = c.chunk_type in ("table", "equation", "figure")
    if not is_self_contained and c.token_count > config.CHUNK_HARD_LIMIT + 50:
        issues.append(
            f"text chunk exceeds hard limit: {c.token_count} tokens "
            f"(max {config.CHUNK_HARD_LIMIT})"
        )

    if issues:
        bad_chunks.append((c.chunk_id, issues))

check(
    "All chunks have valid required fields",
    len(bad_chunks) == 0,
    "\n       ".join([f"{cid}: {issues}" for cid, issues in bad_chunks[:3]])
)

# Check no References chunks were embedded
ref_chunks = [
    c for c in chunks
    if "references" in c.section_title.lower()
]
check(
    "No Reference section chunks in output",
    len(ref_chunks) == 0,
    f"Found {len(ref_chunks)} reference chunks: "
    f"{[c.chunk_id for c in ref_chunks[:3]]}"
)

# Check no Acknowledgement chunks
ack_chunks = [
    c for c in chunks
    if "acknowledgement" in c.text_for_embedding.lower()[:50]
]
check(
    "No Acknowledgement text in chunk embeddings",
    len(ack_chunks) == 0,
    f"Found {len(ack_chunks)} ack chunks"
)

# Check section boundaries respected
section_changes = []
for i in range(1, len(chunks)):
    prev_section = chunks[i-1].section_title
    curr_section = chunks[i].section_title
    if prev_section != curr_section:
        section_changes.append((prev_section, curr_section))

check(
    "Section boundaries produce section changes in chunks",
    len(section_changes) > 0,
    f"Found {len(section_changes)} section transitions"
)

# Print chunk stats
token_counts = [c.token_count for c in chunks]
print(f"\n  📊 Chunk Statistics:")
print(f"     Total chunks:    {len(chunks)}")
print(f"     Min tokens:      {min(token_counts)}")
print(f"     Max tokens:      {max(token_counts)}")
print(f"     Avg tokens:      {sum(token_counts)/len(token_counts):.1f}")
print(f"     Soft splits:     {sum(1 for c in chunks if c.soft_split)}")
print(f"     Tables (solo):   {sum(1 for c in chunks if c.chunk_type == 'table')}")
print(f"     Mixed chunks:    {sum(1 for c in chunks if c.chunk_type == 'mixed')}")
print(f"     Sections seen:   {len({c.section_title for c in chunks})}")
print(f"     References:      {len(references)} entries")

# Print first 3 chunks for visual inspection
print(f"\n  📄 First 3 chunks preview:")
for i, c in enumerate(chunks[:3]):
    print(f"\n  Chunk {i+1}: {c.chunk_id}")
    print(f"    Section:  {c.section_title}")
    print(f"    Tokens:   {c.token_count}")
    print(f"    Type:     {c.chunk_type}")
    print(f"    Modalities: {c.modalities}")
    print(f"    Text preview: {c.text_for_embedding[:100]}...")


# ── 5. QDRANT STORAGE TEST (no embeddings yet) ─────────────────────────────────
separator("TEST 5: Qdrant Storage — Connection and Collection")

try:
    storage = QdrantStorage()
    check("Qdrant connects successfully", True, "")
except Exception as e:
    check("Qdrant connects successfully", False, str(e))

# Verify collection exists
try:
    collections = [
        c.name for c in storage.client.get_collections().collections
    ]
    check(
        f"Collection '{config.QDRANT_COLLECTION}' exists",
        config.QDRANT_COLLECTION in collections,
        f"Found collections: {collections}"
    )
except Exception as e:
    check("Can list Qdrant collections", False, str(e))

# Verify indexes exist
try:
    info = storage.client.get_collection(config.QDRANT_COLLECTION)
    check(
        "Collection info retrievable",
        info is not None,
        f"vectors_count: {info.vectors_count}"
    )
except Exception as e:
    check("Collection info retrievable", False, str(e))


# ── 6. EMBEDDER TEST (small batch only) ───────────────────────────────────────
separator("TEST 6: Embedder — Dense + Sparse on 3 chunks")

print("  ⏳ Loading models (this takes 30-60 seconds first time)...")

try:
    from ingestion.embedder import DenseEmbedder, SparseEmbedder
    dense_embedder  = DenseEmbedder()
    check("DenseEmbedder loads", True, "")
except Exception as e:
    check("DenseEmbedder loads", False, str(e))

try:
    sparse_embedder = SparseEmbedder()
    check("SparseEmbedder (SPLADE) loads", True, "")
except Exception as e:
    check("SparseEmbedder loads", False, str(e))

# Test dense on 3 chunks
test_texts  = [c.text_for_embedding for c in chunks[:3]]
dense_vecs  = dense_embedder.embed(test_texts)

check(
    "Dense embedding returns 3 vectors",
    len(dense_vecs) == 3,
    f"got {len(dense_vecs)}"
)
check(
    f"Dense vector dimension = {config.EMBEDDING_DIM}",
    len(dense_vecs[0]) == config.EMBEDDING_DIM,
    f"got {len(dense_vecs[0])}"
)
check(
    "Dense vectors are non-zero",
    any(v != 0 for v in dense_vecs[0]),
    ""
)

# Check normalization (cosine similarity requires unit vectors)
import math
magnitude = math.sqrt(sum(v**2 for v in dense_vecs[0]))
check(
    "Dense vector is normalized (magnitude ≈ 1.0)",
    abs(magnitude - 1.0) < 0.01,
    f"magnitude = {magnitude:.4f}"
)

# Test sparse on 1 chunk
sparse_idx, sparse_val = sparse_embedder.embed_single(test_texts[0])
check(
    "Sparse embedding returns indices",
    len(sparse_idx) > 0,
    f"got {len(sparse_idx)} non-zero entries"
)
check(
    "Sparse indices and values same length",
    len(sparse_idx) == len(sparse_val),
    f"idx: {len(sparse_idx)}, val: {len(sparse_val)}"
)
check(
    "Sparse values are positive",
    all(v >= 0 for v in sparse_val),
    f"min value: {min(sparse_val):.4f}"
)

print(f"\n  📊 Embedding Statistics:")
print(f"     Dense dim:          {len(dense_vecs[0])}")
print(f"     Sparse non-zeros:   {len(sparse_idx)}")
print(f"     Sparse density:     "
      f"{len(sparse_idx)/30522*100:.2f}% of vocab")


# ── 7. FULL PIPELINE END-TO-END TEST (3 chunks only) ──────────────────────────
separator("TEST 7: Full Pipeline — 3 Chunks End to End")

from ingestion.embedder import Embedder

embedder = Embedder()
test_chunks_meta = chunks[:3]

print("  ⏳ Embedding 3 chunks (dense + sparse)...")
embedded = embedder.embed_chunks(test_chunks_meta)

check(
    "Embedder returns 3 Chunk objects",
    len(embedded) == 3,
    f"got {len(embedded)}"
)
check(
    "Each Chunk has dense_vector",
    all(c.dense_vector is not None for c in embedded),
    ""
)
check(
    "Each Chunk has sparse_indices",
    all(c.sparse_indices is not None for c in embedded),
    ""
)

# Upsert to Qdrant
print("  ⏳ Upserting 3 chunks to Qdrant...")
upserted = storage.upsert_chunks(embedded)
check(
    "Upsert runs without error",
    True,
    f"{upserted} points upserted"
)

# Verify they are in Qdrant
count_after = storage.client.count(
    collection_name = config.QDRANT_COLLECTION,
    exact           = True,
).count
check(
    "Qdrant has points after upsert",
    count_after > 0,
    f"total points in collection: {count_after}"
)

# Test deduplication: upsert same 3 again
upserted_again = storage.upsert_chunks(embedded)
check(
    "Duplicate upsert skipped (content-hash dedup)",
    upserted_again == 0,
    f"expected 0, got {upserted_again}"
)


# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
separator("ALL TESTS PASSED")
print("""
  The ingestion pipeline is working correctly.
  
  What was verified:
  ✅ All imports resolve
  ✅ Utility functions work correctly
  ✅ Section filter correctly skips/processes sections
  ✅ Chunker produces valid chunks from real elements.json
  ✅ No References or Acknowledgements in embedded chunks
  ✅ Section boundaries are respected
  ✅ Qdrant connects and collection exists
  ✅ Dense embeddings: correct dimension, normalized
  ✅ Sparse embeddings: correct format, positive values
  ✅ Full pipeline: chunk → embed → upsert works
  ✅ Deduplication prevents duplicate upserts

  Ready to build Phase 3: RAPTOR Tree Construction.
""")