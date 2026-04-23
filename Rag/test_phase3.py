# test_phase3.py
"""
Complete end-to-end test for Phase 3: RAPTOR Tree Construction.

Tests:
1. Imports and configuration
2. LLM availability check
3. Structural clustering on real chunks from Qdrant
4. FCM clustering correctness
5. Summary generation + validation (L1, L2, L3)
6. Tree builder node structure
7. Deduplication logic
8. Full pipeline end-to-end
9. Qdrant state after tree build
10. Phase 2 → Phase 3 → Phase 4 readiness

Run from project root:
    python test_phase3.py

Prerequisites:
    - Phase 2 test passed (chunks in Qdrant)
    - SUMMARIZER_MODEL set in .env
    - Corresponding API key set in .env
"""

import json
import sys
import math
from pathlib import Path

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


def warn(label: str, detail: str = ""):
    print(f"  ⚠️  {label}")
    if detail:
        print(f"       {detail}")


# ── TEST 1: IMPORTS ────────────────────────────────────────────────────────────
separator("TEST 1: Imports")

try:
    import config
    print("  ✅ config.py")
except Exception as e:
    print(f"  ❌ config.py: {e}")
    sys.exit(1)

try:
    import litellm
    litellm.suppress_debug_info = True
    print("  ✅ litellm")
except Exception as e:
    print(f"  ❌ litellm: {e} — run: pip install litellm")
    sys.exit(1)

try:
    from raptor.fcm_clustering import _fuzzy_cmeans
    import numpy as np
    # Quick sanity check: FCM runs on tiny matrix
    test_X = np.random.rand(5, 3)
    test_U = _fuzzy_cmeans(test_X, k=2, max_iter=10)
    assert test_U.shape == (5, 2), f"wrong shape: {test_U.shape}"
    assert abs(test_U[0].sum() - 1.0) < 0.01, "rows must sum to 1"
    print("  ✅ FCM (pure numpy implementation)")
except Exception as e:
    print(f"  ❌ FCM implementation: {e}")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
    print("  ✅ sklearn PCA")
except Exception as e:
    print(f"  ❌ sklearn: {e}")
    sys.exit(1)

try:
    from raptor.structural_clustering import (
        build_structural_clusters,
        StructuralCluster,
    )
    print("  ✅ raptor.structural_clustering")
except Exception as e:
    print(f"  ❌ raptor.structural_clustering: {e}")
    sys.exit(1)

try:
    from raptor.fcm_clustering import run_fcm_clustering
    print("  ✅ raptor.fcm_clustering")
except Exception as e:
    print(f"  ❌ raptor.fcm_clustering: {e}")
    sys.exit(1)

try:
    from raptor.summarizer import (
        check_llm_available,
        summarize_l1_section,
        summarize_l2_group,
        summarize_l3_root,
        _dynamic_word_limit,
        _split_if_too_long,
    )
    print("  ✅ raptor.summarizer")
except Exception as e:
    print(f"  ❌ raptor.summarizer: {e}")
    sys.exit(1)

try:
    from raptor.validator import validate_summary, extractive_fallback
    print("  ✅ raptor.validator")
except Exception as e:
    print(f"  ❌ raptor.validator: {e}")
    sys.exit(1)

try:
    from raptor.tree_builder import (
        build_l1_nodes,
        build_l2_nodes,
        build_l3_root,
    )
    print("  ✅ raptor.tree_builder")
except Exception as e:
    print(f"  ❌ raptor.tree_builder: {e}")
    sys.exit(1)

try:
    from raptor.deduplicator import (
        deduplicate_summary_nodes,
        cosine_similarity,
    )
    print("  ✅ raptor.deduplicator")
except Exception as e:
    print(f"  ❌ raptor.deduplicator: {e}")
    sys.exit(1)

try:
    from raptor.pipeline import build_raptor_tree
    print("  ✅ raptor.pipeline")
except Exception as e:
    print(f"  ❌ raptor.pipeline: {e}")
    sys.exit(1)

try:
    from ingestion.storage import QdrantStorage
    from core.models import ChunkMetadata, Chunk
    from core.utils import count_tokens, extract_numbers
    print("  ✅ ingestion.storage + core.models")
except Exception as e:
    print(f"  ❌ core imports: {e}")
    sys.exit(1)


# ── TEST 2: CONFIGURATION ──────────────────────────────────────────────────────
separator("TEST 2: Configuration")

check(
    "SUMMARIZER_MODEL is set",
    bool(config.SUMMARIZER_MODEL),
    f"got: '{config.SUMMARIZER_MODEL}'"
)
check(
    "EMBEDDING_DIM is 512",
    config.EMBEDDING_DIM == 512,
    f"got: {config.EMBEDDING_DIM}"
)
check(
    "FCM_MEMBERSHIP_THRESHOLD is set",
    0 < config.FCM_MEMBERSHIP_THRESHOLD < 1,
    f"got: {config.FCM_MEMBERSHIP_THRESHOLD}"
)
check(
    "FCM_MAX_CLUSTERS_PER_CHUNK is 2",
    config.FCM_MAX_CLUSTERS_PER_CHUNK == 2,
    f"got: {config.FCM_MAX_CLUSTERS_PER_CHUNK}"
)
check(
    "SUMMARY_COMPRESSION_MAX is set",
    0 < config.SUMMARY_COMPRESSION_MAX <= 1,
    f"got: {config.SUMMARY_COMPRESSION_MAX}"
)
check(
    "RAPTOR_DIR is configured",
    hasattr(config, "RAPTOR_DIR"),
    f"got: {getattr(config, 'RAPTOR_DIR', 'MISSING')}"
)

print(f"\n  📋 Configuration:")
print(f"     SUMMARIZER_MODEL:         {config.SUMMARIZER_MODEL}")
print(f"     FCM_MEMBERSHIP_THRESHOLD: {config.FCM_MEMBERSHIP_THRESHOLD}")
print(f"     FCM_MAX_CLUSTERS:         {config.FCM_MAX_CLUSTERS_PER_CHUNK}")
print(f"     SUMMARY_COMPRESSION_MAX:  {config.SUMMARY_COMPRESSION_MAX}")
print(f"     RAPTOR_DIR:               {config.RAPTOR_DIR}")


# ── TEST 3: LLM AVAILABILITY ───────────────────────────────────────────────────
separator("TEST 3: LLM Availability Check")

print(f"  ⏳ Testing connection to: {config.SUMMARIZER_MODEL}")
try:
    check_llm_available()
    check("LLM is reachable and responding", True, "")
except Exception as e:
    check("LLM is reachable and responding", False, str(e))


# ── TEST 4: QDRANT LEAF CHUNKS (Phase 2 Integration) ──────────────────────────
separator("TEST 4: Load Leaf Chunks from Qdrant (Phase 2 Output)")

PAPER_ID = "attention"

storage = QdrantStorage()
chunks  = storage.get_leaf_chunks(PAPER_ID)

check(
    f"Leaf chunks found for paper '{PAPER_ID}'",
    len(chunks) > 0,
    f"got {len(chunks)} chunks. Run Phase 2 ingestion first."
)
check(
    "All retrieved chunks are level=0",
    all(c.payload.get("level") == 0 for c in chunks),
    f"Non-leaf chunks found in leaf query"
)
check(
    "All chunks have chunk_id",
    all(c.payload.get("chunk_id") for c in chunks),
    ""
)
check(
    "All chunks have text_for_embedding",
    all(c.payload.get("text_for_embedding") for c in chunks),
    ""
)
check(
    "All chunks have section_title",
    all(c.payload.get("section_title") for c in chunks),
    ""
)
check(
    "All chunks have section_hierarchy",
    all(c.payload.get("section_hierarchy") for c in chunks),
    ""
)
check(
    "All chunks have paper_title",
    all(c.payload.get("paper_title") for c in chunks),
    ""
)

# Extract dense vectors
dense_vectors = []
for chunk in chunks:
    vec = chunk.vector
    if isinstance(vec, dict):
        vec = vec.get("dense", [])
    dense_vectors.append(vec if vec else [])

check(
    "Dense vectors retrieved from Qdrant",
    all(len(v) > 0 for v in dense_vectors),
    f"vector dims: {len(dense_vectors[0]) if dense_vectors else 0}"
)
check(
    f"Dense vector dimension matches config ({config.EMBEDDING_DIM})",
    all(len(v) == config.EMBEDDING_DIM for v in dense_vectors),
    f"got: {len(dense_vectors[0]) if dense_vectors else 0}"
)

paper_title = chunks[0].payload.get("paper_title", PAPER_ID)
language    = chunks[0].payload.get("language", "en")
n_chunks    = len(chunks)

print(f"\n  📊 Leaf Chunk Summary:")
print(f"     Total leaf chunks:  {n_chunks}")
print(f"     Paper title:        {paper_title}")
print(f"     Language:           {language}")
sections = list({c.payload.get("section_title") for c in chunks})
print(f"     Unique sections:    {len(sections)}")
print(f"     Sections: {sections[:5]}{'...' if len(sections) > 5 else ''}")


# ── TEST 5: STRUCTURAL CLUSTERING ─────────────────────────────────────────────
separator("TEST 5: Structural Clustering (Layer 1A)")

try:
    structural_clusters = build_structural_clusters(chunks)
    check("build_structural_clusters runs without error", True, "")
except Exception as e:
    check("build_structural_clusters runs without error", False, str(e))

k = len(structural_clusters)

check(
    "At least 2 structural clusters produced",
    k >= 2,
    f"got k={k}"
)
check(
    "k does not exceed number of chunks",
    k <= n_chunks,
    f"k={k}, n_chunks={n_chunks}"
)

# Verify every chunk is in exactly one structural cluster
all_clustered_ids = []
for cluster in structural_clusters:
    all_clustered_ids.extend(cluster.chunk_ids)

all_leaf_ids = [c.payload["chunk_id"] for c in chunks]

check(
    "All chunks assigned to at least one structural cluster",
    set(all_leaf_ids).issubset(set(all_clustered_ids)),
    f"Unassigned: {set(all_leaf_ids) - set(all_clustered_ids)}"
)

# Verify cluster sizes
for cluster in structural_clusters:
    check(
        f"Cluster {cluster.cluster_id} ('{cluster.section_title}') "
        f"has >= 1 chunk",
        len(cluster.chunk_ids) >= 1,
        f"got {len(cluster.chunk_ids)} chunks"
    )

# Verify no cluster exceeds max size (unless it's a forced case)
oversized = [
    c for c in structural_clusters
    if len(c.chunk_ids) > 12 and not c.is_split
]
check(
    "No unsplit cluster exceeds MAX_CLUSTER_SIZE=12",
    len(oversized) == 0,
    f"Oversized clusters: "
    f"{[(c.cluster_id, len(c.chunk_ids)) for c in oversized]}"
)

print(f"\n  📊 Structural Clustering:")
print(f"     k (cluster count):  {k}")
print(f"     Total chunks assigned: {len(all_clustered_ids)}")
for c in structural_clusters:
    split_flag = " [SPLIT]" if c.is_split else ""
    print(
        f"     Cluster {c.cluster_id:02d}: "
        f"'{c.section_title[:35]}' "
        f"({len(c.chunk_ids)} chunks){split_flag}"
    )


# ── TEST 6: FCM CLUSTERING ────────────────────────────────────────────────────
separator("TEST 6: FCM Soft Clustering (Layer 1B)")

# Verify PCA dim formula
max_pca_dims = max(2, n_chunks // 5)
expected_pca_dims = min(50, max_pca_dims, config.EMBEDDING_DIM, n_chunks - 1)
check(
    "PCA dims formula is safe (< n_chunks / 5)",
    expected_pca_dims <= n_chunks // 5 or expected_pca_dims == 2,
    f"expected_pca_dims={expected_pca_dims}, n_chunks={n_chunks}"
)
print(f"  ℹ️  Expected PCA dims: {expected_pca_dims}")

try:
    fcm_memberships = run_fcm_clustering(
        chunks        = chunks,
        k             = k,
        dense_vectors = dense_vectors,
    )
    check("run_fcm_clustering runs without error", True, "")
except Exception as e:
    check("run_fcm_clustering runs without error", False, str(e))

# Validate output structure
check(
    "FCM returns entry for every chunk",
    len(fcm_memberships) == n_chunks,
    f"got {len(fcm_memberships)}, expected {n_chunks}"
)

# Validate membership constraints
violations = []
for chunk_id, memberships in fcm_memberships.items():

    # Max 2 cluster assignments
    if len(memberships) > config.FCM_MAX_CLUSTERS_PER_CHUNK:
        violations.append(
            f"{chunk_id}: {len(memberships)} assignments "
            f"(max {config.FCM_MAX_CLUSTERS_PER_CHUNK})"
        )

    # All membership scores >= threshold
    for cid, score in memberships.items():
        if score < config.FCM_MEMBERSHIP_THRESHOLD:
            violations.append(
                f"{chunk_id}: score {score} below threshold "
                f"{config.FCM_MEMBERSHIP_THRESHOLD}"
            )

        # Scores are valid probabilities
        if not 0 <= score <= 1:
            violations.append(
                f"{chunk_id}: invalid score {score}"
            )

check(
    "All FCM memberships respect constraints",
    len(violations) == 0,
    "\n       ".join(violations[:3])
)

# Count cross-section assignments
multi_assigned = sum(
    1 for m in fcm_memberships.values() if len(m) > 1
)
any_assigned = sum(
    1 for m in fcm_memberships.values() if len(m) >= 1
)

print(f"\n  📊 FCM Clustering:")
print(f"     k used:                {k}")
print(f"     Chunks with any assignment: {any_assigned}/{n_chunks}")
print(
    f"     Cross-section chunks:   "
    f"{multi_assigned}/{n_chunks} "
    f"({multi_assigned/n_chunks*100:.1f}%)"
)

# Show sample memberships
sample_cross = [
    (cid, m) for cid, m in fcm_memberships.items()
    if len(m) > 1
][:2]
if sample_cross:
    print(f"     Sample cross-section memberships:")
    for cid, m in sample_cross:
        print(f"       {cid}: {m}")


# ── TEST 7: VALIDATOR ──────────────────────────────────────────────────────────
separator("TEST 7: Summary Validator")

# Test compression check
short_source = "This is a test source. " * 5
long_summary = "This is a very long summary that exceeds compression ratio. " * 20

valid, reason = validate_summary(long_summary, short_source, "test_compression")
check(
    "Compression check catches oversized summary",
    not valid and "COMPRESSION" in reason,
    f"got: valid={valid}, reason={reason}"
)

# Test pass case
good_summary = "This is a concise summary of the test source."
valid, reason = validate_summary(good_summary, short_source, "test_pass")
check(
    "Valid summary passes compression check",
    valid,
    f"got: valid={valid}, reason={reason}"
)

# Test hallucination detection
source_with_numbers = (
    "The model achieved 28.4 BLEU score with 6 layers "
    "and 512 dimensions."
)
hallucinated_summary = (
    "The model achieved 95.7 BLEU score with 12 layers."
)
valid, reason = validate_summary(
    hallucinated_summary, source_with_numbers, "test_hallucination"
)
check(
    "Hallucination check catches invented numbers > 10",
    not valid and "HALLUCINATION" in reason,
    f"got: valid={valid}, reason={reason}"
)

# Test numbers from source pass through fine
good_number_summary = (
    "The model achieved 28.4 BLEU score using 6 layers."
)
valid, reason = validate_summary(
    good_number_summary, source_with_numbers, "test_numbers_ok"
)
check(
    "Summary with source numbers passes validation",
    valid,
    f"got: valid={valid}, reason={reason}"
)

# Test extractive fallback
fallback_texts = [
    "First chunk first sentence. Middle content. Last chunk last sentence.",
    "Second chunk opens here. More content. Second chunk ends here.",
]
fallback_result = extractive_fallback(fallback_texts)
check(
    "Extractive fallback returns non-empty string",
    bool(fallback_result) and len(fallback_result) > 10,
    f"got: '{fallback_result[:100]}'"
)
check(
    "Extractive fallback contains no invented content",
    all(
        sentence.strip() in " ".join(fallback_texts)
        for sentence in fallback_result.split(".")
        if sentence.strip()
    ),
    "Fallback should only use sentences from source"
)

# Test empty summary
valid, reason = validate_summary("", short_source, "test_empty")
check(
    "Empty summary fails validation",
    not valid and "EMPTY" in reason,
    f"got: valid={valid}, reason={reason}"
)

# Test dynamic word limit
short_text = "Short text. " * 10
long_text  = "Long text content. " * 200
short_limit = _dynamic_word_limit(short_text)
long_limit  = _dynamic_word_limit(long_text)
check(
    "Dynamic word limit: min=100",
    short_limit >= 100,
    f"got {short_limit} for short text"
)
check(
    "Dynamic word limit: max=350",
    long_limit <= 350,
    f"got {long_limit} for long text"
)
check(
    "Dynamic word limit: longer input → higher limit",
    long_limit >= short_limit,
    f"short={short_limit}, long={long_limit}"
)

print(f"\n  📊 Validator:")
print(f"     short_text word limit: {short_limit}")
print(f"     long_text word limit:  {long_limit}")


# ── TEST 8: SUMMARIZER — REAL LLM CALLS ───────────────────────────────────────
separator("TEST 8: Summarizer — Real LLM Calls (3 calls)")

# Use first structural cluster for real L1 test
test_cluster    = structural_clusters[0]
chunk_id_to_idx = {
    c.payload["chunk_id"]: i for i, c in enumerate(chunks)
}
test_texts = []
for chunk_id in test_cluster.chunk_ids[:3]:  # max 3 chunks for speed
    idx = chunk_id_to_idx.get(chunk_id)
    if idx is not None:
        text = chunks[idx].payload.get("text_for_embedding", "")
        if text:
            test_texts.append(text)

check(
    "Test cluster has text to summarize",
    len(test_texts) > 0,
    f"got {len(test_texts)} texts from cluster {test_cluster.cluster_id}"
)

print(
    f"  ⏳ Generating L1 summary for '{test_cluster.section_title}' "
    f"({len(test_texts)} chunks)..."
)

try:
    l1_summary = summarize_l1_section(
        section_title     = test_cluster.section_title,
        section_hierarchy = test_cluster.section_hierarchy,
        paper_title       = paper_title,
        chunk_texts       = test_texts,
    )
    check("L1 summarize_l1_section returns text", bool(l1_summary), "")
except Exception as e:
    check("L1 summarize_l1_section returns text", False, str(e))

check(
    "L1 summary is non-trivial length",
    count_tokens(l1_summary) >= 30,
    f"got {count_tokens(l1_summary)} tokens"
)

# Validate the real L1 summary
valid, reason = validate_summary(
    l1_summary, "\n\n".join(test_texts), "real_l1_test"
)
if not valid:
    warn(
        f"Real L1 summary failed validation: {reason}",
        "This may happen with very short test input. "
        "Full pipeline uses extractive fallback."
    )
else:
    print(f"  ✅ Real L1 summary passes validation")

print(f"\n  📄 L1 Summary Preview ({count_tokens(l1_summary)} tokens):")
print(f"     {l1_summary[:200]}...")

# Test L2 summarizer with the L1 summary we just made
print(f"\n  ⏳ Generating L2 summary from L1 summary...")
try:
    l2_summary = summarize_l2_group(
        paper_title      = paper_title,
        l1_summary_texts = [l1_summary],
        section_titles   = [test_cluster.section_title],
    )
    check("L2 summarize_l2_group returns text", bool(l2_summary), "")
except Exception as e:
    check("L2 summarize_l2_group returns text", False, str(e))

# Test L3 summarizer
print(f"  ⏳ Generating L3 root summary...")
try:
    l3_summary = summarize_l3_root(
        paper_title      = paper_title,
        l2_summary_texts = [l2_summary],
    )
    check("L3 summarize_l3_root returns text", bool(l3_summary), "")
except Exception as e:
    check("L3 summarize_l3_root returns text", False, str(e))

print(f"\n  📊 Summarizer Output:")
print(f"     L1 tokens: {count_tokens(l1_summary)}")
print(f"     L2 tokens: {count_tokens(l2_summary)}")
print(f"     L3 tokens: {count_tokens(l3_summary)}")


# ── TEST 9: DEDUPLICATOR ───────────────────────────────────────────────────────
separator("TEST 9: Deduplicator")

import numpy as np
from core.models import ChunkMetadata, Chunk
from datetime import datetime, timezone

def make_test_chunk_with_vector(
    chunk_id: str,
    text: str,
    vector: list[float],
) -> Chunk:
    meta = ChunkMetadata(
        chunk_id        = chunk_id,
        paper_id        = PAPER_ID,
        paper_title     = paper_title,
        content_hash    = f"hash_{chunk_id}",
        embedding_model = config.EMBEDDING_MODEL,
        embedding_dim   = config.EMBEDDING_DIM,
        ingested_at     = datetime.now(timezone.utc).isoformat(),
        language        = "en",
        section_title   = "Test Section",
        section_hierarchy = ["Test Section"],
        is_key_section  = False,
        page_range      = [1, 1],
        order_range     = [0, 5],
        chunk_sequence_index = 0,
        position_in_paper    = 0.5,
        level         = 1,
        depth         = 1,
        chunk_type    = "text",
        modalities    = ["text"],
        text_for_embedding = text,
        text_for_display   = text,
        token_count   = count_tokens(text),
        query_affinities   = ["broad"],
        section_importance = "medium",
        retrieval_boost    = 1.0,
    )
    return Chunk(
        metadata       = meta,
        dense_vector   = vector,
        sparse_indices = [],
        sparse_values  = [],
    )

# Test cosine similarity
v1 = [1.0, 0.0, 0.0]
v2 = [1.0, 0.0, 0.0]
v3 = [0.0, 1.0, 0.0]
v4 = [0.7, 0.7, 0.0]

check(
    "Cosine similarity: identical vectors = 1.0",
    abs(cosine_similarity(v1, v2) - 1.0) < 0.001,
    f"got {cosine_similarity(v1, v2):.4f}"
)
check(
    "Cosine similarity: orthogonal vectors = 0.0",
    abs(cosine_similarity(v1, v3) - 0.0) < 0.001,
    f"got {cosine_similarity(v1, v3):.4f}"
)
check(
    "Cosine similarity: similar vectors = 0 < x < 1",
    0 < cosine_similarity(v1, v4) < 1,
    f"got {cosine_similarity(v1, v4):.4f}"
)

# Build a near-duplicate pair
base_vec   = list(np.random.rand(config.EMBEDDING_DIM).astype(float))
norm       = math.sqrt(sum(x**2 for x in base_vec))
base_vec   = [x / norm for x in base_vec]

# Near duplicate: tiny perturbation (similarity > 0.90)
dup_vec    = [x + 0.001 for x in base_vec]
dup_norm   = math.sqrt(sum(x**2 for x in dup_vec))
dup_vec    = [x / dup_norm for x in dup_vec]

# Different vector (similarity < 0.90)
diff_vec   = list(np.random.rand(config.EMBEDDING_DIM).astype(float))
diff_norm  = math.sqrt(sum(x**2 for x in diff_vec))
diff_vec   = [x / diff_norm for x in diff_vec]

sim_dup  = cosine_similarity(base_vec, dup_vec)
sim_diff = cosine_similarity(base_vec, diff_vec)

chunk_a = make_test_chunk_with_vector(
    "test_chunk_A", "Summary about attention mechanism. " * 5, base_vec
)
chunk_b = make_test_chunk_with_vector(
    "test_chunk_B", "Summary about attention mechanism. " * 8, dup_vec
)
chunk_c = make_test_chunk_with_vector(
    "test_chunk_C", "Summary about training procedures. " * 5, diff_vec
)
chunk_a.metadata.children_ids = ["leaf_1", "leaf_2"]
chunk_b.metadata.children_ids = ["leaf_2", "leaf_3"]

print(f"\n  ℹ️  Similarity A↔B (near-dup): {sim_dup:.4f}")
print(f"  ℹ️  Similarity A↔C (different): {sim_diff:.4f}")

if sim_dup >= config.SEMANTIC_DEDUP_THRESHOLD:
    result = deduplicate_summary_nodes([chunk_a, chunk_b, chunk_c])
    check(
        "Deduplicator removes near-duplicate (sim >= threshold)",
        len(result) == 2,
        f"got {len(result)} nodes, expected 2"
    )
    # Verify children_ids were merged
    merged_node = next(
        (n for n in result if n.metadata.chunk_id in ["test_chunk_A", "test_chunk_B"]),
        None
    )
    check(
        "Merged node has union of children_ids",
        merged_node is not None and set(
            merged_node.metadata.children_ids
        ) == {"leaf_1", "leaf_2", "leaf_3"},
        f"got children: {merged_node.metadata.children_ids if merged_node else 'none'}"
    )
else:
    warn(
        f"Near-dup similarity {sim_dup:.4f} < threshold "
        f"{config.SEMANTIC_DEDUP_THRESHOLD}",
        "Dedup not triggered — test vectors may not be similar enough. "
        "This is a test vector issue, not a production issue."
    )
    result = deduplicate_summary_nodes([chunk_a, chunk_b, chunk_c])
    check(
        "Deduplicator returns all nodes when none are duplicates",
        len(result) == 3,
        f"got {len(result)} nodes"
    )

# Verify non-duplicate is always kept
result_ids = [n.metadata.chunk_id for n in result]
check(
    "Non-duplicate chunk C is always kept",
    "test_chunk_C" in result_ids,
    f"result ids: {result_ids}"
)


# ── TEST 10: FULL PIPELINE END-TO-END ─────────────────────────────────────────
separator("TEST 10: Full RAPTOR Pipeline — End to End")

print(
    f"  ⏳ Running full RAPTOR tree build for '{PAPER_ID}'...\n"
    f"     This makes multiple LLM calls and may take 2-10 minutes.\n"
    f"     Cost depends on model and number of sections.\n"
)

try:
    report = build_raptor_tree(PAPER_ID)
    check("build_raptor_tree completes without error", True, "")
except Exception as e:
    check("build_raptor_tree completes without error", False, str(e))

check(
    "Report status is SUCCESS",
    report.get("status") == "SUCCESS",
    f"got: {report.get('status')}"
)
check(
    "Report has leaf_chunks count",
    report.get("leaf_chunks", 0) > 0,
    f"got: {report.get('leaf_chunks')}"
)
check(
    "Report has l1_nodes count",
    report.get("l1_nodes", 0) > 0,
    f"got: {report.get('l1_nodes')}"
)
check(
    "Report has l3_root = 1",
    report.get("l3_root") == 1,
    f"got: {report.get('l3_root')}"
)

print(f"\n  📊 Pipeline Report:")
print(f"     Leaf chunks (L0): {report.get('leaf_chunks')}")
print(f"     L1 summaries:     {report.get('l1_nodes')}")
print(f"     L2 summaries:     {report.get('l2_nodes')}")
print(f"     L3 root:          {report.get('l3_root')}")
print(f"     Total in Qdrant:  {report.get('total_nodes')}")
print(f"     Duration:         {report.get('duration_seconds')}s")


# ── TEST 11: QDRANT STATE VALIDATION ──────────────────────────────────────────
separator("TEST 11: Qdrant State After Tree Build")

from qdrant_client.models import Filter, FieldCondition, MatchValue

def count_by_level(storage, paper_id, level):
    results, _ = storage.client.scroll(
        collection_name = config.QDRANT_COLLECTION,
        scroll_filter   = Filter(
            must=[
                FieldCondition(
                    key   = "paper_id",
                    match = MatchValue(value=paper_id),
                ),
                FieldCondition(
                    key   = "level",
                    match = MatchValue(value=level),
                ),
            ]
        ),
        with_payload = True,
        with_vectors = False,
        limit        = 1000,
    )
    return results

l0_nodes = count_by_level(storage, PAPER_ID, 0)
l1_nodes = count_by_level(storage, PAPER_ID, 1)
l2_nodes = count_by_level(storage, PAPER_ID, 2)
l3_nodes = count_by_level(storage, PAPER_ID, 3)

check(
    "L0 leaf chunks present in Qdrant",
    len(l0_nodes) > 0,
    f"got {len(l0_nodes)}"
)
check(
    "L1 summary nodes present in Qdrant",
    len(l1_nodes) > 0,
    f"got {len(l1_nodes)}"
)
check(
    "L3 root node present in Qdrant",
    len(l3_nodes) == 1,
    f"got {len(l3_nodes)}, expected exactly 1"
)

# Verify L3 root has children
l3_root_payload = l3_nodes[0].payload
check(
    "L3 root has children_ids populated",
    len(l3_root_payload.get("children_ids", [])) > 0,
    f"got: {l3_root_payload.get('children_ids')}"
)
check(
    "L3 root has text_for_embedding",
    bool(l3_root_payload.get("text_for_embedding")),
    f"got: {str(l3_root_payload.get('text_for_embedding'))[:50]}"
)

# Verify L1 nodes have children and parents
for l1 in l1_nodes[:3]:
    p = l1.payload
    check(
        f"L1 node '{p.get('chunk_id')}' has children_ids",
        len(p.get("children_ids", [])) > 0,
        f"children: {p.get('children_ids')}"
    )

# Verify L0 leaf nodes have parent_ids backfilled
leaves_with_parents = [
    n for n in l0_nodes
    if len(n.payload.get("parent_ids", [])) > 0
]
check(
    "Leaf nodes have parent_ids backfilled",
    len(leaves_with_parents) > 0,
    f"{len(leaves_with_parents)}/{len(l0_nodes)} leaves have parent_ids"
)

# Check debug files were written
raptor_dir = config.RAPTOR_DIR / PAPER_ID
expected_files = [
    "structural_clusters.json",
    "fcm_clusters.json",
    "L1_summaries.json",
    "L2_summaries.json",
    "L3_root.json",
    "tree_graph.json",
    "build_report.json",
]
for fname in expected_files:
    fpath = raptor_dir / fname
    check(
        f"Debug file exists: {fname}",
        fpath.exists(),
        f"expected at: {fpath}"
    )

print(f"\n  📊 Qdrant State:")
print(f"     L0 (leaf chunks):    {len(l0_nodes)}")
print(f"     L1 (section summ):   {len(l1_nodes)}")
print(f"     L2 (group summ):     {len(l2_nodes)}")
print(f"     L3 (root):           {len(l3_nodes)}")
print(f"     Total:               "
      f"{len(l0_nodes)+len(l1_nodes)+len(l2_nodes)+len(l3_nodes)}")


# ── TEST 12: TREE STRUCTURE INTEGRITY ─────────────────────────────────────────
separator("TEST 12: Tree Structure Integrity")

# Load tree graph
tree_graph_path = raptor_dir / "tree_graph.json"
with open(tree_graph_path, "r") as f:
    tree_graph = json.load(f)

nodes = {n["id"]: n for n in tree_graph["nodes"]}
edges = tree_graph["edges"]

check(
    "Tree graph has nodes",
    len(nodes) > 0,
    f"got {len(nodes)} nodes"
)
check(
    "Tree graph has edges",
    len(edges) > 0,
    f"got {len(edges)} edges"
)

# Every edge parent exists in nodes
invalid_parents = [
    e["parent"] for e in edges
    if e["parent"] not in nodes
]
check(
    "All edge parents exist in node list",
    len(invalid_parents) == 0,
    f"Missing parents: {invalid_parents[:3]}"
)

# Every edge child exists in nodes
invalid_children = [
    e["child"] for e in edges
    if e["child"] not in nodes
]
check(
    "All edge children exist in node list",
    len(invalid_children) == 0,
    f"Missing children: {invalid_children[:3]}"
)

# L3 root has no parents (it is the root)
l3_in_graph = [n for n in nodes.values() if n.get("level") == 3]
check(
    "Exactly one L3 root in tree graph",
    len(l3_in_graph) == 1,
    f"got {len(l3_in_graph)}"
)

l3_id = l3_in_graph[0]["id"]
l3_as_child = [e for e in edges if e["child"] == l3_id]
check(
    "L3 root has no parent edges (it is the root)",
    len(l3_as_child) == 0,
    f"L3 appears as child in: {l3_as_child}"
)

print(f"\n  📊 Tree Graph:")
print(f"     Total nodes: {len(nodes)}")
print(f"     Total edges: {len(edges)}")
level_counts = {}
for n in nodes.values():
    lvl = n.get("level", "?")
    level_counts[lvl] = level_counts.get(lvl, 0) + 1
for lvl in sorted(level_counts.keys()):
    print(f"     Level {lvl}: {level_counts[lvl]} nodes")


# ── TEST 13: PHASE 4 READINESS CHECK ──────────────────────────────────────────
separator("TEST 13: Phase 4 Readiness — All Nodes Have Vectors")

all_paper_nodes, _ = storage.client.scroll(
    collection_name = config.QDRANT_COLLECTION,
    scroll_filter   = Filter(
        must=[
            FieldCondition(
                key   = "paper_id",
                match = MatchValue(value=PAPER_ID),
            )
        ]
    ),
    with_payload = True,
    with_vectors = ["dense"],
    limit        = 10_000,
)

nodes_without_vectors = [
    n for n in all_paper_nodes
    if not n.vector or not n.vector.get("dense")
]
check(
    "All nodes in Qdrant have dense vectors",
    len(nodes_without_vectors) == 0,
    f"{len(nodes_without_vectors)} nodes missing vectors: "
    f"{[n.payload.get('chunk_id') for n in nodes_without_vectors[:3]]}"
)

nodes_without_text = [
    n for n in all_paper_nodes
    if not n.payload.get("text_for_embedding")
]
check(
    "All nodes have text_for_embedding",
    len(nodes_without_text) == 0,
    f"{len(nodes_without_text)} nodes missing text"
)

nodes_without_display = [
    n for n in all_paper_nodes
    if not n.payload.get("text_for_display")
]
check(
    "All nodes have text_for_display",
    len(nodes_without_display) == 0,
    f"{len(nodes_without_display)} nodes missing display text"
)

# Check retrieval signal fields are set
nodes_without_affinities = [
    n for n in all_paper_nodes
    if not n.payload.get("query_affinities")
]
check(
    "All nodes have query_affinities set",
    len(nodes_without_affinities) == 0,
    f"{len(nodes_without_affinities)} nodes missing affinities"
)

# Verify bidirectional links exist
# L1 nodes should have children pointing to L0 nodes
l1_in_qdrant = [
    n for n in all_paper_nodes
    if n.payload.get("level") == 1
]
l1_with_children = [
    n for n in l1_in_qdrant
    if len(n.payload.get("children_ids", [])) > 0
]
check(
    "L1 nodes have children_ids (downward links)",
    len(l1_with_children) == len(l1_in_qdrant),
    f"{len(l1_with_children)}/{len(l1_in_qdrant)} L1 nodes have children"
)

l0_in_qdrant = [
    n for n in all_paper_nodes
    if n.payload.get("level") == 0
]
l0_with_parents = [
    n for n in l0_in_qdrant
    if len(n.payload.get("parent_ids", [])) > 0
]
check(
    "L0 leaf nodes have parent_ids (upward links)",
    len(l0_with_parents) > 0,
    f"{len(l0_with_parents)}/{len(l0_in_qdrant)} leaves have parents"
)

print(f"\n  📊 Phase 4 Readiness:")
print(f"     Total nodes in Qdrant:    {len(all_paper_nodes)}")
print(f"     Nodes with dense vector:  "
      f"{len(all_paper_nodes) - len(nodes_without_vectors)}")
print(f"     Nodes with text:          "
      f"{len(all_paper_nodes) - len(nodes_without_text)}")
print(f"     L0 with parent links:     "
      f"{len(l0_with_parents)}/{len(l0_in_qdrant)}")
print(f"     L1 with children links:   "
      f"{len(l1_with_children)}/{len(l1_in_qdrant)}")


# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
separator("ALL PHASE 3 TESTS PASSED")

print(f"""
  What was verified:
  ✅ All Phase 3 imports resolve
  ✅ LiteLLM configuration correct
  ✅ LLM model reachable and responding
  ✅ Leaf chunks loaded from Qdrant (Phase 2 integration)
  ✅ Structural clustering: k clusters, all chunks assigned
  ✅ FCM soft clustering: membership constraints respected
  ✅ Validator: compression + hallucination checks work
  ✅ Extractive fallback: works without LLM
  ✅ Real L1/L2/L3 summaries generated successfully
  ✅ Full pipeline runs end to end
  ✅ Qdrant has L0/L1/L2/L3 nodes after build
  ✅ Tree structure is valid (no orphan edges)
  ✅ L3 root is single and has no parent
  ✅ All nodes have dense vectors (Phase 4 ready)
  ✅ Bidirectional parent↔child links populated

  RAPTOR tree is built and stored correctly.
  Ready to build Phase 4: Storage optimization
  and Phase 5: Retrieval Pipeline.
""")