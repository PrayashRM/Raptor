# scripts/test_retrieval.py
"""
Quick manual test for retrieval pipeline.
Run from project root:
    python scripts/test_retrieval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.pipeline import retrieve
from core.logger import get_logger

logger = get_logger("test_retrieval")

PAPER_ID = "attention"

TEST_QUERIES = [
    # Mode A: specific
    "what BLEU score did the transformer achieve on WMT 2014",
    "how many attention heads does the base model use",
    "what is the dropout rate used in training",

    # Mode B: broad
    "explain the main contribution of this paper",
    "what problem does the transformer solve",
    "summarize the paper",

    # Mode C: balanced
    "how does self-attention work",
    "what is positional encoding",

    # Mode D: modality
    "show me the attention formula",
    "what does Table 2 show",
]


def main():
    print("\n" + "="*60)
    print("RETRIEVAL PIPELINE TEST")
    print("="*60)
    print(f"Paper: {PAPER_ID}")
    print("Loading models (30-60 seconds)...\n")

    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        result = retrieve(query=query, paper_id=PAPER_ID)

        print(f"Mode:       {result.mode}")
        print(f"Candidates: {result.total_candidates}")
        print(f"Reranked:   {result.after_rerank}")
        print(f"Final:      {result.after_traversal} chunks")

        if result.final_chunks:
            print(f"\nTop chunks:")
            for i, chunk in enumerate(result.final_chunks[:3]):
                print(
                    f"  [{i+1}] L{chunk.get('level')} "
                    f"section='{chunk.get('section_title', '')[:35]}' "
                    f"tokens={chunk.get('token_count')} "
                    f"rerank={chunk.get('_rerank_score', 0):.3f} "
                    f"fetched_as={chunk.get('_fetched_as', 'direct')}"
                )
                text = chunk.get('text_for_display', '')
                print(f"       preview: {text[:100]}...")
        else:
            print("  No chunks retrieved")


if __name__ == "__main__":
    main()