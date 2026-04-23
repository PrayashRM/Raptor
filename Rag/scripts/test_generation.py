# scripts/test_generation.py
"""
Quick end-to-end generation test.
Run from project root:
    python scripts/test_generation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.pipeline import generate_answer

PAPER_ID    = "attention"
PAPER_TITLE = "Attention Is All You Need"

TEST_QUERIES = [
    "what BLEU score did the transformer achieve on WMT 2014",
    "explain the main contribution of this paper",
    "what is the attention formula",
    "how many attention heads does the base model use",
    "summarize the paper",
]


def main():
    print("\n" + "="*60)
    print("GENERATION PIPELINE TEST")
    print("="*60)
    print(f"Paper: {PAPER_TITLE}")
    print("Loading models...\n")

    for query in TEST_QUERIES:
        print(f"\nQUESTION: {query}")
        print("-" * 60)

        result = generate_answer(
            query       = query,
            paper_id    = PAPER_ID,
            paper_title = PAPER_TITLE,
        )

        print(f"Mode:     {result.retrieval_mode}")
        print(f"Model:    {result.model_used}")
        print(f"Chunks:   {result.chunks_in_context}")
        print(f"Tokens:   {result.context_tokens}")
        print(f"Sections: {result.section_list}")

        if result.has_equations and result.equation_latex:
            print(f"Equations: {result.equation_latex[:1]}")

        if result.error:
            print(f"ERROR:    {result.error}")

        print(f"\nANSWER:\n{result.answer}")
        print("="*60)


if __name__ == "__main__":
    main()