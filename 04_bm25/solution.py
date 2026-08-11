from rank_bm25 import BM25Okapi
import sys
from pathlib import Path

# sube un nivel desde 04_bm25/ a la raíz del repo, y entra a 01_tokenization/
sys.path.insert(0, str(Path(__file__).parent.parent / "01_tokenization"))

from solution import preprocess

def build_bm25_index(documents: list[str], tokenizer) -> BM25Okapi:
    """Tokenize the documents with `tokenizer` and build a BM25 index."""
    tokenized_docs = [tokenizer(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)

def search_bm25(bm25: BM25Okapi, documents: list[str], query: str, tokenizer, k: int = 2) -> list[tuple[str, float]]:
    """Tokenize the query and return the k most relevant documents according
    to BM25, ordered by descending score."""
    tokenized_query = tokenizer(query)
    scores = bm25.get_scores(tokenized_query)
    # sorted() ranks the whole corpus (O(n log n)) even though we only need
    # the top k. For large corpora, heapq.nlargest(k, ..., key=...) is
    # O(n log k) instead — see NOTES.md.
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    documents = [
        "The cat sits on the mat",
        "Dogs are loyal animals",
        "I love programming in Python",
        "Cats and dogs are common pets",
    ]

    query_semantic = "A feline is resting on a rug"  # same semantic query as exercise 3
    query_literal = "cat mat"                        # shares literal words with a document

    bm25 = build_bm25_index(documents, preprocess)

    for label, query in [("semantic", query_semantic), ("literal", query_literal)]:
        print(f"\nQuery ({label}): {query!r}")
        for doc, score in search_bm25(bm25, documents, query, preprocess):
            print(f"  {score:.4f}  {doc}")

    # with embeddings (exercise 3), the top result for the semantic query was
    # "The cat sits on the mat" even without shared words.
    # with BM25: what wins for each query here, and why?