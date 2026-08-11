import importlib.util
from pathlib import Path

def _import_solution(exercise_dir: str, module_name: str):
    path = Path(__file__).parent.parent / exercise_dir / "solution.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

tokenization = _import_solution("01_tokenization", "tokenization_solution")
faiss_solution = _import_solution("03_faiss_vector_search", "faiss_solution")
bm25_solution = _import_solution("04_bm25", "bm25_solution")

preprocess = tokenization.preprocess
MODEL = faiss_solution.MODEL

def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to the [0, 1] range.
    Handles the case where every score is equal (division by zero).
    """
    min_score, max_score = min(scores), max(scores)

    # edge case
    if max_score == min_score:
        return [0.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_search(
    query: str,
    documents: list[str],
    bm25_index,           # BM25Okapi from exercise 4
    faiss_index,           # FAISS index from exercise 3
    tokenizer,             # preprocess
    alpha: float = 0.5,   # weight: alpha for dense, (1 - alpha) for BM25
    k: int = 2,
) -> list[tuple[str, float]]:
    """Combine BM25 and embeddings:
    1. Get BM25 scores for every document
    2. Get dense (cosine similarity) scores for every document
    3. Normalize BOTH to [0, 1] separately
    4. combined_score = alpha * dense_norm + (1 - alpha) * bm25_norm
    5. Return the top-k documents by combined_score
    """
    # step 1
    tokenized_query = tokenizer(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # step 2
    query_vector = MODEL.encode([query], normalize_embeddings=True)
    distances, indexes = faiss_index.search(query_vector, k=len(documents))

    # FAISS returns results ranked by relevance, not in `documents` order,
    # so we scatter each distance back to its original document position.
    dense_scores = [0.0] * len(documents)
    for idx, dist in zip(indexes[0], distances[0]):
        dense_scores[idx] = dist

    # step 3
    dense_norm = normalize_scores(dense_scores)
    bm25_norm = normalize_scores(bm25_scores)

    # step 4
    combined = [alpha * d + (1 - alpha) * b for d, b in zip(dense_norm, bm25_norm)]

    # step 5
    ranked = sorted(zip(documents, combined), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]

if __name__ == "__main__":
    documents = [
        "The cat sits on the mat",
        "Dogs are loyal animals",
        "I love programming in Python",
        "Cats and dogs are common pets",
    ]

    # same two queries as exercise 4
    query_semantic = "A feline is resting on a rug"
    query_literal = "cat mat"

    bm25_index = bm25_solution.build_bm25_index(documents, preprocess)
    faiss_index, embeddings = faiss_solution.build_index(documents)

    # alpha=1.0 -> dense only, alpha=0.0 -> BM25 only, alpha=0.5 -> balanced
    for label, query in [("semantic", query_semantic), ("literal", query_literal)]:
        print(f"\nQuery ({label}): {query!r}")
        for alpha in (1.0, 0.5, 0.0):
            result = hybrid_search(query, documents, bm25_index, faiss_index, preprocess, alpha=alpha)
            print(f"  alpha={alpha}: {result}")