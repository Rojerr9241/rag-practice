import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_index(documents: list[str]) -> tuple[faiss.Index, np.ndarray]:
    """Generate embeddings for the documents and build a FAISS index.
    Returns the index and the embeddings matrix (in case you need it later).
    """
    embeddings = MODEL.encode(documents, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return (index, embeddings)

def search(index: faiss.Index, documents: list[str], query: str, k: int = 2) -> list[tuple[str, float]]:
    """Search for the k most similar documents to the query in the index.
    Returns a list of (document, score) ordered by relevance.
    """
    query_vector = MODEL.encode([query], normalize_embeddings=True)
    distances, indexes = index.search(query_vector, k)
    return [(documents[i], dist) for dist, i in zip(distances[0], indexes[0])]


if __name__ == "__main__":
    documents = [
        "The cat sits on the mat",
        "Dogs are loyal animals",
        "I love programming in Python",
        "Cats and dogs are common pets",
    ]
    query = "A feline is resting on a rug"

    index, embeddings = build_index(documents)
    print(f"embeddings dtype: {embeddings.dtype}")   # should be float32
    print(f"vectors in index: {index.ntotal}")        # should equal len(documents) = 4

    print(f"\nQuery: {query!r}")
    for doc, score in search(index, documents, query):
        print(f"  {score:.4f}  {doc}")