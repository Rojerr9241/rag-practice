from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pretrained Sentence Transformer model
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) -> np.ndarray:
    """Converts a list of texts into a matrix of embeddings.
    Uses the 'all-MiniLM-L6-v2' model (fast and lightweight).
    """
    return MODEL.encode(texts)

def cosine_similarity_metric(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors
    — The formula is:
        cos(a, b) = (a . b) / (||a|| * ||b||)
    """
    # Numpy: cs = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]

def most_similar(query: str, documents: list[str]) -> tuple[str, float]:
    """Given a query and a list of documents, returns the most similar
    document and its cosine similarity score."""
    doc_vectors = get_embeddings(documents)
    query_vector = get_embeddings([query])[0]
    best_similarity = -1  # cosine similarity ranges [-1, 1], not [0, inf) like a distance
    for i in range(doc_vectors.shape[0]):
        similarity = cosine_similarity_metric(query_vector, doc_vectors[i])
        if similarity > best_similarity:
            best_match = (documents[i], similarity)
            best_similarity = similarity
    return best_match

def most_similar_vectorized(query: str, documents: list[str]) -> tuple[str, float]:
    query_vector = get_embeddings([query])        # shape (1, 384)
    doc_vectors = get_embeddings(documents)       # shape (n_docs, 384)
    
    similarities = cosine_similarity(query_vector, doc_vectors)  # shape (1, n_docs)
    # similarities[0] is an array with a score per document

    best_index = np.argmax(similarities[0])
    return documents[best_index], similarities[0][best_index]

if __name__ == "__main__":

    documents = [
    "The cat sits on the mat",
    "Dogs are loyal animals",
    "I love programming in Python",
    "Cats and dogs are common pets",
    ]

    query = "A feline is resting on a rug"
    
    loop_result = most_similar(query, documents)
    vectorized_result = most_similar_vectorized(query, documents)

    print(f"Loop:       {loop_result}")
    print(f"Vectorized: {vectorized_result}")

    same_document = loop_result[0] == vectorized_result[0]
    same_score = np.isclose(loop_result[1], vectorized_result[1])
    assert same_document and same_score, "Loop and vectorized versions disagree!"
    print("Both versions agree.")

# expected: the most similar doc should be "The cat sits on the mat"
# even though they don't share a single literal word — that's exactly
# what shows embeddings capture MEANING, not just word overlap
# (unlike your tokenizer from exercise 1)