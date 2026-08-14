import importlib.util
from pathlib import Path

def _import_solution(exercise_dir: str, module_name: str):
    path = Path(__file__).parent.parent / exercise_dir / "solution.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

hybrid_solution = _import_solution("05_hybrid_search", "hybrid_search_solution")

hybrid_search = hybrid_solution.hybrid_search
bm25_solution = hybrid_solution.bm25_solution
faiss_solution = hybrid_solution.faiss_solution
preprocess = hybrid_solution.preprocess

def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    return f"""Responde la pregunta usando SOLO el siguiente contexto.
Si la respuesta no está en el contexto, di que no lo sabes.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""

def rag_pipeline(query: str, documents: list[str], k: int = 2) -> str:
    """Full pipeline:
    1. Retrieve the k most relevant chunks/documents (reuses FAISS
       or hybrid_search from previous exercises)
    2. Build the prompt with build_prompt()
    3. Return the final prompt (no need to call a real LLM,
       just print/return the assembled prompt — that already
       demonstrates you understand the pipeline)
    """
    bm25_index = bm25_solution.build_bm25_index(documents, preprocess)
    faiss_index, embeddings = faiss_solution.build_index(documents)
    results = hybrid_search(query, documents, bm25_index, faiss_index, preprocess, alpha=0.5, k=k)
    retrieved_chunks = [text for text, _ in results]
    return build_prompt(query, retrieved_chunks)

if __name__ == "__main__":
    documents = [
        "The cat sits on the mat",
        "Dogs are loyal animals",
        "I love programming in Python",
        "Cats and dogs are common pets",
    ]
    query = "What is sitting on the mat?"

    print(rag_pipeline(query, documents, k=2))
    # should print the full prompt, with the 2 most relevant chunks
    # inserted into the context
    