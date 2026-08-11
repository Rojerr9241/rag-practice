def fixed_size_chunking(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    """Split text into chunks of `chunk_size` characters, with `overlap`
    characters shared between consecutive chunks.

    Example with chunk_size=10, overlap=3:
    text: "ABCDEFGHIJKLMNOPQRST"
    chunk 1: "ABCDEFGHIJ"      (positions 0-10)
    chunk 2: "HIJKLMNOPQ"      (positions 7-17, starts 3 before the end of chunk 1)
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def sentence_chunking(text: str, max_sentences: int = 2) -> list[str]:
    """Split text into chunks by grouping `max_sentences` sentences per chunk,
    respecting sentence boundaries (never cuts a sentence in half).
    """
    sentences = text.split(". ")
    sentences = [s if s.endswith(".") else s + "." for s in sentences]

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        group = sentences[i:i + max_sentences]
        chunks.append(" ".join(group))

    return chunks

if __name__ == "__main__":
    text = (
        "Cats are independent animals. They sleep most of the day. "
        "Dogs are more social and loyal. They need daily exercise. "
        "Both make great pets depending on your lifestyle."
    )

    fixed_chunks = fixed_size_chunking("ABCDEFGHIJKLMNOPQRST", chunk_size=10, overlap=3)
    print(f"Test 1 (fixed_size_chunking): {fixed_chunks}")

    sentence_chunks = sentence_chunking(text, max_sentences=2)
    print(f"Test 2 (sentence_chunking): {sentence_chunks}")