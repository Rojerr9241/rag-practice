"""Ejercicio 1: Tokenización y preprocesamiento de texto. Ver README.md."""


def normalize(text: str) -> str:
    """Minúsculas, sin puntuación, sin espacios redundantes."""
    pass


def tokenize(text: str) -> list[str]:
    """Divide un texto ya normalizado en una lista de tokens."""
    pass


def preprocess(text: str) -> list[str]:
    """Pipeline completo: normalize + tokenize."""
    pass


if __name__ == "__main__":
    sample_text = "  Hello, World! This is a Test Sentence... for TOKENIZATION.  "
    tokens = preprocess(sample_text)
    print(f"Texto original: {sample_text!r}")
    print(f"Tokens: {tokens}")
