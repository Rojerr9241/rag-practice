"""Exercise 1: Tokenization and text preprocessing. See README.md."""
import re

# frozenset: only used for "word in STOPWORDS" (membership), order doesn't matter
# and it gives O(1) lookup instead of O(n) like a list.
STOPWORDS = frozenset({"the", "a", "an", "is", "are", "in", "on", "and", "of", "to"})

# tuple: order matters here, because _strip_suffix walks this sequence
# and stops at the first match (break). A set doesn't guarantee iteration order.
SUFFIXES = ("ing", "ed", "s", "ly")
MIN_TOKEN_LENGTH = 1


def normalize(text: str) -> str:
    """Lowercase, no punctuation, no redundant whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _strip_suffix(word: str) -> str:
    """Removes the first suffix in SUFFIXES that matches the end of word."""
    for suffix in SUFFIXES:
        if word.endswith(suffix):
            return word[: -len(suffix)]
    return word

def tokenize(text: str) -> list[str]:
    """Splits an already-normalized text into a list of tokens.

    Steps:

    1. Remove stopwords
    2. Apply simple stemming (strip common suffixes)
    """
    words = [word for word in text.split() if word not in STOPWORDS]
    stemmed = [_strip_suffix(word) for word in words]
    return [word for word in stemmed if len(word) > MIN_TOKEN_LENGTH]

def preprocess(text: str) -> list[str]:
    """Full pipeline: normalize + tokenize."""
    text = normalize(text)
    return tokenize(text)


if __name__ == "__main__":
    sample_text = "The cats are running quickly in the gardens"
    tokens = preprocess(sample_text)
    print(f"Original text: {sample_text!r}")
    print(f"Tokens: {tokens}")
    # Edge cases
    print(preprocess(""))            # 1. empty string
    print(preprocess("running")) # 2. already saw this above, but explain it to me
    print(preprocess("Don't stop!!")) # 3. apostrophe + repeated punctuation
    print(preprocess("He was as tall as a tree")) # 4. "as" is a candidate stopword post-stemming