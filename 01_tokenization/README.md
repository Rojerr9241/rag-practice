# 01 — Tokenización y preprocesamiento de texto

## Objetivo

Implementar un pipeline básico de preprocesamiento de texto: normalización
(minúsculas, limpieza de puntuación/espacios) y tokenización (dividir en
palabras). Es el paso previo a cualquier sistema de retrieval (BM25,
embeddings, etc.), así que conviene dejarlo sólido antes de avanzar.

## Funciones esperadas

- `normalize(text: str) -> str`
  Minúsculas, sin puntuación, sin espacios redundantes.
- `tokenize(text: str) -> list[str]`
  Divide un texto ya normalizado en tokens (palabras).
- `preprocess(text: str) -> list[str]`
  Pipeline completo: `normalize` + `tokenize`.

## Casos a considerar

- Puntuación pegada a palabras (`"test."`, `"hello,"`).
- Espacios múltiples o al inicio/final.
- Mayúsculas mezcladas.
- (Opcional, para ir más allá) stopwords, stemming/lemmatización.

## Cómo correrlo

```bash
uv run 01_tokenization/solution.py
```
