# rag-practice

Repo de práctica para preparación de prueba técnica de HackerRank (RAG, problem solving, prompt engineering).

Gestionado con `uv`. Las dependencias se van agregando sobre la marcha, según lo que cada ejercicio requiera (`uv add <paquete>`).

## Plan de ejercicios

- [x] 01 — Tokenización y preprocesamiento de texto
- [x] 02 — Embeddings y similitud coseno (cálculo manual)
- [x] 03 — Vector search con FAISS
- [x] 04 — Retrieval disperso con BM25
- [x] 05 — Búsqueda híbrida (BM25 + denso)
- [x] 06 — Chunking de documentos
- [x] 07 — Pipeline RAG completo (retrieval + generación)

## Estructura

Cada carpeta `NN_ejercicio/` contiene:
- `README.md` — enunciado del ejercicio (cuando aplica)
- `solution.py` — draft de la solución
- `NOTES.md` — apuntes técnicos y aclaraciones que surgen al resolverlo (comparaciones, alternativas, bugs encontrados y por qué)

`data/` contiene documentos de ejemplo compartidos entre varios ejercicios.

`CHEATSHEET.md` (raíz) resume los conceptos clave de los 7 ejercicios para
repaso rápido antes de la prueba técnica.