# rag-practice

Repo de práctica para preparación de prueba técnica de HackerRank (RAG, problem solving, prompt engineering).

Gestionado con `uv`. Las dependencias se van agregando sobre la marcha, según lo que cada ejercicio requiera (`uv add <paquete>`).

## Plan de ejercicios

- [ ] 01 — Tokenización y preprocesamiento de texto
- [ ] 02 — Embeddings y similitud coseno (cálculo manual)
- [ ] 03 — Vector search con FAISS
- [ ] 04 — Retrieval disperso con BM25
- [ ] 05 — Búsqueda híbrida (BM25 + denso)
- [ ] 06 — Chunking de documentos
- [ ] 07 — Pipeline RAG completo (retrieval + generación)

## Estructura

Cada carpeta `NN_ejercicio/` contiene:
- `README.md` — enunciado del ejercicio
- `solution.py` — draft de la solución

`data/` contiene documentos de ejemplo compartidos entre varios ejercicios.