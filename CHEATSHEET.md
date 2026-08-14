# Cheatsheet — RAG (ejercicios 01-07)

Resumen rápido de los conceptos clave de cada ejercicio. Para el detalle y el
razonamiento completo, ver el `NOTES.md` de cada carpeta.

## 01 — Tokenización y preprocesamiento

- **`normalize`**: lowercase → quitar puntuación (`re.sub(r'[^\w\s]', '', text)`)
  → colapsar espacios repetidos.
- **`tokenize`**: quitar stopwords, luego stemming simple (quitar sufijos
  comunes: `ing`, `ed`, `s`, `ly`).
- `frozenset` para stopwords (lookup O(1), el orden no importa) vs. `tuple`
  para sufijos (el orden sí importa — se detiene en el primer match).
- `preprocess = normalize + tokenize` — se reutiliza como `tokenizer` en 04 y 05.

## 02 — Embeddings y similitud coseno

- `SentenceTransformer("all-MiniLM-L6-v2")` → vectores de 384 dimensiones que
  capturan **significado**, no solo palabras.
- Fórmula: `cos(a, b) = (a · b) / (||a|| * ||b||)`, rango `[-1, 1]`.
- Ejemplo clave: "A feline is resting on a rug" encuentra "The cat sits on the
  mat" como más similar, sin compartir ni una palabra — eso es lo que distingue
  embeddings de BM25.
- Versión vectorizada (matrices completas con `cosine_similarity`) da el mismo
  resultado que el loop manual, pero más rápido — preferible en la práctica.

## 03 — FAISS vector search

- `IndexFlatIP` (inner product) + embeddings **normalizados** (norma 1) =
  equivalente matemático a similitud coseno (el denominador de la fórmula
  desaparece).
- `MODEL.encode([texto], ...)` necesita una **lista**, no un string suelto —
  da shape `(1, dim)`; hacerle `[0]` lo rompe a `(dim,)` y `index.search` lo
  rechaza (espera 2D).
- `index.search()` devuelve `(distances, indices)`, ambos arrays 2D de shape
  `(n_queries, k)` — hay que tomar la fila `[0]` y recorrer con
  `zip(distances[0], indices[0])`.
- Cuidado con nombrar una variable de loop `index` — hace shadowing del
  parámetro `index: faiss.Index`.

## 04 — BM25 (retrieval disperso / léxico)

- `BM25Okapi` combina: TF (con saturación vía `k1`, rendimientos decrecientes),
  IDF (penaliza palabras muy comunes), normalización por longitud (`b`).
- Puramente léxico: si el query y el documento no comparten tokens, el score
  es `0` sin importar cuán relacionados estén en significado.
- **Bug clásico**: `zip(get_top_n(...), get_scores(...))` empareja por
  *posición*, no por documento (`get_top_n` reordena, `get_scores` no) —
  emparejar mal el score con el documento equivocado.
- `heapq.nlargest(k, ..., key=...)` es más eficiente que `sorted()` para
  top-k en corpus grandes: `O(n log k)` vs. `O(n log n)`.

## 05 — Hybrid search (BM25 + denso)

- Combina denso (semántico) y disperso (léxico) — cada uno cubre el punto
  ciego del otro (BM25 falla sin overlap léxico; denso falla con nombres
  propios/jerga muy específica).
- Hay que **normalizar** ambos scores a `[0, 1]` (min-max) antes de combinar,
  porque tienen escalas distintas.
- `combined_score = alpha * dense_norm + (1 - alpha) * bm25_norm` — combinación
  convexa (pesos suman 1), siempre queda en `[0, 1]`. `alpha=1.0` → denso puro,
  `alpha=0.0` → BM25 puro.
- FAISS devuelve resultados en orden de relevancia, no en el orden original de
  `documents` — hay que "esparcir" (scatter) cada score a su posición original
  antes de combinar con BM25.
- **Import cruzado entre ejercicios**: todos los archivos se llaman
  `solution.py`, así que un `sys.path.insert` + `from solution import x` dos
  veces reutiliza el primer módulo (Python cachea por nombre). Solución:
  `importlib.util.spec_from_file_location(module_name, path)` con un nombre
  único por módulo importado.

## 06 — Chunking de documentos

- **Fixed-size**: simple y rápido, pero puede cortar una oración a la mitad.
  `overlap` no evita el corte — mitiga la pérdida repitiendo el contenido del
  borde en ambos chunks.
- **Sentence-based**: respeta límites de oración, más semántico, pero un split
  ingenuo (`". "`) falla con abreviaturas ("Mr.", "etc.").
- `chunk_size` / `max_sentences` controlan el mismo trade-off: chunks grandes
  → más contexto pero retrieval menos preciso (y más costo si van a un LLM);
  chunks chicos → más precisión pero más vectores (más almacenamiento/costo).
- Los modelos de embeddings truncan silenciosamente el texto que excede su
  límite de tokens (ej. 256-512) — un `chunk_size` mal elegido puede perder
  contenido sin error visible.

## 07 — Pipeline RAG completo

- Retrieval (`hybrid_search`) + construcción de prompt con el contexto
  recuperado.
- El prompt instruye al modelo a responder **solo** con el contexto dado, y a
  decir que no sabe si la respuesta no está ahí — mitiga alucinaciones.
- No hace falta llamar a un LLM real para demostrar que entendés el pipeline
  — el prompt final ya armado lo demuestra.

## Errores/gotchas que se repiten en varios ejercicios

- **Shape mismatches**: `encode()` necesita listas (da 2D); `index.search`
  espera y devuelve 2D — el error típico es "desempaquetar" a 1D de más.
- **`zip` mal alineado** cuando dos estructuras no comparten el mismo orden
  (bug de `get_top_n` en 04, scatter de FAISS en 05).
- **Nombres de archivo duplicados** (`solution.py` en cada carpeta) rompen
  imports simples entre ejercicios — usar `importlib` con nombre único.
- **`sorted()` es estable**: ante empates, conserva el orden original de la
  lista — eso no significa relevancia real, es un artefacto del orden de
  entrada.

## Guía rápida: ¿qué método de retrieval usar?

| Situación | Método |
|---|---|
| Coincidencias exactas importan (nombres propios, códigos, jerga específica) | BM25 |
| El significado importa más que las palabras exactas | Embeddings / FAISS |
| Ambos casos coexisten en el dominio | Hybrid search (ajustar `alpha`) |