# Notas técnicas — 04 BM25

## Denso (FAISS/embeddings, ej. 03) vs disperso (BM25)

- **Denso**: cada documento es un vector que captura *significado*. "A feline is
  resting on a rug" y "The cat sits on the mat" quedan cerca en el espacio
  vectorial aunque no comparten ni una palabra.
- **Disperso (BM25)**: puramente léxico, cuenta tokens. No tiene ninguna noción
  de significado — si el query y el documento no comparten tokens (tras
  tokenizar), el score es 0, sin importar qué tan relacionados estén en
  significado.

Por eso en producción se combinan ambos (ejercicio 05, hybrid search): BM25
aporta precisión en coincidencias exactas, embeddings aportan recall
semántico — cada uno cubre el punto ciego del otro.

## Términos de la fórmula BM25

`rank_bm25.BM25Okapi` combina, por cada término del query:

- **TF (term frequency)**: cuántas veces aparece el término en el documento.
- **Saturación (`k1`)**: TF no crece sin límite — repetir una palabra 20 veces
  no la hace 20 veces más relevante. `k1` controla qué tan rápido se satura
  ese aporte (rendimientos decrecientes).
- **IDF (inverse document frequency)**: términos que aparecen en casi todos
  los documentos (stopwords) pesan poco o nada; términos raros pesan más.
- **Normalización por longitud (`b`)**: compara la longitud del documento
  contra el promedio del corpus (`avgdl`), para no favorecer a documentos
  largos solo por su tamaño.

## Tokenizer: ¿limpieza + stemming necesarios?

- **Stemming**: sí, importante. BM25 compara tokens exactos — sin stemming,
  "cats" (doc) y "cat" (query) son tokens distintos y no matchean. Es lo que
  le da a BM25 tolerancia a variaciones morfológicas.
- **Stopwords**: más opcional. BM25 ya las castiga naturalmente vía IDF (una
  palabra que aparece en casi todos los documentos tiene IDF cercano a 0, o
  incluso negativo en la fórmula clásica). Quitarlas a mano es más una
  optimización de vocabulario/velocidad que una necesidad estricta.

## Bug encontrado: `get_top_n` vs `get_scores`

`get_top_n` reordena los documentos por relevancia; `get_scores` los deja en
el orden original del corpus. Hacer `zip(get_top_n(...), get_scores(...))`
empareja por *posición*, no por documento — le pega el score equivocado a
cada uno. No se nota si hay muchos empates (ej. varios scores en 0), pero es
un bug real con scores distintos.

**Fix**: emparejar `documents` (orden original) directamente con
`get_scores(...)` (mismo orden), y ordenar uno mismo:

```python
scores = bm25.get_scores(tokenized_query)
ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
return ranked[:k]
```

## `sorted()` es estable → empates conservan el orden original

Cuando todos los scores empatan (ej. query semántico sin overlap léxico,
todos en 0.0), `sorted()` no "decide" nada — conserva el orden de entrada
para los elementos empatados. El resultado no significa relevancia real, es
un artefacto del orden de `documents`.

Importante: esto **solo aplica a los empates**. Si los documentos se
reordenaran, el query literal ("cat mat") seguiría eligiendo siempre
"The cat sits on the mat" primero, porque su score (0.8777) domina
genuinamente sobre los demás — no depende del orden de la lista. El query
semántico sí seguiría el nuevo orden de `documents`, porque ahí todos los
scores siguen empatados en 0.0.

## Alternativa más eficiente a `sorted()`: `heapq.nlargest`

`sorted()` ordena **todo** el corpus (`O(n log n)`) aunque solo quieras los
top `k`. Para corpus grandes con `k << n`, `heapq.nlargest(k, iterable,
key=...)` es `O(n log k)` — usa un heap en vez de ordenar todo. Es
conceptualmente lo mismo que hace FAISS internamente al buscar top-k sin
ordenar el índice completo.

```python
import heapq
ranked = heapq.nlargest(k, zip(documents, scores), key=lambda pair: pair[1])
```

Para este corpus de 4 documentos no hay diferencia práctica — se documenta
aquí como la respuesta "correcta" a escala, y como comentario en el código
(ver `solution.py`).