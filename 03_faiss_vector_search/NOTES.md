# Notas técnicas — 03 FAISS vector search

## Por qué `IndexFlatIP` + embeddings normalizados = similitud coseno

`IndexFlatIP` calcula *inner product* (producto punto) entre vectores, no
similitud coseno directamente. Pero si los vectores están normalizados a
norma 1 (`MODEL.encode(documents, normalize_embeddings=True)`), el producto
punto es matemáticamente equivalente al coseno:

```
cos(a, b) = (a · b) / (||a|| * ||b||)
```

Si `||a|| = ||b|| = 1`, el denominador desaparece y `cos(a, b) = a · b`. Por
eso normalizar antes de indexar es lo que permite usar `IndexFlatIP` (más
simple/rápido que `IndexFlatL2` + normalización manual) y seguir obteniendo
resultados equivalentes a similitud coseno.

## `index.search` espera un array 2D, no 1D

`MODEL.encode([query], normalize_embeddings=True)` ya regresa shape
`(1, dim)` porque se le pasa una lista de un elemento. El error común es
hacerle `[0]` para "sacar el vector" — eso lo deja en `(dim,)` (1D), y
`index.search` lo rechaza porque espera `(n_queries, dim)`, igual que
`index.add()` espera una matriz de documentos, no un vector suelto.

## `index.search` regresa una tupla `(distances, indices)`, no pares

```python
distances, indices = index.search(query_vector, k)
```

Ambos son arrays 2D de shape `(n_queries, k)` — una fila por query. Como acá
solo se manda una query, hay que tomar la fila 0 de cada uno
(`distances[0]`, `indices[0]`) antes de iterar. El bug típico es intentar
`for distance, index in result:` directamente sobre la tupla — eso
desempaqueta los *dos elementos de la tupla* (el array completo de
distancias y el array completo de índices) en dos variables, no "un
resultado a la vez".

La forma correcta es `zip(distances[0], indices[0])` para recorrer ambos
arrays en paralelo, documento por documento.

## Cuidado con nombres que hacen shadowing

Usar `index` como nombre de variable de loop (`for dist, index in ...`)
choca con el parámetro `index: faiss.Index` de la función. En Python 3 el
scope del list comprehension aísla la variable y no rompe nada *fuera* del
comprehension, pero es confuso de leer — mejor usar otro nombre (`i`,
`doc_idx`).

## Denso vs disperso

Ver la comparación completa en `04_bm25/NOTES.md`. En corto: embeddings
capturan significado — el query "A feline is resting on a rug" encuentra
"The cat sits on the mat" como el resultado más relevante (score 0.5842)
aunque no comparten ni una palabra. BM25 (ejercicio 04) no puede hacer eso.