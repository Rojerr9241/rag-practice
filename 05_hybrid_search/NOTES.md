# Notas técnicas — 05 Hybrid search

## Por qué "denso" vs "disperso" (sparse)

No es terminología arbitraria — describe la forma literal del vector:

- **Denso (embeddings)**: cada una de las ~384 dimensiones tiene un valor
  real, casi ninguna en 0. El significado está repartido entre todas las
  dimensiones.
- **Disperso (BM25 / TF-IDF)**: la dimensionalidad es del tamaño del
  vocabulario (potencialmente miles de palabras), pero para un documento
  dado casi todas esas posiciones son 0 — solo las palabras que aparecen en
  ese documento tienen valor distinto de cero.

## Qué es `alpha` realmente

Es el mediador entre **semántico** (denso, `alpha=1.0`) y **léxico** (BM25,
`alpha=0.0`) — no "sintáctico" (eso sería estructura gramatical, que ninguno
de los dos métodos analiza).

## Por qué combinar ambos: robustez

Cada método tiene un punto ciego distinto:
- BM25 falla si el query no comparte tokens literales con el documento,
  aunque sea relevante en significado (ver ejercicio 04, query semántico →
  todos los scores en 0.0).
- El denso puede fallar donde importa una coincidencia exacta (nombres
  propios, códigos, jerga muy específica) que el embedding no distingue
  bien.

Al calificar con ambas señales y combinarlas, un documento fuerte en
cualquiera de las dos dimensiones sigue teniendo oportunidad de aparecer
arriba, en vez de depender de una sola métrica que podría fallar en
silencio.

## Rango de `combined_score`: siempre `[0, 1]`

`combined_score = alpha * dense_norm + (1 - alpha) * bm25_norm` es una
combinación **convexa** (los pesos `alpha` y `(1 - alpha)` suman 1). Un
promedio ponderado de valores que ya están en `[0, 1]`, con pesos que suman
1, nunca puede salirse de ese rango — el máximo (1.0) solo se alcanza si
ambas fuentes (o la que tiene peso 1) valen 1.0 para ese documento.

## Limitación conocida: empates se resuelven por orden de lista, no por relevancia

`sorted(zip(documents, combined), key=..., reverse=True)` es estable — ante
un empate en `combined_score`, conserva el orden original de `documents`.
Ejemplo: si 3 documentos empatan en `combined_score = 1.0` y `k=2`, ganan
los primeros dos que aparecen en la lista original de `documents`, no por
ninguna razón de relevancia — el tercero queda fuera por pura casualidad
posicional.

No hay ningún criterio de desempate implementado (ej. longitud del
documento, score bruto de una sola fuente, recencia). Si esto importara en
un caso real, habría que agregar una clave secundaria explícita al
`sorted()`. Se deja documentado aquí como limitación conocida, no resuelta
a propósito para mantener el ejercicio simple.