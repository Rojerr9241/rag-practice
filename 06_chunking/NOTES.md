# Notas técnicas — 06 Chunking

## Por qué chunking

Los documentos suelen ser demasiado largos para pasarlos enteros a un modelo
de embeddings o al contexto de un LLM, y en retrieval interesa recuperar la
porción específica relevante, no el documento completo. Chunking es el paso
que divide un documento en piezas manejables antes de indexarlo.

## Fixed-size vs sentence-based: trade-off

- **Fixed-size**: simple y rápido (corta por cantidad de caracteres), pero
  puede partir una oración o palabra a la mitad, perdiendo coherencia
  semántica en el borde del corte.
- **Sentence-based**: respeta límites de oración (nunca corta una a la
  mitad), captura mejor el significado, pero requiere detectar dónde termina
  cada oración — más complejo que un corte por longitud fija.

`overlap` en `fixed_size_chunking` **no evita** que una oración se corte —
sigue cortándose igual. Lo que hace es que el contenido del borde aparezca
repetido en ambos chunks (el que termina a mitad de oración y el que empieza
ahí), así ningún chunk se queda sin el contexto completo de esa zona límite.
Mitiga la pérdida de información, no previene el corte.

## Limitación conocida: `sentence_chunking` y abreviaturas

El split usado (`text.split(". ")`) es ingenuo: asume que todo `". "` marca
el final de una oración. Falla con abreviaturas como "Mr.", "Ms.", "etc." —
por ejemplo, `"Mr. Smith is here."` se partiría incorrectamente en
`"Mr"` / `"Smith is here."`.

No se implementó un manejo especial para esto (ej. regex con excepciones
para abreviaturas conocidas) porque le agrega complejidad y nunca cubre
todos los casos — es un problema real de NLP que herramientas como
`nltk.sent_tokenize` o spaCy resuelven con modelos entrenados para eso, no
con reglas manuales. Se deja documentado aquí como limitación conocida,
aceptable para el alcance de este ejercicio.

## `chunk_size` / `max_sentences`: precisión vs. contexto (y costo)

Ambos parámetros controlan la granularidad del chunk (uno en caracteres,
el otro en oraciones) y el mismo trade-off aplica a los dos:

- **Chunks grandes**: más contexto por chunk (útil para documentos donde el
  significado depende de varias oraciones juntas, ej. cláusulas legales),
  pero el embedding "promedia" varias ideas y el retrieval pierde precisión
  ante preguntas puntuales. Si además se pasan a un LLM como contexto, más
  tokens = más costo por consulta.
- **Chunks chicos**: retrieval más preciso (útil para Q&A muy factual, ej.
  "¿cuál es el límite de peso máximo?"), pero más chunks = más vectores =
  más memoria en el índice y más embeddings que generar. También hay más
  riesgo de que una respuesta quede partida entre dos chunks distintos.

Límite técnico real a tener en cuenta: los modelos de embeddings tienen un
límite de tokens de entrada (ej. 256-512 tokens en muchos
`sentence-transformers`). Un `chunk_size` que genere chunks más largos que
ese límite se trunca silenciosamente — el chunk queda representado solo por
su primera parte sin que se note el error.

Caso real de chunking más sofisticado: sistemas de Q&A sobre documentación
técnica o legal suelen cortar por sección/encabezado en vez de por longitud
fija — una extensión de la idea de `sentence_chunking` (respetar límites
semánticos) pero a nivel de estructura del documento, no de oración.