# Resumen para la memoria

Resumen del servicio de *entity linking* a Wikidata siguiendo los apartados de la memoria, seguido de las opciones consideradas en cada fase y el motivo de cada decisión.

## 1. Alcance y casos de uso

**Problema.** Dado un texto en lenguaje natural, la herramienta detecta las entidades nombradas que aparecen en él (personas, organizaciones, lugares, obras…) y enlaza cada una con su elemento de Wikidata (QID). Si la entidad no está en Wikidata, devuelve `NIL`. La dificultad principal es la ambigüedad: un mismo nombre puede referirse a entidades distintas ("Michael Jordan", el jugador o el profesor de Berkeley; "Barcelona", la ciudad o el club). La herramienta elige la más probable según el contexto y, cuando no está segura, lo indica (`confidence`, `needs_review`) junto con los candidatos alternativos.

**Para qué se usaría.**
- **Poblar o mantener un grafo de conocimiento:** conectar menciones de documentos con identificadores estables y con sus propiedades en Wikidata.
- **Asistir a documentalistas (*human in the loop*):** lo seguro se acepta automáticamente y lo dudoso llega a una persona con las alternativas ya ordenadas. Sus correcciones sirven para reajustar el sistema a cada dominio.

**Tipos de entidad.** 19 tipos finos (*person*, *sports team*, *company*, *political party*, *university*, *country*, *city*, *region*, *award*, *sports competition*, *work of art*…), agrupados en las cuatro categorías estándar `PER`, `ORG`, `LOC` y `MISC`. El tipo fino no solo etiqueta: también ayuda a desambiguar (un "sports team" no puede ser una ciudad). Con tipos finos, el F1 del reconocimiento en AIDA es 0,849, frente a 0,667 pidiendo solo las cuatro etiquetas genéricas.

**Idiomas.** El inglés es el idioma evaluado, porque es el de los conjuntos de referencia anotados con QIDs (AIDA y los *benchmarks* clásicos). El español está disponible con `language=es`: el modelo de NER es multilingüe y Wikidata y Wikipedia se consultan en el idioma del texto (no se ha medido cuantitativamente).

**Nivel de ambigüedad esperado.**

*Qué tipos de ambigüedad debe resolver:*
- **homónimos de clase distinta:** un mismo nombre para entidades de tipo diferente ("Paris", la ciudad o la persona; "Columbia", la universidad o el sello discográfico);
- **homónimos de la misma clase:** dos personas o dos organizaciones con el mismo nombre, donde el tipo no ayuda y decide el resto del documento;
- **nombres incompletos o apodos:** apellidos, nombres de pila o formas cortas ("Dylan", "Clapton");
- **entidades que no están en Wikidata:** deben quedar como `NIL` o, al menos, llegar a revisión.


*Cuánta hay en la práctica.* Se mide en las menciones cuyo candidato correcto está entre los 20 que devuelve Wikidata: si no es el primero, hace falta desambiguar.

| Tipo de texto | El correcto es el primero | Necesita desambiguación | Acierto del sistema (desambiguación) |
|---|---|---|---|
| Noticias generales (MSNBC, ACE2004, AQUAINT) | 80–98 % | 2–20 % | 92–96 % |
| Noticias deportivas (AIDA) | 69 % | 31 % | 76 % |
| Homónimos difíciles a propósito (KORE50) | 42 % | 58 % | 77 % |

En textos generales la ambigüedad es moderada; aumenta con los homónimos y en las crónicas deportivas, donde abundan los nombres de lugar usados para equipos. A esa medida se suman dos fuentes de dificultad que no recoge: las menciones cuya entidad no está en Wikidata (entre el 14 % y el 29 % según el conjunto) y aquellas cuyo candidato correcto no aparece en la búsqueda (19 % en AIDA), que marcan el techo actual del sistema. El acierto de la tabla es el de la desambiguación aislada; el rendimiento *end-to-end*, que incluye también los errores del NER y de la recuperación, se detalla en la sección 4.

**Volumen de texto.** Pensado para documentos cortos y medios: noticias, biografías, fichas, párrafos enciclopédicos.

| Medida | Valor |
|---|---|
| Documento típico (AIDA test) | mediana 827 caracteres (~140 palabras), el 95 % por debajo de 2.862; ~20–30 menciones |
| Documentos más largos procesados | hasta 5.253 caracteres (~850 palabras, MSNBC); no se ha medido por encima |
| Sin LLM, con caché | ~1,5 s por documento (mediana 1,4 s en documentos de mediana 930 caracteres; ~2.400 documentos/hora por proceso) |
| Sin LLM, primera vez (sin caché) | ~10 s por documento, por las consultas a Wikidata |
| Con LLM local | se consulta en ~28 % de las menciones; de 5 a 30 s por documento según la GPU |

El NER crece aproximadamente con la longitud (~1,5 ms por carácter en CPU); el resto depende más del número de menciones distintas que de la longitud. Los textos largos se procesan por fragmentos de frases (~1.500 caracteres). 


## 2. Arquitectura de la solución

Servicio FastAPI (`POST /api/v1/link/text`, con Swagger UI): el cuerpo es el texto y las opciones van como parámetros de la URL (`language`, `use_llm`, `review_threshold`, `max_candidates`). Ejecuta un pipeline de tres fases y devuelve una salida pensada para la revisión:

1. **NER con GLiNER** (`urchade/gliner_multi-v2.1`, zero-shot), con tipos finos. Los textos largos se trocean por frases.
2. **Recuperación de candidatos** (hasta 20 por mención):
   - búsqueda por etiqueta y alias en Wikidata (`wbsearchentities`, por prefijo);
   - en paralelo, **redirecciones de Wikipedia**: si la mención es un título o una redirección ("Asian Cup" → *AFC Asian Cup*), su elemento pasa a ser el primero;
   - si no hay resultados, **búsqueda aproximada**: texto completo de Wikidata filtrado con RapidFuzz ("S.Philpott" → *Sam Philpott*);
   - sin páginas de desambiguación, con caché en SQLite, reintentos que respetan `Retry-After` y un máximo de 4 peticiones simultáneas.
3. **Desambiguación** (sección 3), por documento:
   - **señales** de cada candidato: las baratas (`prior`, `popularity`, `context`) para todas las menciones y las de Wikidata (`type`, `relation`, por SPARQL) solo para las que siguen inciertas (carga diferida);
   - **logit condicional:** combina las señales con pesos aprendidos y elige el candidato más probable; su probabilidad es la confianza devuelta, que decide `needs_review`;
   - **LLM local** (Ollama, `qwen2.5:3b`), opcional, solo para las menciones que siguen inciertas: elige entre los candidatos y, si discrepa de las señales, la mención va a revisión.
4. **Salida:** por cada entidad, sus *offsets* en el texto, la etiqueta (`PER`/`ORG`/`LOC`/`MISC` y el tipo fino), el QID o `NIL` con su etiqueta y descripción de Wikidata, `confidence`, `reasoning`, `needs_review`, `llm_confirmed` y los candidatos alternativos ordenados por probabilidad (`max_candidates`).

Estructura de la respuesta, con cada valor sustituido por su significado:

```json
{
  "original_text": "texto recibido, sin cambios",
  "entities": [
    {
      "text": "mención tal como aparece en el texto",
      "start_char": "posición inicial (carácter) de la mención en el texto",
      "end_char": "posición final (exclusiva) de la mención en el texto",
      "label": "categoría estándar: PER, ORG, LOC o MISC",
      "fine_label": "tipo fino pedido a GLiNER (p. ej. 'sports team'); alimenta la señal type",
      "qid": "QID de Wikidata elegido, o NIL",
      "wikidata_label": "etiqueta en Wikidata del QID elegido, en el idioma del texto",
      "wikidata_description": "descripción en Wikidata del QID elegido",
      "confidence": "probabilidad softmax del candidato elegido según las señales: puntuación de ranking, no calibrada (sección 3.3)",
      "is_nil": "true si la entidad no se enlaza con ningún QID",
      "reasoning": "explicación: p del elegido, número de candidatos y las dos señales que más lo separan del segundo; o la decisión del LLM",
      "llm_confirmed": "true: el LLM eligió lo mismo que las señales; false: eligió otro o ninguno; null: no se le consultó",
      "needs_review": "true si es NIL, si el LLM discrepa o si confidence < review_threshold sin confirmar por el LLM",
      "candidates": [
        "los max_candidates más probables según las señales, de mayor a menor p (el primero es el de las señales, aunque el LLM haya elegido otro)",
        {
          "qid": "QID del candidato",
          "label": "etiqueta del candidato en Wikidata",
          "description": "descripción del candidato en Wikidata (puede ser null)",
          "concept_uri": "URI del elemento en Wikidata"
        }
      ]
    }
  ],
  "processing_time_ms": "tiempo de proceso de la petición, en milisegundos"
}
```

Los recursos (modelo, cliente HTTP, caché) se crean una sola vez en el `lifespan` de FastAPI. Los parámetros ajustables están centralizados en `config.py` y se pueden sobrescribir con variables `ENTITY_LINKER_*`. Hay una suite de pruebas con `pytest`, con la red y el LLM simulados, que se ejecuta en ~1 s.

## 3. Resolución de la ambigüedad

### 3.1 Señales

Cada candidato recibe cinco señales entre 0 y 1:

| Señal | Qué mide | Peso |
|---|---|---|
| `prior` | 1 / posición en la búsqueda | 1,35 |
| `popularity` | artículos de Wikipedia (*sitelinks*) del candidato, en escala logarítmica, frente al más popular de la mención | 4,98 |
| `context` | palabras de la descripción del candidato que aparecen alrededor de la mención (±300 caracteres), **sin contar la propia mención** | 3,72 |
| `type` | si el candidato pertenece (P31/P279*) a la clase del tipo fino del NER | 0,90 |
| `relation` | entidades ya resueltas del documento **unidas al candidato por una declaración** de Wikidata | 4,31 |

El contexto excluye la mención, como en la literatura: la similitud de nombre ya está en `prior`, y contarla de nuevo favorecería descripciones que repiten el nombre ("Nobel Prize in Physics controversies").

### 3.2 Logit condicional

Las señales se combinan linealmente y un softmax sobre los candidatos de la mención (hasta 20) da la probabilidad de cada uno; se elige el de mayor $p_i$:

$$\text{score}_i = \sum_s w_s \, x_{s,i} \qquad\qquad p_i = \frac{e^{\text{score}_i}}{\sum_j e^{\text{score}_j}}$$

**Pesos** por máxima verosimilitud del candidato correcto $g$ de cada mención, con L2 y pesos no negativos (`scripts/tune_weights.py`):

$$\min_{w \,\geq\, 0} \; -\sum_m \log p_{g(m)} \;+\; \lambda \lVert w \rVert^2, \qquad \lambda = 0{,}1$$

- **L2** mantiene los pesos finitos cuando una señal separa casi perfectamente y estabiliza los de señales correlacionadas (`prior` y `popularity`).
- **$w \geq 0$**: cada señal es evidencia a favor, y así `reasoning` puede explicar cada decisión por las aportaciones $w_s x_{s,i}$.
- **Datos:** protocolo de AIDA: se aprenden en **validation** (2.867 menciones) y se miden en **test** (557 menciones, sin documentos en común): 0,763 frente a 0,689 del primer candidato.
- `popularity` empuja hacia la entidad conocida y `relation` la contrarresta cuando el documento menciona entidades relacionadas con otro candidato. Los pesos dependen del dominio (noticias en inglés de 1996).

### 3.3 Confianza: una puntuación de *ranking*, no una probabilidad calibrada

`confidence` es la $p_i$ del candidato elegido: dice **cuál** de los candidatos recuperados es el mejor y con qué margen, pero **no** la probabilidad de que el enlace sea correcto.

- **Es relativa a los candidatos:** supone que el correcto está entre ellos; no contempla NIL ni fallos de recuperación.
- **Con un único candidato vale 1** por construcción ("no hay alternativa", no "es correcto").

Medido *end-to-end* en AIDA test (3.763 menciones enlazadas): ECE = 0,141; con el umbral 0,7 llega a revisión el 52 % de los errores; con un único candidato (16 % de las menciones) la confianza es 1,00 frente a un acierto real de 0,66. `needs_review` filtra bien la duda **entre** candidatos, pero deja pasar errores en que la lista entera es mala. Mejora propuesta: un candidato NIL virtual dentro del softmax (sección 6.2).

### 3.4 Flujo de decisión (carga diferida)

1. **Señales baratas** (`prior`, `popularity`, `context`): si el mejor alcanza $p_i \geq 0{,}7$, se elige.
2. **Señales de Wikidata** (`type`, `relation`, por SPARQL), solo para las menciones inciertas.
3. **LLM** (si `use_llm`), solo si sigue incierta: elige entre los candidatos y **su elección es la respuesta**, sin peso; la confianza sale de las señales:
   - si **coincide** con el mejor de las señales, se acepta sin revisión (`llm_confirmed`);
   - si elige **otro**, se devuelve, pero para revisión;
   - si **ninguno encaja**, NIL con el más probable como sugerencia.
4. **Revisión:** `needs_review` marca los NIL, los desacuerdos con el LLM y los enlaces con confianza menor que `review_threshold` (0,7, configurable por petición) no confirmados por el LLM.

**Límite de la carga diferida.** Si las señales baratas ya superan 0,7, `type` y `relation` no se calculan. En "Michael Jordan, a professor at the University of California, Berkeley", el jugador obtiene $p = 0{,}701$ solo por popularidad y `relation`, que uniría a Michael I. Jordan con Berkeley, no llega a consultarse. Igual ocurre con los candidatos únicos ($p = 1$).

### 3.5 Explicabilidad

`reasoning` da la $p_i$ del elegido y las dos señales que más lo separan del segundo, por ejemplo: "p=0.88 sobre 20 candidatos; frente a Q49088 (Columbia University): relation +4.3, context +0.7".

## 4. Evaluación y casos límite

**Métricas**, por fase y del conjunto:
- NER y enlace: `nervaluate` con el QID como etiqueta, al estilo GERBIL (F1 `mention`, `all`, `linked`, `nil`).
- Recuperación: `ranx` (recall@k, MRR).
- Desambiguación: accuracy con IC del 95 % por *bootstrap* de documentos; las configuraciones se comparan sobre las mismas menciones (*bootstrap* pareado).
- Confianza: ECE y qué parte de los errores llega a revisión.

**Datos:**

| Conjunto | Origen | Uso | Tamaño |
|---|---|---|---|
| AIDA CoNLL-YAGO (con QIDs) | Noticias Reuters de 1996, todas las entidades anotadas | Todas las fases: **validation** para los pesos, **test** para medir | Test: 231 documentos, 5.614 menciones (20 % NIL) |
| KORE50, MSNBC, AQUAINT, ACE2004 | Literatura (naist-nlp), convertidos a QIDs | Generalización fuera de dominio (KORE50: homónimos difíciles) | 136 / 595 / 399 / 283 menciones |
| WikiAnc | Enlaces internos de Wikipedia | Recuperación en texto enciclopédico | 198 enlaces |

**Muestras que aíslan la desambiguación** (`scripts/build_disambiguation_sample.py`): menciones en que GLiNER acierta los límites y el correcto está entre los 20 candidatos, comparadas con el primer candidato de la búsqueda. Ejemplo, KORE50: 57 menciones; primer candidato 0,421, sistema 0,772 (**+35,1 puntos, IC 95 % [+21,6, +47,9]**); arregla 20 de los 33 fallos y no estropea ninguno ("Dylan": el primer candidato es el nombre de pila; `popularity` y `type` eligen a Bob Dylan).

**Resultados:**

| Fase | Métrica | Resultado |
|---|---|---|
| NER (10 documentos de AIDA) | F1 estricto | **0,849** con tipos finos frente a 0,667 con etiquetas genéricas |
| Recuperación, AIDA test completo (4.483 menciones con QID) | recall@1 / @5 / @10 / @20 · MRR | 0,58 / 0,74 / 0,78 / **0,81** · 0,65 |
| Recuperación, WikiAnc | recall@1 / @10 / @20 · MRR | 0,72 / 0,80 / **0,82** · 0,75 (sin redirecciones: 0,64 / 0,75 / 0,77 · 0,68) |
| Desambiguación (557 menciones de test, sin LLM) | accuracy | primer candidato 0,689 → **0,763** (+0,074, IC 95 % [+0,037, +0,120]); `relation` aporta +0,036 (IC [+0,015, +0,064]). Con LLM: sección 5 |
| End-to-end sin LLM (10 documentos) | F1 `mention` / `all` / `linked` / `nil` | 0,864 / **0,463** / **0,500** / 0,273 |
| End-to-end con LLM (10 documentos) | F1 `all` / `linked` / `nil` | 0,475 / 0,514 / 0,281 (~8 llamadas por documento) |

**Lectura:**
- **El cuello de botella es la recuperación:** el correcto falta en el 19 % de las menciones de AIDA. `wbsearchentities` falla cuando el texto no nombra la entidad como su etiqueta: nombres abreviados ("S.Philpott"), gentilicios ("Indian" → India) y orden por etiqueta literal ("Asian Cup"). Las redirecciones de Wikipedia son la mejora más rentable (recall@1 +4 a +8 puntos).
- **La popularidad es la señal más fuerte y la fuente de los errores con confianza:** en crónicas deportivas, "Syria" es la selección, pero el país se acepta en el paso 1.
- **`relation` resuelve homónimos de la misma clase**, donde `type` no ayuda: "Columbia" junto a Sinatra, Dylan y Santana pasa de la universidad a **Columbia Records**. Es la coherencia por relaciones de la literatura (Milne y Witten, AIDA, REL): accuracy 0,727 → 0,763; en posiciones 2-10, 0,24 → 0,38.
- **Fuera de dominio el primer candidato ya es muy bueno:** en AQUAINT acierta el 98,4 % y el sistema el 96,2 % (diferencia no significativa); en KORE50 el sistema sube del 42 % al 77 %.
- **NIL:** sin LLM solo se predice cuando no hay candidatos (precisión 0,83, recall 0,16). La medida es pesimista: en AIDA, NIL significa «sin artículo en Wikipedia en 2010», y muchas de esas entidades existen hoy en Wikidata (26 de 40 revisados estaban bien enlazados).

**Casos límite:**
- **Inexistentes en Wikidata:** NIL si no hay candidatos o si el LLM lo indica (con sugerencia); si comparten nombre con una entidad existente, se enlazan con ella, a menudo con confianza alta: el punto débil del sistema (sección 3.3).
- **Nuevas:** Wikidata se consulta en vivo; una entidad creada hoy ya es candidata.
- **Poco conocidas o ambiguas:** confianza baja, `needs_review` y candidatos alternativos ordenados.
- **Un solo candidato o fallo de recuperación:** pueden salir con confianza alta siendo errores (sección 3.3).
- **Falsos positivos del NER** ("village pie contest"): sin candidatos, salen como NIL con revisión.

## 5. Optimización con LLM propio

**Dónde se usa.** Solo en la desambiguación de las menciones inciertas (~26–28 % en AIDA, unas 8 llamadas por documento). En el NER se descartó: mucho más lento que GLiNER y sin *offsets* exactos.

**Lo medido** (557 menciones de AIDA test, mismas señales):

| Selector de las inciertas | Accuracy | Mejora sobre solo señales | Coste |
|---|---|---|---|
| Solo señales | 0,763 | — | ~1,5 s/documento |
| qwen2.5:3b local (Ollama, Q4, GPU de 4 GB) | 0,786 | +2,3 (IC [−0,5, +5,9], no significativa) | 15 s por llamada con la GPU limitada por temperatura |
| Claude Sonnet (API de terceros) | **0,873** | **+11,0** (IC [+5,5, +18,8]) | ~1,5 s por llamada |

El 3B ayuda cuando el correcto no es el primero (posiciones 2-10: 0,38 → 0,53), pero estropea aciertos (posición 1: 0,966 → 0,943); Claude apenas estropea (0,961). Un modelo pequeño propio apenas aporta y uno grande sí: las optimizaciones buscan acercar el propio al grande sin su coste.

| Optimización | Qué | Impacto | Estado |
|---|---|---|---|
| **Cascada (*fast-path*)** | El LLM solo si $p < 0{,}7$ | Evita ~3 de cada 4 llamadas | Aplicada |
| **Prompt y salida mínimos** | Elección múltiple → `{"choice": n}` (máximo 20 tokens, sin las puntuaciones de las señales); top-5 candidatos en vez de 20 | Leer el prompt domina el tiempo (4,0 s para ~1.000 tokens frente a 0,5 s para generar 8) | Salida JSON aplicada; top-5 propuesto |
| **Servidor con *batching*** | vLLM o TensorRT-LLM: *continuous batching*, *prefix caching* del documento | Varias menciones en paralelo | Propuesta |
| **Destilación (LoRA / QLoRA)** | Ajustar un modelo de 1,5B–3B con las decisiones de un LLM grande y las correcciones de la revisión | Acercar los +11 puntos del grande al coste del pequeño | Propuesta |

**Alternativas valoradas:**

| Alternativa | Decisión | Motivo |
|---|---|---|
| **Pipeline híbrido:** GLiNER + señales (*fast-path*) + LLM local cuantizado solo en las dudas | ✅ elegida | **Trazabilidad y explicabilidad**: cada decisión se explica por señales con su peso, y el LLM solo interviene en las dudas, con su desacuerdo visible; además, lo barato resuelve la mayoría, sin coste por token, sin depender de terceros y sin que los datos salgan de la infraestructura |
| API de terceros (Claude, GPT) | ❌ referencia | El mejor acierto, pero con coste por token, dependencia externa y datos fuera. Sirve de «profesor» para destilar el modelo propio |
| LLM como señal con peso en el logit | ❌ | Había que fijar su peso a mano y sus errores salían con confianza alta (p = 0,88) |
| Modelos de *entity linking* entrenados (ReFinED) | ❌  | 0,944 en AIDA (su dominio de entrenamiento), empata fuera de dominio; índice congelado. Caja negra. Adaptación más costosa |

## 6. Producción y mantenimiento

### 6.1 Valoración de la arquitectura actual

| A favor | En contra |
|---|---|
| **Explicable:** cada decisión lleva sus señales | **Techo de acierto:** 0,763 en la muestra; los sistemas entrenados de la literatura rondan el 90 % en AIDA (en su dominio) |
| **Revisión humana integrada:** `needs_review` y candidatos alternativos ordenados | **Dependencia de la red:** Wikidata, Wikipedia y SPARQL, con límites de uso y latencia |
| **Reentrenable con pocos datos:** 5 pesos; unos cientos de menciones por dominio | **La recuperación es el cuello de botella** (19 % de fallos en AIDA) |
| **Datos al día:** una entidad nueva existe al instante | **Contexto superficial:** palabras compartidas, sin comprensión semántica |
| **Multilingüe por diseño** | **Pesos aprendidos con noticias en inglés de 1996** |
| **Barata:** CPU, más un LLM pequeño solo para las inciertas | **Recall de NIL bajo sin LLM** y **confianza no calibrada** (sección 3.3) |

### 6.2 Opciones para escalar

**A. Conservar la metodología y adaptarla al dominio.** Para un dominio nuevo bastaría con:
- **adaptar los tipos finos** que se piden a GLiNER y su clase de Wikidata para la señal `type` (`ner.py`, `disambiguation/signals.py`); GLiNER es *zero-shot*, así que no hace falta reentrenarlo salvo que el dominio sea muy específico;
- **reaprender los pesos** con menciones anotadas del dominio (`tune_weights.py`), que el bucle de revisión humana va produciendo;
- **revisar los umbrales** (`CONFIDENT_PROB`, `review_threshold`) con esos mismos datos.

Posibles mejoras:

**Candidato NIL virtual.** El softmax reparte el 100 % de la probabilidad entre los candidatos, aunque todos sean malos. Un candidato nulo con una puntuación aprendida $b$ amplía la decisión a K + 1 opciones (con $\text{score}_i$ como en la sección 3.2):

$$p_i = \frac{e^{\text{score}_i}}{e^{b} + \sum_j e^{\text{score}_j}} \qquad\qquad p_{\text{NIL}} = \frac{e^{b}}{e^{b} + \sum_j e^{\text{score}_j}}$$

Se entrena en un solo paso junto a los pesos; si ningún candidato supera a $b$, gana NIL sin depender del LLM; y resuelve a la vez la detección de NIL y la confianza no calibrada (un candidato único deja de valer 1).

**Independencia de Wikidata en tiempo de ejecución.** Las mismas señales leídas de un índice local (etiquetas, alias, redirecciones y p(e | m) de los enlaces de Wikipedia) y de tablas precalculadas de P31/P279* y relaciones, construidos a partir del *dump*. Elimina la red y mejora la recuperación; a cambio, el índice se congela hasta su regeneración. Con las señales en local, `type` y `relation` se calcularían siempre (desaparece el límite de la sección 3.4), y solo el LLM se reserva para las inciertas.

Implicación: **mantiene la explicabilidad y la trazabilidad**: cada decisión sigue explicada por señales con su peso, y cada cambio de pesos o de datos se puede medir y auditar.

**B. Modelos entrenados de la literatura.**

| Modelo | Qué es | Encaje |
|---|---|---|
| **ReFinED** (Amazon) | NER, tipado y desambiguación sobre Wikidata en una red | Muy preciso y rápido en GPU; caja negra, índice congelado, inglés |
| **BLINK** | Recuperación densa por *embeddings* + reordenación | Resuelve la **recuperación**; infraestructura pesada (índice ANN, GPU) |
| **mGENRE** | Generativo y multilingüe | Preciso; caro de servir |

Más acierto **en su dominio**, pero se pierden la explicabilidad y la trazabilidad; su puntuación tampoco está calibrada; índice congelado; adaptarlos exige **miles** de ejemplos anotados y GPU.

**C. LLM de terceros (APIs).** El mismo prompt de elección múltiple, sustituyendo al LLM local (nunca pedirle "el QID": inventa identificadores). Razona mucho mejor (+11,0 frente a +2,3), pero con coste por token, latencia y límites de uso, datos fuera de la infraestructura, y versiones que cambian sin aviso (resultados no reproducibles); su elección sigue necesitando contraste con las señales y revisión.

## 7. Uso de la IA en la resolución de la prueba

He usado Claude Code (Claude, de Anthropic) y NotebookLM (Gemini, de Google) como asistentes durante todo el desarrollo: para la revisión de literatura, para escribir y refactorizar el código, para lanzar los experimentos de evaluación y para redactar la documentación (memoria, README y notebook). Las propuestas de la IA se han corregido, replanteado o simplificado de forma iterativa. Todo el código y las cifras se han revisado y reejecutado antes de incluirlos.

## 8. Resumen

| Apartado | Decisión | Evidencia |
|---|---|---|
| 1. Alcance | Inglés y español; 19 tipos finos agrupados en PER/ORG/LOC/MISC | Correcto 1.º: 69 % (AIDA) a 98 % (AQUAINT); 20 % NIL en AIDA; ~1,5 s/documento sin LLM |
| 2. NER | GLiNER zero-shot con tipos finos | F1 0,849 frente a 0,667 |
| 2. Recuperación | `wbsearchentities` + redirecciones de Wikipedia + búsqueda aproximada, 20 candidatos | recall@20 0,81 en AIDA y 0,82 en WikiAnc |
| 3. Desambiguación | Logit condicional con pesos aprendidos y carga diferida | accuracy 0,689 → 0,763; explicación por señales |
| 4. NIL y revisión | NIL si no hay candidatos o lo indica el LLM; `needs_review` con la confianza del ranker | precisión de NIL 0,83, recall bajo; confianza no calibrada (ECE 0,141) |
| 5. LLM propio | qwen2.5:3b, solo para inciertas; decide sin peso y su desacuerdo va a revisión | Local +2,3 (no significativo) frente a Claude +11,0 |

---

# Opciones consideradas en cada fase

### NER

| Opción | Resultado | Motivo |
|---|---|---|
| spaCy (`en_core_web_*`) | ❌ | Etiquetas fijas; no permite pedir tipos útiles para Wikidata (*sports team*, *award*) |
| GLiNER con etiquetas genéricas | ❌ | F1 0,667; LOC 0,248 (confunde países con nacionalidades y equipos) |
| **GLiNER con 19 tipos finos** | ✅ | F1 0,849; el tipo fino alimenta la señal `type`. *work of art* añade obras sin empeorar AIDA (0,843 → 0,849) |
| LLM local (qwen2.5:7b / 3b) | ❌ | Mucho más lento y sin *offsets* exactos |

### Recuperación de candidatos

| Opción | Resultado | Motivo |
|---|---|---|
| **`wbsearchentities` (etiqueta y alias, por prefijo)** | ✅ | Rápida y estable |
| **+ redirecciones de Wikipedia** | ✅ | Diccionario de nombres mantenido por los editores (formas cortas, abreviaturas); recall@1 +4 a +8 puntos |
| **+ búsqueda aproximada** (RapidFuzz `token_sort_ratio` ≥ 80), solo sin resultados | ✅ | Recupera nombres abreviados; `WRatio` dejaba pasar ruido |
| Búsqueda de texto completo siempre | ❌ | Más llamadas y ruido |
| Búsqueda por etiqueta con SPARQL | ❌ | Lenta y con límites de uso estrictos |
| Búsqueda semántica (wd-vectordb) | ❌ | Latencia, errores 429 e índice desactualizado |

### Información de tipos (clases de Wikidata)

| Opción | Resultado | Motivo |
|---|---|---|
| SPARQL con la jerarquía P279* completa | ❌ | Respuestas enormes (235 MB de caché) y lentas |
| **SPARQL solo de las clases raíz pedidas, con caché por QID** | ✅ | Unas 60 veces más ligera |
| Pedir las clases de todas las menciones | ❌ | Coste inútil → **solo las inciertas** |

### Desambiguación

| Opción | Resultado | Motivo |
|---|---|---|
| Primer candidato de la búsqueda | Referencia | accuracy 0,689 |
| Heurística con pesos a mano y umbral NIL de 0,35 | ❌ | Arbitraria y difícil de defender |
| **Logit condicional (`scipy.optimize`, pesos ≥ 0)** | ✅ | 0,763; su probabilidad decide la carga diferida y es la confianza devuelta |
| Contexto que incluye la mención | ❌ | Mezclaba nombre y contexto → **contexto sin la mención** |
| **+ popularidad (sitelinks)** | ✅ | La señal más fuerte; +3 puntos |
| Coherencia por clases compartidas (`coherence`) | ❌ eliminada | Con `relation`, no aportaba en ningún dataset |
| **+ coherencia por relaciones** | ✅ | 0,727 → 0,763; resuelve "Columbia" junto a Sinatra, Dylan y Santana |
| Candidato NIL virtual dentro del softmax | Propuesto (sección 6.2) | NIL y confianza absoluta en el mismo modelo; exige reentrenar y volver a medir |
| Cross-encoder (bge-reranker) como señal semántica | Aplazado | Entrenado para ordenar pasajes, no entidades: en una crónica deportiva prefería *Syria* país a la selección. ~1,6 s por mención en CPU. Requiere un *cross-encoder* de *entity linking* (BLINK) y GPU |

### LLM en la desambiguación

| Opción | Resultado | Motivo |
|---|---|---|
| APIs de terceros (Claude Sonnet) | ❌ referencia | +11,0 puntos, pero coste por token, privacidad y dependencia externa; se pide un LLM propio |
| qwen2.5:7b | ❌ | No cabe en 4 GB de GPU: más de 100 s por documento |
| **qwen2.5:3b Q4 con Ollama** | ✅ | Cabe en la GPU; 0,3–15 s por llamada; +2,3 puntos (no significativo) |
| LLM en todas las menciones | ❌ | Muy lento → **solo las inciertas** |
| Prompt libre, o con las puntuaciones de las señales | ❌ | → **elección múltiple** `{"choice": n}`, sin puntuaciones (evita que las copie) |
