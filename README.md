# LLM Eval Harness

Un arnes de evaluacion y pruebas automatizadas de calidad de respuesta para modelos de lenguaje (LLMs), orientado a auditar rendimientos y alertar regresiones de comportamiento en entornos de integracion continua de MLOps.

El modulo implementa metricas lexicas de comparacion textual y similitud n-gramas de forma nativa en Python, eliminando la necesidad de dependencias externas complejas y garantizando su ejecucion local fuera de linea.

## Metricas de Evaluacion Implementadas

El harness evalua de manera integral las siguientes dimensiones del texto generado:

1.  **Exact Match (Coincidencia Exacta):**
    *   Determina si la salida generada es exactamente igual a la referencia dorada tras remover espacios en blanco iniciales y finales. Es util en tareas deterministas como generacion de codigo estructurado o respuestas de clasificacion.
2.  **Similitud Jaccard:**
    *   Mide la relacion de tokens compartidos entre el candidato y la referencia dividiendo la cantidad de palabras de la interseccion entre el tamano de la union de conjuntos. Ofrece una lectura rapida de superposicion de palabras clave.
3.  **Similitud de Coseno de Bolsa de Palabras (Bag of Words):**
    *   Construye vectores de frecuencias de terminos y calcula el coseno del angulo formado entre ambos. Es sensible a la frecuencia de repeticion de terminos clave en la generacion.
4.  **BLEU-4 Score (Bilingual Evaluation Understudy):**
    *   Implementa el algoritmo de precision de n-gramas recortada para $N=1,2,3,4$ con penalizacion por brevedad (Brevity Penalty). Incorpora metodos de suavizado (smoothing) para evitar puntuaciones nulas en cadenas cortas que no llegan a copar 4-gramas.
5.  **Ratio de Coherencia de Longitud:**
    *   Compara el conteo de tokens de la respuesta respecto a la referencia para vigilar alucinaciones verborreicas o respuestas incompletas de una sola palabra.

## Control de Regresiones en Produccion

El sistema permite serializar reportes completos de evaluacion como "Líneas Base" (Baselines). En ejecuciones posteriores (por ejemplo, tras re-entrenar un modelo o actualizar prompts), el arnes compara el nuevo reporte contra la linea base. Si alguna metrica de calidad se degrada mas alla del porcentaje de tolerancia configurado (por ejemplo, 5%), el sistema activa una alerta indicando la regresion y el desglose de perdida de precision.

## Requisitos de Instalacion

*   Python 3.8 o superior
*   Pydantic

Para instalar las dependencias locales, ejecute:

```bash
pip install -r requirements.txt
```

## Pruebas y Verificacion

1.  **Ejecutar la suite de pruebas unitarias:**
    ```bash
    python -m unittest test_eval.py
    ```
2.  **Ejecutar la demostracion de evaluacion:**
    ```bash
    python example.py
    ```
    El script demostrara la evaluacion de un modelo base de referencia, su almacenamiento en formato JSON como baseline, la evaluacion de un modelo experimental de inferior calidad, y el disparo de las alertas de regresion en consola.
