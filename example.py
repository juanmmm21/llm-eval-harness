import os
import shutil
from evaluator import EvalHarnessSuite

def run_demo() -> None:
    print("=" * 75)
    print("      Demostración de Evaluación y Detección de Regresiones (MLOps)     ")
    print("=" * 75)
    
    # 1. Dataset de prueba con referencias doradas (ground truth)
    print("\n--- PASO 1: Cargando dataset de evaluacion con referencias doradas ---")
    prompts = [
        "Escribe una funcion en Python para calcular el factorial de un numero de forma recursiva.",
        "Explica brevemente que es RAG (Generacion Aumentada por Recuperacion) en Inteligencia Artificial.",
        "Como se calcula la similitud del coseno entre dos vectores?"
    ]
    
    references = [
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "RAG es una tecnica que combina la recuperacion de documentos externos de una base de datos con un modelo de lenguaje (LLM) para generar respuestas mas precisas y actualizadas.",
        "La similitud del coseno se calcula dividiendo el producto punto de los dos vectores por el producto de sus normas euclidianas."
    ]
    
    # 2. Respuestas simuladas del Modelo A (Línea Base)
    # Genera respuestas completas y muy cercanas al contenido de referencia.
    candidates_model_a = [
        "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)",
        "RAG (Retrieval-Augmented Generation) es un enfoque que recupera informacion o documentos relevantes de una base de datos de vectores para complementar el prompt de un LLM, reduciendo alucinaciones.",
        "Se calcula haciendo el producto punto de los vectores y dividiendolo por la multiplicacion de la norma de cada vector."
    ]
    
    # 3. Respuestas simuladas del Modelo B (Modelo Experimental con Regresiones)
    # Genera respuestas incompletas, con errores o excesivamente cortas.
    candidates_model_b = [
        "def fact(n):\n    return n * fact(n)", # error lógico e infinito
        "RAG es algo para buscar en google antes de hablar con el chat.", # pobre e incorrecta
        "Es el coseno del angulo entre ellos." # extremadamente corto
    ]
    
    # Empaquetar muestras
    samples_a = []
    samples_b = []
    for i in range(len(prompts)):
        samples_a.append({
            "prompt": prompts[i],
            "reference": references[i],
            "candidate": candidates_model_a[i]
        })
        samples_b.append({
            "prompt": prompts[i],
            "reference": references[i],
            "candidate": candidates_model_b[i]
        })
        
    # 4. Evaluar Modelo A y guardar como Línea Base (Baseline)
    print("\n--- PASO 2: Evaluando Modelo A (Modelo Base de Referencia) ---")
    suite = EvalHarnessSuite()
    summary_a = suite.run_evaluation(samples_a)
    
    print("\n[Métricas Modelo A (Baseline)]:")
    print(f"  Exact Match Promedio: {summary_a.avg_exact_match:.4f}")
    print(f"  Similitud Jaccard Promedio: {summary_a.avg_jaccard_similarity:.4f}")
    print(f"  Similitud Coseno Promedio: {summary_a.avg_cosine_similarity:.4f}")
    print(f"  BLEU-4 Score Promedio: {summary_a.avg_bleu_score:.4f}")
    print(f"  Ratio de Longitud Promedio: {summary_a.avg_length_ratio:.4f}")
    
    baseline_file = "./baselines/model_a_report.json"
    suite.save_as_baseline(summary_a, baseline_file)
    
    # 5. Evaluar Modelo B y comparar contra la línea base del Modelo A
    print("\n--- PASO 3: Evaluando Modelo B (Candidato Experimental) ---")
    summary_b = suite.run_evaluation(samples_b)
    
    print("\n[Métricas Modelo B (Actual)]:")
    print(f"  Exact Match Promedio: {summary_b.avg_exact_match:.4f}")
    print(f"  Similitud Jaccard Promedio: {summary_b.avg_jaccard_similarity:.4f}")
    print(f"  Similitud Coseno Promedio: {summary_b.avg_cosine_similarity:.4f}")
    print(f"  BLEU-4 Score Promedio: {summary_b.avg_bleu_score:.4f}")
    print(f"  Ratio de Longitud Promedio: {summary_b.avg_length_ratio:.4f}")
    
    # 6. Analizar regresiones lógicas
    print("\n--- PASO 4: Analizando regresiones con respecto al Modelo A ---")
    # Cargar línea base guardada
    regression_suite = EvalHarnessSuite(baseline_file)
    
    # Definimos una tolerancia del 5% (0.05)
    regression_detected, warnings = regression_suite.check_for_regressions(summary_b, tolerance=0.05)
    
    if regression_detected:
        print("\n!!! ALERTA DE COMPILACION DE MODELO: REGRESIONES DETECTADAS !!!")
        for warning in warnings:
            print(f"  [ALERTA] {warning}")
    else:
        print("\n-> [EXITO] Ninguna regression detectada. El modelo B esta aprobado para produccion.")
        
    # Limpieza
    if os.path.exists("./baselines"):
        shutil.rmtree("./baselines")
        print("\nEntorno de evaluacion temporal limpiado correctamente.")


if __name__ == "__main__":
    run_demo()
