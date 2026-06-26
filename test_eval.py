import unittest
import os
import shutil
import tempfile
from evaluator import LLMEvaluator, EvalHarnessSuite, DatasetEvaluationSummary

class TestLlmEvalHarness(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.temp_dir, "baseline.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_tokenization_and_jaccard(self) -> None:
        """Verifica el tokenizador nativo y la similitud Jaccard."""
        text_a = "Hola, mundo! Esto es una prueba."
        tokens_a = LLMEvaluator.clean_and_tokenize(text_a)
        # Deberia remover la puntuacion y separar
        self.assertEqual(tokens_a, ["hola", "mundo", "esto", "es", "una", "prueba"])
        
        text_b = "Hola mundo de pruebas."
        tokens_b = LLMEvaluator.clean_and_tokenize(text_b)
        
        # Jaccard de A y B
        # Intersección: {"hola", "mundo", "prueba"/"pruebas" no coincide} -> {"hola", "mundo"} (2 tokens)
        # Unión: {"hola", "mundo", "esto", "es", "una", "prueba", "de", "pruebas"} (8 tokens)
        # Jaccard = 2 / 8 = 0.25
        jaccard = LLMEvaluator.compute_jaccard(tokens_a, tokens_b)
        self.assertAlmostEqual(jaccard, 0.25)

    def test_cosine_similarity(self) -> None:
        """Verifica la similitud de coseno basada en frecuencias."""
        tokens_a = ["el", "gato", "duerme", "el", "gato"]
        tokens_b = ["el", "perro", "juega"]
        
        # Vocabulario único: {"el", "gato", "duerme", "perro", "juega"} (5 términos)
        # Vector A: [2, 2, 1, 0, 0] -> Norm: sqrt(4+4+1) = 3.0
        # Vector B: [1, 0, 0, 1, 1] -> Norm: sqrt(1+1+1) = sqrt(3) = 1.732
        # Producto punto: 2*1 + 2*0 + 1*0 + 0*1 + 0*1 = 2
        # Similitud coseno: 2 / (3 * 1.732) = 2 / 5.196 = 0.3849
        cos_sim = LLMEvaluator.compute_cosine_similarity(tokens_a, tokens_b)
        self.assertAlmostEqual(cos_sim, 2.0 / (3.0 * 1.7320508075688772), places=5)

    def test_native_bleu_and_brevity_penalty(self) -> None:
        """Prueba el cálculo de BLEU nativo y penalización por brevedad."""
        # Candidato idéntico al de referencia
        tokens_cand_exact = ["the", "cat", "sat", "on", "the", "mat"]
        tokens_ref = ["the", "cat", "sat", "on", "the", "mat"]
        
        bleu_exact = LLMEvaluator.compute_bleu(tokens_cand_exact, tokens_ref)
        # Debería dar un valor cercano a 1.0 (exactamente 1.0 ya que coinciden todos los n-gramas y longitudes)
        self.assertAlmostEqual(bleu_exact, 1.0)
        
        # Candidato muy corto (Penalización por brevedad activa)
        tokens_cand_short = ["the", "cat"]
        bleu_short = LLMEvaluator.compute_bleu(tokens_cand_short, tokens_ref)
        # Al ser más corto que la referencia (longitud 2 vs 6), BP = exp(1 - 6/2) = exp(-2) = 0.135
        # BLEU debería ser significativamente penalizado
        self.assertLess(bleu_short, 0.25)

    def test_regression_detection(self) -> None:
        """Verifica que la detección de regresión lance advertencias correctas."""
        # 1. Crear resumen baseline
        baseline = DatasetEvaluationSummary(
            avg_exact_match=0.5,
            avg_jaccard_similarity=0.8,
            avg_cosine_similarity=0.85,
            avg_bleu_score=0.75,
            avg_length_ratio=1.0,
            total_samples=10
        )
        
        suite = EvalHarnessSuite()
        suite.save_as_baseline(baseline, self.baseline_path)
        
        # Cargar baseline
        suite.load_baseline(self.baseline_path)
        
        # Caso A: Reporte actual similar o mejor (sin regresiones)
        current_ok = DatasetEvaluationSummary(
            avg_exact_match=0.5,
            avg_jaccard_similarity=0.81, # mejor
            avg_cosine_similarity=0.84, # levemente menor pero dentro de tolerancia (1.1% menor)
            avg_bleu_score=0.74, # levemente menor pero dentro de tolerancia (1.3% menor)
            avg_length_ratio=0.98,
            total_samples=10
        )
        regression_a, warnings_a = suite.check_for_regressions(current_ok, tolerance=0.05)
        self.assertFalse(regression_a)
        self.assertEqual(len(warnings_a), 0)
        
        # Caso B: Reporte actual con regresión grave (BLEU cae más del 5% de tolerancia)
        # 0.75 -> 0.70 (caída del 6.67%)
        current_regression = DatasetEvaluationSummary(
            avg_exact_match=0.5,
            avg_jaccard_similarity=0.8,
            avg_cosine_similarity=0.85,
            avg_bleu_score=0.70, # Caída significativa
            avg_length_ratio=1.0,
            total_samples=10
        )
        regression_b, warnings_b = suite.check_for_regressions(current_regression, tolerance=0.05)
        self.assertTrue(regression_b)
        self.assertTrue(any("avg_bleu_score" in w or "BLEU" in w for w in warnings_b))


if __name__ == '__main__':
    unittest.main()
