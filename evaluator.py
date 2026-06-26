import os
import json
import math
import string
import logging
from typing import Dict, List, Any, Tuple, Optional
from pydantic import BaseModel, Field

# Configuración del log para auditoría en MLOps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricReport(BaseModel):
    """Reporte de métricas individuales de un texto evaluado."""
    exact_match: float
    jaccard_similarity: float
    cosine_similarity: float
    bleu_score: float
    length_ratio: float


class DatasetEvaluationSummary(BaseModel):
    """Resumen consolidado de la evaluación sobre un conjunto de datos."""
    avg_exact_match: float
    avg_jaccard_similarity: float
    avg_cosine_similarity: float
    avg_bleu_score: float
    avg_length_ratio: float
    total_samples: int
    individual_results: List[Dict[str, Any]] = Field(default_factory=list)


class LLMEvaluator:
    """
    Evaluador automatizado de calidad de generación para LLMs.
    
    Implementa métricas léxicas, solapamiento n-gramas (BLEU) y similitud de términos
    de forma nativa sin dependencias externas pesadas para asegurar su portabilidad.
    """

    @staticmethod
    def clean_and_tokenize(text: str) -> List[str]:
        """Limpia el texto convirtiéndolo a minúsculas y removiendo la puntuación básica."""
        if not text:
            return []
        text = text.lower()
        # Reemplazar puntuación por espacios
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        return [word for word in text.split() if word]

    @staticmethod
    def compute_jaccard(candidate_tokens: List[str], reference_tokens: List[str]) -> float:
        """Calcula la similitud de Jaccard (intersección sobre unión) a nivel de tokens."""
        set_cand = set(candidate_tokens)
        set_ref = set(reference_tokens)
        
        if not set_cand and not set_ref:
            return 1.0
        if not set_cand or not set_ref:
            return 0.0
            
        intersection = set_cand.intersection(set_ref)
        union = set_cand.union(set_ref)
        return len(intersection) / len(union)

    @staticmethod
    def compute_cosine_similarity(candidate_tokens: List[str], reference_tokens: List[str]) -> float:
        """Calcula la similitud de coseno basada en bolsa de palabras (Bag-of-Words)."""
        if not candidate_tokens and not reference_tokens:
            return 1.0
        if not candidate_tokens or not reference_tokens:
            return 0.0
            
        # Contar frecuencias de términos
        vocab = set(candidate_tokens + reference_tokens)
        
        freq_cand = {}
        freq_ref = {}
        for token in vocab:
            freq_cand[token] = candidate_tokens.count(token)
            freq_ref[token] = reference_tokens.count(token)
            
        # Producto punto e intensidades vectoriales
        dot_product = 0.0
        norm_cand = 0.0
        norm_ref = 0.0
        
        for token in vocab:
            dot_product += freq_cand[token] * freq_ref[token]
            norm_cand += freq_cand[token] ** 2
            norm_ref += freq_ref[token] ** 2
            
        norm_cand = math.sqrt(norm_cand)
        norm_ref = math.sqrt(norm_ref)
        
        if norm_cand == 0.0 or norm_ref == 0.0:
            return 0.0
            
        return dot_product / (norm_cand * norm_ref)

    @staticmethod
    def _extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extrae tuplas de n-gramas a partir de una lista de tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams

    @classmethod
    def compute_bleu(cls, candidate_tokens: List[str], reference_tokens: List[str]) -> float:
        """
        Calcula el BLEU score (BLEU-4) nativo con suavizado para oraciones cortas.
        
        Implementa conteos recortados de n-gramas y penalización por brevedad.
        """
        c = len(candidate_tokens)
        r = len(reference_tokens)
        
        if c == 0:
            return 0.0
            
        # 1. Brevity Penalty (Penalización por brevedad)
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - (r / c))
            
        # 2. Precisión n-gramas modificada para n=1..4
        precisions = []
        for n in range(1, 5):
            cand_ngrams = cls._extract_ngrams(candidate_tokens, n)
            ref_ngrams = cls._extract_ngrams(reference_tokens, n)
            
            if not cand_ngrams:
                # Si el candidato no tiene suficientes palabras para este n-grama, suavizamos
                precisions.append(0.1 / len(candidate_tokens))
                continue
                
            # Conteo de referencias recortado (para evitar engaños de duplicados)
            ref_counts = {}
            for ng in ref_ngrams:
                ref_counts[ng] = ref_counts.get(ng, 0) + 1
                
            match_count = 0
            cand_seen = {}
            for ng in cand_ngrams:
                if ng in ref_counts:
                    # Permitir sólo hasta la frecuencia máxima observada en la referencia
                    if cand_seen.get(ng, 0) < ref_counts[ng]:
                        match_count += 1
                        cand_seen[ng] = cand_seen.get(ng, 0) + 1
                        
            # Suavizado de Laplace simple si no hay coincidencias exactas en n-gramas superiores
            if match_count == 0:
                p_n = 0.1 / len(cand_ngrams)
            else:
                p_n = match_count / len(cand_ngrams)
                
            precisions.append(p_n)
            
        # 3. Media geométrica de precisiones
        # BLEU = BP * exp(sum(w_n * log(p_n))) con w_n = 0.25
        try:
            geom_mean = math.exp(0.25 * sum(math.log(p) for p in precisions))
        except ValueError:
            # En caso de errores matemáticos o log de 0
            geom_mean = 0.0
            
        return bp * geom_mean

    @classmethod
    def evaluate_sample(cls, candidate: str, reference: str) -> MetricReport:
        """Calcula el reporte de métricas completo para un único par de textos."""
        # Limpieza básica
        cand_clean = candidate.strip()
        ref_clean = reference.strip()
        
        # Exact match
        exact = 1.0 if cand_clean == ref_clean else 0.0
        
        # Tokenizar
        cand_tokens = cls.clean_and_tokenize(cand_clean)
        ref_tokens = cls.clean_and_tokenize(ref_clean)
        
        # Calcular similitudes
        jaccard = cls.compute_jaccard(cand_tokens, ref_tokens)
        cosine = cls.compute_cosine_similarity(cand_tokens, ref_tokens)
        bleu = cls.compute_bleu(cand_tokens, ref_tokens)
        
        # Coherencia de longitud
        c_len = len(cand_tokens)
        r_len = len(ref_tokens)
        length_ratio = c_len / max(r_len, 1)
        
        return MetricReport(
            exact_match=exact,
            jaccard_similarity=jaccard,
            cosine_similarity=cosine,
            bleu_score=bleu,
            length_ratio=length_ratio
        )


class EvalHarnessSuite:
    """
    Arnés de ejecución de pruebas y detección de regresiones de modelos.
    """

    def __init__(self, baseline_filepath: Optional[str] = None) -> None:
        self.baseline_filepath = baseline_filepath
        self.baseline_data: Optional[DatasetEvaluationSummary] = None
        if baseline_filepath and os.path.exists(baseline_filepath):
            self.load_baseline(baseline_filepath)

    def load_baseline(self, filepath: str) -> None:
        """Carga un reporte previo de evaluación como línea base para comparar regresiones."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.baseline_data = DatasetEvaluationSummary(**data)
        logger.info(f"Línea base cargada desde '{filepath}'. Muestras: {self.baseline_data.total_samples}")

    def save_as_baseline(self, summary: DatasetEvaluationSummary, filepath: str) -> None:
        """Guarda la evaluación actual como un archivo de línea base de referencia."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Reporte de evaluación guardado como línea base en '{filepath}'")

    def run_evaluation(self, samples: List[Dict[str, str]]) -> DatasetEvaluationSummary:
        """
        Ejecuta la evaluación sobre una colección de dicts con llaves 'candidate' y 'reference'.
        """
        total_exact = 0.0
        total_jaccard = 0.0
        total_cosine = 0.0
        total_bleu = 0.0
        total_len_ratio = 0.0
        individual_results = []
        
        for idx, sample in enumerate(samples):
            candidate = sample.get("candidate", "")
            reference = sample.get("reference", "")
            prompt = sample.get("prompt", "")
            
            report = LLMEvaluator.evaluate_sample(candidate, reference)
            
            total_exact += report.exact_match
            total_jaccard += report.jaccard_similarity
            total_cosine += report.cosine_similarity
            total_bleu += report.bleu_score
            total_len_ratio += report.length_ratio
            
            # Guardamos los metadatos individuales para auditoría
            res = report.model_dump()
            res["prompt"] = prompt
            res["candidate"] = candidate
            res["reference"] = reference
            res["id"] = idx
            individual_results.append(res)
            
        total_samples = len(samples) if samples else 1
        summary = DatasetEvaluationSummary(
            avg_exact_match=total_exact / total_samples,
            avg_jaccard_similarity=total_jaccard / total_samples,
            avg_cosine_similarity=total_cosine / total_samples,
            avg_bleu_score=total_bleu / total_samples,
            avg_length_ratio=total_len_ratio / total_samples,
            total_samples=len(samples),
            individual_results=individual_results
        )
        return summary

    def check_for_regressions(self, current_summary: DatasetEvaluationSummary, tolerance: float = 0.05) -> Tuple[bool, List[str]]:
        """
        Compara las métricas actuales contra la línea base para reportar regresiones significativas.
        
        Args:
            current_summary: El reporte actual a verificar.
            tolerance: El porcentaje máximo admisible de disminución (por defecto 5%).
            
        Returns:
            - regression_detected: Booleano que indica si existe una degradación.
            - warnings: Lista de alertas detallando qué métricas sufrieron caídas.
        """
        if not self.baseline_data:
            return False, ["No hay datos de línea base cargados para comparar regresiones."]
            
        warnings = []
        regression_detected = False
        
        metrics_to_check = [
            ("avg_bleu_score", "BLEU Score", current_summary.avg_bleu_score, self.baseline_data.avg_bleu_score),
            ("avg_cosine_similarity", "Similitud Coseno", current_summary.avg_cosine_similarity, self.baseline_data.avg_cosine_similarity),
            ("avg_jaccard_similarity", "Similitud Jaccard", current_summary.avg_jaccard_similarity, self.baseline_data.avg_jaccard_similarity),
            ("avg_exact_match", "Exact Match", current_summary.avg_exact_match, self.baseline_data.avg_exact_match)
        ]
        
        for key, name, current_val, baseline_val in metrics_to_check:
            # Evitamos alertas si el baseline es cero
            if baseline_val == 0.0:
                continue
                
            # Calculamos la variación relativa
            change = (current_val - baseline_val) / baseline_val
            
            # Si el valor disminuye más de la tolerancia configurada
            if change < -tolerance:
                warnings.append(
                    f"REGRESION DETECTADA: La metrica '{name}' cayo un {abs(change)*100:.2f}% "
                    f"(Línea base: {baseline_val:.4f} | Actual: {current_val:.4f})"
                )
                regression_detected = True
            elif change < 0.0:
                logger.info(
                    f"Leve degradacion en '{name}': -{abs(change)*100:.2f}% (Dentro del margen de tolerancia)"
                )
                
        return regression_detected, warnings
