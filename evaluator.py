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
