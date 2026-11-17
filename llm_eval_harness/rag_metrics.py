"""Métricas de RAG: exactitud de la respuesta, fidelidad al contexto y recall del contexto.

* **Exactitud**: ``exact_match`` y ``f1`` con la normalización oficial de SQuAD/HotpotQA
  (minúsculas, sin puntuación, sin artículos ingleses, espacios colapsados), tomando el máximo
  sobre las respuestas válidas; ``contains`` cuenta como acierto que una respuesta válida aparezca
  dentro de la predicción (útil con modelos que contestan con una frase).
* **Fidelidad** (``faithfulness``): fracción de las palabras de contenido de la respuesta que
  aparecen en el contexto recuperado. Es una cota léxica, barata y determinista: detecta
  respuestas inventadas, no paráfrasis infieles. Para un juicio semántico se puede pasar un
  ``judge`` (p. ej. un LLM) a :func:`judged_faithfulness`.
* **Recall del contexto** (``context_recall``): fracción de los párrafos de apoyo (por título)
  que están entre los recuperados; ``context_precision`` es la fracción de los recuperados que
  son de apoyo.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .datasets import RAGExample

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)
# Palabras sin contenido (inglés y español) que no cuentan para la fidelidad.
_STOP = set(
    "a an the of in on at to for from by with and or but is are was were be been being it its this that "
    "these those as which who whom what when where why how not no yes do does did has have had will would "
    "can could should may might than then there here into about over under also just only "
    "el la los las un una unos unas de del al en con por para y o pero es son fue fueron ser que como "
    "su sus lo le les se no si mas muy ya este esta estos estas ese esa".split()
)


def normalize_answer(text: str) -> str:
    """Normalización de SQuAD/HotpotQA."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in answers))


def f1_score(prediction: str, answers: Sequence[str]) -> float:
    """F1 de tokens normalizados, máximo sobre las respuestas válidas. Como en HotpotQA, si la
    respuesta es yes/no/noanswer solo cuenta la coincidencia exacta."""
    best = 0.0
    pred_norm = normalize_answer(prediction)
    for answer in answers:
        gold_norm = normalize_answer(answer)
        if gold_norm in {"yes", "no", "noanswer"} or pred_norm in {"yes", "no", "noanswer"}:
            best = max(best, float(pred_norm == gold_norm))
            continue
        pred_toks, gold_toks = pred_norm.split(), gold_norm.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        same = sum(common.values())
        if same == 0:
            continue
        precision, recall = same / len(pred_toks), same / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(prediction: str, answers: Sequence[str]) -> float:
    pred = f" {normalize_answer(prediction)} "
    return float(any(f" {normalize_answer(a)} " in pred for a in answers if normalize_answer(a)))


def content_words(text: str) -> List[str]:
    return [w for w in normalize_answer(text).split() if w not in _STOP]


def faithfulness(prediction: str, contexts: Sequence[str]) -> float:
    """Fracción de las palabras de contenido de la respuesta presentes en el contexto. Una
    respuesta sin palabras de contenido (``yes``, ``no``, vacía) cuenta como fiel."""
    words = content_words(prediction)
    if not words:
        return 1.0
    ctx = set(normalize_answer(" ".join(contexts)).split())
    return sum(1 for w in words if w in ctx) / len(words)


def judged_faithfulness(prediction: str, contexts: Sequence[str], judge: Callable[[str, str], bool]) -> float:
    """``judge(contexto, respuesta) -> bool`` decide si la respuesta se sigue del contexto."""
    return float(judge("\n\n".join(contexts), prediction))


def context_recall(retrieved_titles: Sequence[str], supporting_titles: Sequence[str]) -> float:
    if not supporting_titles:
        return 1.0
    got = set(retrieved_titles)
    return sum(1 for t in set(supporting_titles) if t in got) / len(set(supporting_titles))


def context_precision(retrieved_titles: Sequence[str], supporting_titles: Sequence[str]) -> float:
    if not retrieved_titles:
        return 0.0
    gold = set(supporting_titles)
    return sum(1 for t in retrieved_titles if t in gold) / len(retrieved_titles)


@dataclass
class RAGPrediction:
    """Lo que produjo el sistema evaluado para una pregunta."""

    example_id: str
    answer: str
    retrieved_titles: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    latency_s: Optional[float] = None


@dataclass
class RAGReport:
    """Métricas medias y por pregunta."""

    metrics: Dict[str, float]
    per_example: List[Dict[str, float]]
    n: int

    def table_row(self, name: str, keys: Iterable[str]) -> str:
        return "| " + name + " | " + " | ".join(f"{self.metrics[k]:.3f}" for k in keys) + " |"


METRICS = ("exact_match", "f1", "contains", "faithfulness", "context_recall", "context_precision")


def evaluate_rag(
    examples: Sequence[RAGExample],
    predictions: Sequence[RAGPrediction],
    judge: Optional[Callable[[str, str], bool]] = None,
) -> RAGReport:
    """Puntúa cada predicción contra su ejemplo (emparejados por id) y promedia."""
    by_id = {p.example_id: p for p in predictions}
    rows: List[Dict[str, float]] = []
    for ex in examples:
        pred = by_id.get(ex.id)
        if pred is None:
            raise KeyError(f"Sin predicción para {ex.id}")
        row = {
            "exact_match": exact_match(pred.answer, ex.answers),
            "f1": f1_score(pred.answer, ex.answers),
            "contains": contains_answer(pred.answer, ex.answers),
            "faithfulness": faithfulness(pred.answer, pred.contexts),
            "context_recall": context_recall(pred.retrieved_titles, ex.supporting_titles),
            "context_precision": context_precision(pred.retrieved_titles, ex.supporting_titles),
        }
        if judge is not None:
            row["judged_faithfulness"] = judged_faithfulness(pred.answer, pred.contexts, judge)
        if pred.latency_s is not None:
            row["latency_s"] = pred.latency_s
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    metrics = {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys} if rows else {}
    return RAGReport(metrics=metrics, per_example=rows, n=len(rows))


def regressions(current: Dict[str, float], baseline: Dict[str, float], tolerance: float = 0.02) -> List[str]:
    """Las métricas que han bajado más de ``tolerance`` (absoluta) respecto a la línea base.
    Solo se comparan las que están en la línea base; ``latency_s`` sube cuando empeora."""
    problems = []
    for key, base in baseline.items():
        if key not in current:
            problems.append(f"{key}: falta en la ejecución actual")
            continue
        value = current[key]
        worse = value > base + tolerance if key.endswith("latency_s") else value < base - tolerance
        if worse:
            problems.append(f"{key}: {value:.4f} frente a {base:.4f} (tolerancia {tolerance})")
    return problems
