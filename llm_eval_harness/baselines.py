"""Líneas base de referencia, sin modelos ni otros componentes, para situar cualquier sistema.

* ``lexical_retrieval``: TF-IDF (coseno) entre la pregunta y los párrafos del propio ejemplo
  (el escenario *distractor* de HotpotQA: 2 párrafos de apoyo entre 10).
* ``oracle_retrieval``: devuelve los párrafos de apoyo; da el techo del recall del contexto.
* ``sentence_overlap_answer``: como "respuesta", la frase recuperada que más palabras comparte
  con la pregunta; una cota inferior honesta de lo que aporta un generador.

Sirven para la CI del propio harness (si una métrica cambia aquí, cambió la métrica) y como
filas de referencia en las tablas.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import List, Sequence, Tuple

from .datasets import Passage, RAGExample
from .rag_metrics import RAGPrediction, content_words

_SENT = re.compile(r"(?<=[.!?])\s+")


def _tf(text: str) -> Counter:
    return Counter(content_words(text))


def lexical_retrieval(example: RAGExample, k: int = 2) -> List[Passage]:
    docs = [_tf(f"{p.title} {p.text}") for p in example.passages]
    n = len(docs)
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

    def vec(tf: Counter) -> dict:
        return {w: c * idf.get(w, 1.0) for w, c in tf.items()}

    q = vec(_tf(example.question))
    qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
    scored: List[Tuple[float, int]] = []
    for i, d in enumerate(docs):
        dv = vec(d)
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        scored.append((sum(q[w] * dv.get(w, 0.0) for w in q) / (qn * dn), i))
    scored.sort(reverse=True)
    return [example.passages[i] for _, i in scored[:k]]


def oracle_retrieval(example: RAGExample) -> List[Passage]:
    gold = set(example.supporting_titles)
    return [p for p in example.passages if p.title in gold]


def sentence_overlap_answer(question: str, passages: Sequence[Passage]) -> str:
    q = set(content_words(question))
    best, best_score = "", -1
    for p in passages:
        for s in _SENT.split(p.text):
            score = len(q & set(content_words(s)))
            if score > best_score:
                best, best_score = s.strip(), score
    return best


def run_baseline(examples: Sequence[RAGExample], retrieval: str = "lexical", k: int = 2) -> List[RAGPrediction]:
    preds = []
    for ex in examples:
        start = time.perf_counter()
        passages = oracle_retrieval(ex) if retrieval == "oracle" else lexical_retrieval(ex, k)
        answer = sentence_overlap_answer(ex.question, passages)
        preds.append(
            RAGPrediction(
                example_id=ex.id,
                answer=answer,
                retrieved_titles=[p.title for p in passages],
                contexts=[p.text for p in passages],
                latency_s=time.perf_counter() - start,
            )
        )
    return preds
