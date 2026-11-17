"""Classification metrics: accuracy, macro-F1, per-class precision/recall, confusion, latency.

``evaluate_classifier(predict, texts, labels, contexts)`` times each prediction, so a router's
scorecard includes what it costs per message, not only how often it is right.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

Predict = Callable[[str, Any], str]


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class ClassificationReport:
    n: int
    accuracy: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion: Dict[str, Dict[str, int]]
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classification_report(
    gold: Sequence[str], predicted: Sequence[str], latencies_ms: Optional[Sequence[float]] = None
) -> ClassificationReport:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted differ in length")
    classes = sorted(set(gold) | set(predicted))
    confusion = {g: {p: 0 for p in classes} for g in classes}
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1
    per_class: Dict[str, Dict[str, float]] = {}
    f1s = []
    for c in sorted(set(gold)):
        tp = confusion[c][c]
        fp = sum(confusion[g][c] for g in classes if g != c)
        fn = sum(confusion[c][p] for p in classes if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": float(tp + fn)}
        f1s.append(f1)
    lat = list(latencies_ms or [])
    return ClassificationReport(
        n=len(gold),
        accuracy=sum(g == p for g, p in zip(gold, predicted)) / max(1, len(gold)),
        macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
        per_class=per_class,
        confusion=confusion,
        latency_ms_p50=_pct(lat, 0.5),
        latency_ms_p95=_pct(lat, 0.95),
    )


def evaluate_classifier(
    predict: Predict,
    texts: Sequence[str],
    labels: Sequence[str],
    contexts: Optional[Sequence[Any]] = None,
    keep_errors: int = 20,
) -> ClassificationReport:
    """Run ``predict(text, context)`` over the set, timing each call."""
    contexts = list(contexts) if contexts is not None else [None] * len(texts)
    preds, times = [], []
    for text, ctx in zip(texts, contexts):
        start = time.perf_counter()
        preds.append(predict(text, ctx))
        times.append((time.perf_counter() - start) * 1000)
    report = classification_report(labels, preds, times)
    report.errors = [
        {"text": t, "context": c, "gold": g, "predicted": p}
        for t, c, g, p in zip(texts, contexts, labels, preds)
        if g != p
    ][:keep_errors]
    return report
