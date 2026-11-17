"""``python -m llm_eval_harness.cli``: descargar subconjuntos y medir las líneas base.

    python -m llm_eval_harness.cli fetch hotpotqa --limit 50 --out data/hotpotqa_distractor_dev_50.jsonl
    python -m llm_eval_harness.cli baseline data/hotpotqa_distractor_dev_50.jsonl --gate data/baseline_hotpotqa_50.json

``baseline --gate`` termina con código 1 si alguna métrica baja de la línea base guardada: es lo
que ejecuta la CI.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import List, Optional

from . import datasets
from .baselines import run_baseline
from .rag_metrics import evaluate_rag, regressions

KEYS = ("exact_match", "f1", "contains", "faithfulness", "context_recall", "context_precision")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_eval_harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch", help="descarga un subconjunto a JSONL")
    fetch.add_argument("dataset", choices=["hotpotqa", "nq_open"])
    fetch.add_argument("--limit", type=int, default=50)
    fetch.add_argument("--offset", type=int, default=0)
    fetch.add_argument("--out", required=True)
    base = sub.add_parser("baseline", help="mide las líneas base sobre un JSONL de HotpotQA")
    base.add_argument("path")
    base.add_argument("--k", type=int, default=2)
    base.add_argument("--gate", help="JSON con la línea base: falla si alguna métrica baja")
    base.add_argument("--tolerance", type=float, default=0.02)
    base.add_argument("--write", help="guarda las métricas léxicas como nueva línea base")
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        loader = datasets.load_hotpotqa if args.dataset == "hotpotqa" else datasets.load_nq_open
        examples = loader(limit=args.limit, offset=args.offset)
        datasets.save_jsonl(examples, args.out)
        print(f"{len(examples)} ejemplos -> {args.out}")
        return 0

    examples = datasets.load_jsonl(args.path)
    print(f"{len(examples)} preguntas · Python {platform.python_version()} · {platform.machine()}")
    print("| Sistema | " + " | ".join(KEYS) + " |")
    print("|---" * (len(KEYS) + 1) + "|")
    reports = {}
    for name, retrieval in (("oracle context", "oracle"), (f"lexical TF-IDF top-{args.k}", "lexical")):
        report = evaluate_rag(examples, run_baseline(examples, retrieval, args.k))
        reports[retrieval] = report
        print(report.table_row(name, KEYS))
    lexical = {k: round(reports["lexical"].metrics[k], 4) for k in KEYS}
    if args.write:
        with open(args.write, "w", encoding="utf-8") as fh:
            json.dump(lexical, fh, indent=2)
    if args.gate:
        with open(args.gate, encoding="utf-8") as fh:
            problems = regressions(lexical, json.load(fh), args.tolerance)
        if problems:
            print("REGRESIÓN:\n  " + "\n  ".join(problems))
            return 1
        print("Sin regresiones frente a", args.gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
