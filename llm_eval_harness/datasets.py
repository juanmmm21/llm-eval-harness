"""Cargadores de datasets públicos para evaluar RAG: HotpotQA y Natural Questions (open).

Sin dependencias: las filas se piden a la API de filas de Hugging Face
(``datasets-server.huggingface.co/rows``) con la biblioteca estándar, por páginas de 100, y se
guardan en caché como JSONL. Los mismos ficheros JSONL sirven de subconjunto fijo en el
repositorio (``data/``), para que la CI no dependa de la red.

* **HotpotQA** (``hotpotqa/hotpot_qa``, configuración ``distractor``; CC BY-SA 4.0): cada pregunta
  trae 10 párrafos de Wikipedia, dos de ellos con los hechos que la responden
  (``supporting_titles``), así que sirve para medir recuperación (recall del contexto) y
  respuesta.
* **Natural Questions open** (``google-research-datasets/nq_open``; CC BY-SA 3.0): preguntas reales
  de Google con respuestas cortas, sin contexto: mide exactitud de respuesta con la recuperación
  que ponga quien evalúa.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


@dataclass(frozen=True)
class Passage:
    """Un párrafo del contexto: título del artículo y texto."""

    title: str
    text: str


@dataclass(frozen=True)
class RAGExample:
    """Una pregunta con sus respuestas válidas, su contexto (si lo trae) y los títulos de los
    párrafos que la apoyan (para el recall del contexto)."""

    id: str
    question: str
    answers: List[str]
    passages: List[Passage] = field(default_factory=list)
    supporting_titles: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RAGExample":
        return cls(
            id=str(d["id"]),
            question=d["question"],
            answers=list(d["answers"]),
            passages=[Passage(**p) for p in d.get("passages", [])],
            supporting_titles=list(d.get("supporting_titles", [])),
            meta=dict(d.get("meta", {})),
        )


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int, retries: int = 3) -> List[Dict[str, Any]]:
    """Filas ``[offset, offset+length)`` de un dataset de Hugging Face, por páginas de 100."""
    rows: List[Dict[str, Any]] = []
    while length > 0:
        n = min(PAGE, length)
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": n}
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as resp:
                    payload = json.loads(resp.read())
                break
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        page = [r["row"] for r in payload.get("rows", [])]
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        length -= len(page)
    return rows


def hotpotqa_example(row: Dict[str, Any]) -> RAGExample:
    ctx = row["context"]
    passages = [Passage(t, "".join(s)) for t, s in zip(ctx["title"], ctx["sentences"])]
    supporting = list(dict.fromkeys(row["supporting_facts"]["title"]))
    return RAGExample(
        id=row["id"],
        question=row["question"],
        answers=[row["answer"]],
        passages=passages,
        supporting_titles=supporting,
        meta={"type": row.get("type"), "level": row.get("level")},
    )


def nq_open_example(row: Dict[str, Any], index: int) -> RAGExample:
    return RAGExample(id=f"nq-{index}", question=row["question"], answers=list(row["answer"]))


def _cache_path(cache_dir: Optional[str], name: str) -> Optional[str]:
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def load_hotpotqa(limit: int = 100, offset: int = 0, split: str = "validation", cache_dir: Optional[str] = None) -> List[RAGExample]:
    """HotpotQA (distractor). ``limit`` preguntas desde ``offset``, en el orden del dataset."""
    path = _cache_path(cache_dir, f"hotpotqa_distractor_{split}_{offset}_{limit}.jsonl")
    if path and os.path.exists(path):
        return load_jsonl(path)
    rows = fetch_rows("hotpotqa/hotpot_qa", "distractor", split, offset, limit)
    examples = [hotpotqa_example(r) for r in rows]
    if path:
        save_jsonl(examples, path)
    return examples


def load_nq_open(limit: int = 100, offset: int = 0, split: str = "validation", cache_dir: Optional[str] = None) -> List[RAGExample]:
    """Natural Questions open: pregunta y respuestas cortas válidas, sin contexto."""
    path = _cache_path(cache_dir, f"nq_open_{split}_{offset}_{limit}.jsonl")
    if path and os.path.exists(path):
        return load_jsonl(path)
    rows = fetch_rows("google-research-datasets/nq_open", "nq_open", split, offset, limit)
    examples = [nq_open_example(r, offset + i) for i, r in enumerate(rows)]
    if path:
        save_jsonl(examples, path)
    return examples


def save_jsonl(examples: Iterable[RAGExample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")


def load_jsonl(path: str) -> List[RAGExample]:
    with open(path, encoding="utf-8") as fh:
        return [RAGExample.from_dict(json.loads(line)) for line in fh if line.strip()]
