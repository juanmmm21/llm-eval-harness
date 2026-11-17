"""LLM evaluation harness: lexical and semantic metrics over datasets."""

from .datasets import Passage, RAGExample, load_hotpotqa, load_jsonl, load_nq_open, save_jsonl
from .rag_metrics import RAGPrediction, RAGReport, evaluate_rag, regressions
from .evaluator import LLMEvaluator, EvalHarnessSuite, MetricReport, DatasetEvaluationSummary

__version__ = "0.1.0"

__all__ = ['Passage', 'RAGExample', 'RAGPrediction', 'RAGReport', 'evaluate_rag', 'regressions', 'load_hotpotqa', 'load_nq_open', 'load_jsonl', 'save_jsonl', 'LLMEvaluator', 'EvalHarnessSuite', 'MetricReport', 'DatasetEvaluationSummary']
