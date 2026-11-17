"""LLM evaluation harness: lexical and semantic metrics over datasets."""

from .evaluator import LLMEvaluator, EvalHarnessSuite, MetricReport, DatasetEvaluationSummary

__version__ = "0.1.0"

__all__ = ['LLMEvaluator', 'EvalHarnessSuite', 'MetricReport', 'DatasetEvaluationSummary']
