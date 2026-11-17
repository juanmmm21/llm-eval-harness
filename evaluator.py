"""Compatibilidad: el código vive en el paquete `llm_eval_harness`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `evaluator`
directamente."""

from llm_eval_harness.evaluator import *  # noqa: F401,F403
from llm_eval_harness.evaluator import LLMEvaluator, EvalHarnessSuite, MetricReport, DatasetEvaluationSummary  # noqa: F401
