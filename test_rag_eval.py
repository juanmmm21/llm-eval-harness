import json
import os
import tempfile
import unittest

from llm_eval_harness.baselines import lexical_retrieval, oracle_retrieval, run_baseline, sentence_overlap_answer
from llm_eval_harness.cli import main as cli_main
from llm_eval_harness.datasets import Passage, RAGExample, hotpotqa_example, load_jsonl, save_jsonl
from llm_eval_harness.rag_metrics import (
    contains_answer,
    context_precision,
    context_recall,
    evaluate_rag,
    exact_match,
    f1_score,
    faithfulness,
    normalize_answer,
    regressions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET = os.path.join(HERE, "data", "hotpotqa_distractor_dev_50.jsonl")


def example() -> RAGExample:
    return RAGExample(
        id="q1",
        question="Which city is the Eiffel Tower in?",
        answers=["Paris"],
        passages=[
            Passage("Eiffel Tower", "The Eiffel Tower is a wrought-iron tower in Paris. It opened in 1889."),
            Passage("Berlin", "Berlin is the capital of Germany."),
            Passage("Bread", "Sourdough is a bread."),
        ],
        supporting_titles=["Eiffel Tower"],
    )


class TestMetrics(unittest.TestCase):
    def test_squad_normalization(self) -> None:
        self.assertEqual(normalize_answer("The  Eiffel-Tower!"), "eiffeltower")
        self.assertEqual(exact_match("the Paris", ["Paris"]), 1.0)
        self.assertEqual(exact_match("Paris, France", ["Paris"]), 0.0)

    def test_f1_partial_and_yes_no(self) -> None:
        self.assertAlmostEqual(f1_score("Paris France", ["Paris"]), 2 * 0.5 * 1.0 / 1.5)
        self.assertEqual(f1_score("yes", ["no"]), 0.0)
        self.assertEqual(f1_score("yes", ["yes"]), 1.0)
        self.assertEqual(f1_score("", ["Paris"]), 0.0)

    def test_contains_respects_word_boundaries(self) -> None:
        self.assertEqual(contains_answer("It is in Paris, France.", ["Paris"]), 1.0)
        self.assertEqual(contains_answer("Parisian food", ["Paris"]), 0.0)

    def test_faithfulness(self) -> None:
        ctx = ["The Eiffel Tower is in Paris."]
        self.assertEqual(faithfulness("Paris", ctx), 1.0)
        self.assertEqual(faithfulness("Madrid", ctx), 0.0)
        self.assertEqual(faithfulness("Eiffel Madrid", ctx), 0.5)
        self.assertEqual(faithfulness("yes", ctx), 1.0)

    def test_context_recall_and_precision(self) -> None:
        self.assertEqual(context_recall(["A", "B"], ["A", "C"]), 0.5)
        self.assertEqual(context_precision(["A", "B"], ["A", "C"]), 0.5)
        self.assertEqual(context_recall([], []), 1.0)
        self.assertEqual(context_precision([], ["A"]), 0.0)

    def test_evaluate_rag_requires_every_prediction(self) -> None:
        with self.assertRaises(KeyError):
            evaluate_rag([example()], [])

    def test_regressions(self) -> None:
        self.assertEqual(regressions({"f1": 0.5}, {"f1": 0.51}), [])
        self.assertEqual(len(regressions({"f1": 0.4}, {"f1": 0.5})), 1)
        self.assertEqual(len(regressions({}, {"f1": 0.5})), 1)
        self.assertEqual(len(regressions({"latency_s": 2.0}, {"latency_s": 1.0})), 1)


class TestDatasets(unittest.TestCase):
    def test_hotpotqa_row_conversion(self) -> None:
        row = {
            "id": "x",
            "question": "q?",
            "answer": "a",
            "type": "bridge",
            "level": "easy",
            "supporting_facts": {"title": ["T1", "T1", "T2"], "sent_id": [0, 1, 0]},
            "context": {"title": ["T1", "T2"], "sentences": [["S1. ", "S2."], ["S3."]]},
        }
        ex = hotpotqa_example(row)
        self.assertEqual(ex.supporting_titles, ["T1", "T2"])
        self.assertEqual(ex.passages[0], Passage("T1", "S1. S2."))
        self.assertEqual(ex.meta, {"type": "bridge", "level": "easy"})

    def test_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.jsonl")
            save_jsonl([example()], path)
            self.assertEqual(load_jsonl(path), [example()])

    def test_committed_subset_is_hotpotqa(self) -> None:
        examples = load_jsonl(SUBSET)
        self.assertEqual(len(examples), 50)
        for ex in examples:
            self.assertGreaterEqual(len(ex.passages), 2)  # casi todas traen 10; alguna, solo las de apoyo
            titles = {p.title for p in ex.passages}
            self.assertTrue(set(ex.supporting_titles) <= titles)


class TestBaselines(unittest.TestCase):
    def test_lexical_and_oracle(self) -> None:
        ex = example()
        self.assertEqual(lexical_retrieval(ex, k=1)[0].title, "Eiffel Tower")
        self.assertEqual([p.title for p in oracle_retrieval(ex)], ["Eiffel Tower"])
        self.assertIn("Paris", sentence_overlap_answer(ex.question, oracle_retrieval(ex)))

    def test_run_baseline_scores(self) -> None:
        report = evaluate_rag([example()], run_baseline([example()], "lexical", k=1))
        self.assertEqual(report.metrics["context_recall"], 1.0)
        self.assertEqual(report.metrics["contains"], 1.0)

    def test_cli_gate_passes_on_committed_baseline(self) -> None:
        gate = os.path.join(HERE, "data", "baseline_hotpotqa_50.json")
        self.assertEqual(cli_main(["baseline", SUBSET, "--gate", gate]), 0)

    def test_cli_gate_fails_on_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = os.path.join(tmp, "gate.json")
            with open(gate, "w") as fh:
                json.dump({"context_recall": 0.99}, fh)
            self.assertEqual(cli_main(["baseline", SUBSET, "--gate", gate]), 1)


if __name__ == "__main__":
    unittest.main()
