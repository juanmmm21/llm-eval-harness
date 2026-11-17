import unittest

from llm_eval_harness.classification import classification_report, evaluate_classifier


class TestClassification(unittest.TestCase):
    def test_report(self) -> None:
        r = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
        self.assertEqual(r.accuracy, 0.75)
        self.assertAlmostEqual(r.per_class["a"]["recall"], 0.5)
        self.assertAlmostEqual(r.per_class["b"]["precision"], 2 / 3)
        self.assertAlmostEqual(r.macro_f1, (2 / 3 + 0.8) / 2)
        self.assertEqual(r.confusion["a"]["b"], 1)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            classification_report(["a"], [])

    def test_evaluate_times_and_keeps_errors(self) -> None:
        r = evaluate_classifier(lambda t, c: "x" if c else "y", ["u", "v"], ["x", "x"], [["ctx"], None])
        self.assertEqual(r.accuracy, 0.5)
        self.assertEqual(r.errors, [{"text": "v", "context": None, "gold": "x", "predicted": "y"}])
        self.assertGreaterEqual(r.latency_ms_p95, r.latency_ms_p50)


if __name__ == "__main__":
    unittest.main()
