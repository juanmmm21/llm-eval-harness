# Evaluation subsets

Small, fixed subsets so the evaluation runs in CI without network access. Regenerate them with
`python -m llm_eval_harness.cli fetch ...` (see the main README).

| File | Source | License | Rows |
| --- | --- | --- | --- |
| `hotpotqa_distractor_dev_50.jsonl` | [HotpotQA](https://hotpotqa.github.io/) (`hotpotqa/hotpot_qa`, config `distractor`, split `validation`, rows 0–49) — Yang et al., EMNLP 2018 | CC BY-SA 4.0 | 50 |
| `nq_open_dev_50.jsonl` | [Natural Questions open](https://github.com/google-research-datasets/natural-questions) (`google-research-datasets/nq_open`, split `validation`, rows 0–49) — Kwiatkowski et al., TACL 2019; Lee et al., ACL 2019 | CC BY-SA 3.0 | 50 |
| `baseline_hotpotqa_50.json` | metrics of the lexical TF-IDF baseline on the subset above (the CI gate) | — | — |

The JSONL files are redistributed under their original licenses (share-alike): only the format
was changed (one `RAGExample` per line: question, answers, passages, supporting titles).
