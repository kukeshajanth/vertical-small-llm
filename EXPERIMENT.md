# Experiment Record

## Question

Can a 3B model fine-tuned for one narrow label convention match zero-shot frontier models on that same bounded job?

## Recorded smoke run

- Dataset: LEDGAR from LexGLUE
- Task: 100-class legal clause classification
- Base model: `HuggingFaceTB/SmolLM3-3B`
- Training sample: 4,000 examples from the official training split
- Training: LoRA, 2 epochs, maximum sequence length 1,536
- Test sample: 300 examples from the official test split
- Sampling seed: 0
- Fine-tuned prompt: no label list
- Base and frontier prompt: complete label list
- Metric: top-1 accuracy after mapping model text to a valid dataset label

Recorded results:

| Model | Accuracy |
|---|---:|
| Fine-tuned SmolLM3-3B | 81.7% |
| Claude Sonnet 4.6 | 77.0% |
| GPT-5.5 | 76.7% |
| Base SmolLM3-3B | 39.3% |

## Artifact status

The repository contains the code and the tracked chart from the original experiment. It does not contain the trained adapter, provider responses, original run summaries, or original prediction JSONL files. Those generated artifacts are intentionally ignored because they are large, may contain source text, and can include provider error details.

A later code audit found two permissive paths in the original label matcher: empty output could match the first label through substring logic, and arbitrary text could be forced to the nearest label with a similarity cutoff of zero. The current matcher fixes both paths and includes regression tests. The original prediction files are unavailable, so the historical 81.7%, 77.0%, and 76.7% results cannot be rescored against the hardened matcher.

That means the recorded table is a historical smoke-run record, not a fully self-verifying benchmark package. A fresh run should produce its own summaries and predictions under `runs/`. Report those fresh numbers, including `n` and `errors`, instead of assuming the recorded values will repeat exactly.

## Interpretation boundary

This comparison answers a production-shaped question: specialist trained on examples versus generalist prompted zero-shot. It does not establish that SmolLM3-3B is broadly more capable than either frontier model.

The 300-example sample also leaves meaningful statistical uncertainty around a few-point gap. Treat 81.7% versus roughly 77% as comparable performance with a small observed lead. Increase the sample size and preserve the same protocol before making a stronger claim.

## Reproduction contract

A comparable reproduction must keep these fixed:

- Dataset and official split
- Seed
- Test sample size
- Training sample size
- Prompt format
- Label-matching code
- Model IDs
- Generation settings

Save and inspect every run summary and prediction file. Empty or failed responses must count as incorrect. Report provider errors. Never merge runs with different datasets into one accuracy chart.
