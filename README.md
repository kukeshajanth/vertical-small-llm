# Vertical Small LLM

Fine-tune a genuine 3B model on one narrow task and put it head-to-head with today's frontier models on the **same** test set. The whole thing runs on a single 24GB GPU or a free Colab notebook.

![Fine-tuned 3B vs the frontier on LEDGAR](assets/accuracy.png)

That chart is the result: a fine-tuned [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) classifying legal contract clauses (100 classes) at **81.7%**, matching and edging GPT-5.5 and Claude Sonnet 4.6 (both ~77%) on the same 300-example split, while running locally for a fraction of the cost. The untuned base model gets 39.3%, so the fine-tune is doing the work, not the model.

It is a narrow win on purpose. Point any of these specialists at open-ended reasoning and a frontier model walks all over it. The thesis is *sharper, not smarter*: on one bounded job, a small model you own can match the giants far cheaper.

---

## Two verticals, same code

- **LEDGAR** (default) — legal contract-clause classification, 100 classes. The headline result.
- **Banking77** — banking-support intent classification, 77 classes. Proves the recipe travels to a second domain with zero code changes.

Swap the base model with one flag: `--model Qwen/Qwen3-4B-Instruct-2507`, `--model meta-llama/Llama-3.2-3B-Instruct`, whatever you like. SmolLM3-3B is the default because it is fully open and ungated (no license gate, no auth token to download).

---

## Quickstart

```bash
git clone https://github.com/kukeshajanth/vertical-small-llm.git
cd vertical-small-llm
pip install -r requirements.txt

# reproduce the hero run: base eval -> fine-tune -> fine-tuned eval (LEDGAR)
bash run_hero.sh
```

On a 24GB consumer GPU, drop `--no-4bit` from `run_hero.sh` to train in 4-bit (QLoRA). On a big-unified-memory box (a DGX Spark, an M-series Mac), keep `--no-4bit` for a clean bf16 LoRA.

To add the frontier baselines, drop your keys in `.env` first:

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY / OPENAI_API_KEY
```

---

## The experiment, step by step

```bash
# 1. Baseline: the base 3B, no training
python eval_local.py --dataset ledgar --limit 300 --name "base SmolLM3-3B"

# 2. Fine-tune on LEDGAR (prints train_seconds)
python train_lora.py --dataset ledgar --no-4bit --train-size 4000 --epochs 2 \
       --out adapters/ledgar-smollm3-3b

# 3. Score the fine-tuned 3B
python eval_local.py --dataset ledgar --adapter adapters/ledgar-smollm3-3b \
       --limit 300 --name "fine-tuned 3B (ours)"

# 4. The frontier anchors, same 300 examples
python eval_frontier.py --dataset ledgar --provider anthropic --model claude-sonnet-4-6 --limit 300 --name "Claude Sonnet 4.6"
python eval_frontier.py --dataset ledgar --provider openai    --model gpt-5.5           --limit 300 --name "GPT-5.5"

# 5. Table + bar chart
python report.py
```

Run the whole thing again with `--dataset banking77` to show the recipe travels.

**What to look for:** the fine-tuned 3B, on hardware you own, matches or edges the frontier models on this narrow task, at a fraction of the latency and cost.

---

## What's in the box

| File | Job |
|---|---|
| `data.py` | Loads either vertical, builds the prompt. Frontier models get the full label list; the fine-tuned model learns the labels from data. |
| `train_lora.py` | Fine-tunes the 3B with (Q)LoRA. Prints `train_seconds`. |
| `eval_local.py` | Scores a local model (base or fine-tuned): accuracy, tokens/sec, sec/request. Dumps predictions to `runs/`. |
| `eval_frontier.py` | Scores a frontier model (Claude / GPT) on the same split via the API, prompted fairly. |
| `cli_frontier.py` | Same, but through the authenticated `claude` / `codex` CLIs (no API keys). The path the article's numbers actually used. |
| `match.py` | Maps a free-text answer back to the nearest real label, the same way for every model. |
| `report.py` | Lines up every run into one table and a bar chart. |
| `gen_charts.py` | Regenerates the figures in `assets/` from a completed run (optional). |
| `run_hero.sh` | One command for the full LEDGAR reproduction. |

---

## How the comparison is scored (read this before you cite a number)

This is a **fine-tuned specialist vs zero-shot generalists**, which is the real production choice, not a like-for-like model-quality test. The fine-tuned 3B trained on 4,000 labeled clauses and learned this dataset's exact conventions. The frontier models saw the 100 labels once, in the prompt, with zero examples. Give the frontier models a few in-context examples and they would close most of the gap. The point isn't that a 3B out-thinks GPT-5.5; it's that on a narrow, repeated task a model you own gets *this close, this cheap*.

Other honest notes:

- **It's a narrow win.** The 3B loses to the frontier on open-ended reasoning. That narrowness is the thesis.
- **300 test examples is a small sample.** Treat a few-point gap as "matches," not "crushes." Bump `--limit` for tighter numbers.
- **Same prompt, same scoring for everyone.** Frontier models get the full label list; every answer is matched to the nearest real label by `match.py`.
- **Frontier model IDs drift.** The ones above were current in mid-2026; verify the exact IDs on your account.
- **Banking77 already has a published sub-1B win** (Oumi, 2026), so it's here as a sanity-check second vertical, not a novel claim. LEDGAR is the marquee.
- `bitsandbytes` on a brand-new arch (Blackwell / ARM) can be finicky. `--no-4bit` is the escape hatch.
- SmolLM3 has a reasoning toggle; the scripts call the chat template with `enable_thinking=False` so it runs as a plain classifier.
- TRL moves fast. If `SFTConfig` rejects `max_length` / `processing_class`, your version wants `max_seq_length` / `tokenizer`. One-line rename.

---

## Frontier baselines: API key or subscription CLI

You can score the frontier models two ways. Both hit the same test split and write the same `runs/` files, so `report.py` lines them up either way.

- **API key** (`eval_frontier.py`): bring an `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Calls are one-shot with thinking disabled, the cleanest measurement.
- **Subscription CLI** (`cli_frontier.py`): no keys. It drives the `claude` and `codex` CLIs you already log into, which is exactly how the numbers in the writeup were produced (the box had no API keys).

```bash
# API path
python eval_frontier.py --dataset ledgar --provider anthropic --model claude-sonnet-4-6 --limit 300

# CLI path (Claude / ChatGPT subscription, no keys)
python cli_frontier.py  --dataset ledgar --engine sonnet-4.6 --limit 300 --workers 4
python cli_frontier.py  --dataset ledgar --engine gpt-5.5    --limit 300 --workers 4
```

One honest caveat from doing it the CLI way: `claude -p` runs the model as an *agent*, and on a single-shot label a reasoning model can talk itself into a sibling category. That is a property of the harness, not the model. We saw it on Opus and left Opus out. For the cleanest single-shot number, prefer the API path or a mid-tier model.

## Make it yours

Add a vertical by dropping one entry into the `REGISTRY` in `data.py` (Hugging Face dataset id, text field, label field, a one-line domain description). Everything else is unchanged.

---

## License

MIT. See [LICENSE](LICENSE).
