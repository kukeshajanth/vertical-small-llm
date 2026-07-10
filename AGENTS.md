# Agent Runbook

## Goal

Reproduce the LEDGAR specialist experiment without inventing results, exposing credentials, or silently changing the protocol.

## Preflight

Run these checks before installing dependencies:

```bash
uname -a
python3 --version
nvidia-smi
df -h .
git status --short
```

Continue only when all of the following are true:

- The operating system is Linux.
- Python is 3.10 or 3.11.
- `nvidia-smi` reports a working CUDA GPU.
- At least 50GB is free on the target filesystem.
- GPU memory is at least 24GB.

Do not modify or delete unrelated local files. Never print, commit, or copy API keys.

## Select a mode

- `qlora`: use for an NVIDIA GPU with 24GB to 79GB VRAM.
- `bf16`: use only for a DGX Spark or an accelerator with at least 80GB available memory.

If the hardware does not meet either mode, stop. Do not improvise a smaller model and report it as the same experiment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

For QLoRA:

```bash
pip install -r requirements-qlora.txt
```

For bf16:

```bash
pip install -r requirements.txt
```

Validate the harness before downloading models:

```bash
python3 -m unittest discover -v
python3 -m compileall -q .
```

## Run

Use one command and keep its log:

```bash
bash run_hero.sh qlora
```

or:

```bash
bash run_hero.sh bf16
```

Expected local artifacts:

- `adapters/ledgar-smollm3-3b/`
- `hero.log`
- `runs/base SmolLM3-3B.json`
- `runs/fine-tuned SmolLM3-3B (ours).json`
- Matching `runs/preds-local-*.jsonl` files

Filenames are sanitized only for `/` and `:`. Confirm the actual paths with `find runs -maxdepth 1 -type f -print`.

## Optional frontier evaluation

Run frontier baselines only when credentials already exist in the environment or the required CLI is already authenticated.

API keys:

```bash
python3 eval_frontier.py --dataset ledgar --provider anthropic \
  --model claude-sonnet-4-6 --limit 300 --name "Claude Sonnet 4.6"

python3 eval_frontier.py --dataset ledgar --provider openai \
  --model gpt-5.5 --limit 300 --name "GPT-5.5"
```

Authenticated CLIs:

```bash
python3 cli_frontier.py --dataset ledgar --engine sonnet-4.6 --limit 300 --workers 4
python3 cli_frontier.py --dataset ledgar --engine gpt-5.5 --limit 300 --workers 4
```

If a model ID is unavailable, stop that baseline and report the exact provider error. Do not substitute another model under the same display name.

## Verify

```bash
python3 report.py
python3 -m unittest discover -v
git status --short
```

Inspect every summary under `runs/*.json` and report:

- Hardware and selected mode
- Exact commands
- Model IDs
- Sample count `n`
- Error count
- Accuracy
- Training duration from `hero.log`
- Paths to summaries, predictions, adapter, and charts

Do not repeat the README's 81.7% value unless the new summary contains that value. Do not describe a 300-example smoke run as a full benchmark. Do not hide provider errors.

## Stop conditions

Stop and explain the blocker when CUDA is unavailable, VRAM or disk is insufficient, dependency installation fails, model download is blocked, a provider is unauthenticated, a model ID is unavailable, or tests fail. Preserve partial logs and generated summaries for diagnosis.
