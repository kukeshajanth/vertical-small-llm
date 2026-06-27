"""Frontier baselines via authenticated CLIs (no raw API keys).

Same job as eval_frontier.py, but instead of the Anthropic / OpenAI SDKs it drives
the CLIs you already log into on a Claude or ChatGPT subscription:

  claude -p --model <id>   -> Claude (Opus / Sonnet / Haiku)
  codex exec -m <id>       -> GPT-5.x  (clean final answer via --output-last-message)

This is how the article's frontier numbers were actually produced: no API keys on the
box, just the subscription CLIs. It scores on the SAME test split as eval_local.py, so
report.py lines everything up next to the local models.

  python cli_frontier.py --dataset ledgar --engine sonnet-4.6 --limit 300 --workers 4
  python cli_frontier.py --dataset ledgar --engine gpt-5.5    --limit 300 --workers 4

Heads up: `claude -p` runs as an agent. On a single-shot label task a reasoning model
can over-think its way into a sibling class (we watched Opus do exactly that and dropped
its number). For the cleanest one-shot call, use eval_frontier.py with thinking disabled,
or stick to a mid-tier / non-reasoning model on the CLI.
"""
import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from data import REGISTRY, build_prompt, examples, label_names, system_prompt
from match import to_label

# engine key -> (transport, model id, display name)
ENGINES = {
    "opus-4.8":   ("claude", "claude-opus-4-8",   "Claude Opus 4.8"),
    "sonnet-4.6": ("claude", "claude-sonnet-4-6", "Claude Sonnet 4.6"),
    "haiku-4.5":  ("claude", "claude-haiku-4-5",  "Claude Haiku 4.5"),
    "gpt-5.5":    ("codex",  "gpt-5.5",           "GPT-5.5"),
    "gpt-5.4":    ("codex",  "gpt-5.4",           "GPT-5.4"),
}

TMP = tempfile.gettempdir()


def run_claude(prompt, model):
    r = subprocess.run(["claude", "-p", "--model", model], input=prompt,
                       capture_output=True, text=True, cwd=TMP, timeout=180)
    return r.stdout.strip()


def run_codex(prompt, model):
    fd, out = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        subprocess.run(["codex", "exec", "--skip-git-repo-check",
                        "-c", 'model_reasoning_effort="low"', "-m", model, "-o", out, prompt],
                       capture_output=True, text=True, cwd=TMP, timeout=240)
        with open(out) as f:
            return f.read().strip()
    finally:
        os.unlink(out)


def classify(transport, model, prompt):
    for attempt in range(2):
        try:
            raw = run_claude(prompt, model) if transport == "claude" else run_codex(prompt, model)
            if raw:
                return raw
        except Exception as e:  # one bad call shouldn't kill the run
            if attempt == 1:
                print("  err:", str(e)[:120], flush=True)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(REGISTRY), default="ledgar")
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    transport, model, disp = ENGINES[a.engine]
    name = a.name or disp
    labels = label_names(a.dataset)
    sysmsg = system_prompt(a.dataset)
    rows = list(examples(a.dataset, "test", n=a.limit))
    # claude -p / codex exec take one prompt, so fold the system message in.
    prompts = [sysmsg + "\n\n" + build_prompt(a.dataset, r["text"]) for r in rows]

    t0 = time.time()
    correct = err = done = 0
    preds = [None] * len(rows)

    def work(i):
        raw = classify(transport, model, prompts[i])
        return i, to_label(raw, labels), rows[i]["label"], raw

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(work, i) for i in range(len(rows))]):
            i, pred, gold, raw = fut.result()
            preds[i] = {"gold": gold, "pred": pred, "raw": raw[:240], "correct": pred == gold}
            done += 1
            err += (not raw)
            correct += (pred == gold)
            if done % 25 == 0:
                print(f"  {done}/{len(rows)}  acc~{correct / done:.3f}  err={err}", flush=True)

    os.makedirs("runs", exist_ok=True)
    with open(f"runs/preds-{a.engine}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    n = len(rows)
    res = {"name": name, "dataset": a.dataset, "kind": "frontier", "n": n,
           "errors": err, "accuracy": round(correct / n, 4),
           "sec_per_req": round((time.time() - t0) / n, 4)}
    json.dump(res, open("runs/" + name.replace("/", "_").replace(":", "_") + ".json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
