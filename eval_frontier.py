"""Score a current frontier model on the SAME test split, zero-shot.

Anchors (today's frontier, not 2024) — verify exact ids on your account:
  Anthropic  claude-opus-4-8   ($5 / $25 per 1M)
  Anthropic  claude-sonnet-4-6 ($3 / $15 per 1M)   <- mid-tier cost point
  Anthropic  claude-haiku-4-5  ($1 / $5 per 1M)    <- frontier-cheap tier
  OpenAI     gpt-5.5 / gpt-5.4
  Google     gemini-3-pro

Thinking is disabled on the Claude side: this is fast classification, not reasoning.
Writes runs/<name>.json so report.py can line everything up.
"""
import argparse
import json
import os
import time

from data import REGISTRY, build_prompt, examples, label_names, system_prompt
from match import to_label


def anthropic_clf(model, name):
    import anthropic  # pip install anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    sysmsg = system_prompt(name)

    def f(text):
        r = client.messages.create(
            model=model,
            max_tokens=30,
            system=sysmsg,
            thinking={"type": "disabled"},  # classification, not reasoning
            messages=[{"role": "user", "content": build_prompt(name, text)}],
        )
        return next((b.text for b in r.content if b.type == "text"), "")

    return f


def openai_clf(model, name):
    from openai import OpenAI  # pip install openai
    client = OpenAI()  # reads OPENAI_API_KEY
    sysmsg = system_prompt(name)

    def f(text):
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": build_prompt(name, text)},
            ],
            # Reasoning models spend tokens before answering; raise if outputs come back empty.
            max_completion_tokens=512,
        )
        return r.choices[0].message.content or ""

    return f


PROVIDERS = {"anthropic": anthropic_clf, "openai": openai_clf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(REGISTRY), default="ledgar")
    ap.add_argument("--provider", choices=PROVIDERS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    clf = PROVIDERS[args.provider](args.model, args.dataset)
    labels = label_names(args.dataset)

    n = correct = errors = 0
    t0 = time.time()
    for ex in examples(args.dataset, "test", n=args.limit):
        try:
            gen = clf(ex["text"])
        except Exception as e:  # one bad call shouldn't kill the run
            gen, errors = "", errors + 1
            print("err:", e)
        if to_label(gen, labels) == ex["label"]:
            correct += 1
        n += 1
    dt = time.time() - t0

    res = {
        "name": args.name or f"{args.provider}:{args.model}",
        "dataset": args.dataset,
        "kind": "frontier",
        "n": n,
        "errors": errors,
        "accuracy": round(correct / n, 4),
        "sec_per_req": round(dt / n, 4),
    }
    os.makedirs("runs", exist_ok=True)
    fn = "runs/" + res["name"].replace("/", "_").replace(":", "_") + ".json"
    with open(fn, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
