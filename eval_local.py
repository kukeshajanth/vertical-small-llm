"""Score a local model on a vertical: base model, or fine-tuned (with --adapter).

Captures accuracy, tokens/sec, seconds/request. Writes runs/<name>.json.
Run twice: once with no --adapter (base), once with --adapter (fine-tuned).
"""
import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import REGISTRY, build_prompt, examples, label_names, system_prompt
from match import to_label


def load(model_id, adapter=None):
    tok = AutoTokenizer.from_pretrained(adapter or model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def _encode(tok, msgs):
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:  # tokenizer has no reasoning toggle
        return tok.apply_chat_template(msgs, **kw)


@torch.no_grad()
def classify(tok, model, name, text, with_labels=True):
    msgs = [
        {"role": "system", "content": system_prompt(name)},
        {"role": "user", "content": build_prompt(name, text, with_labels=with_labels)},
    ]
    enc = _encode(tok, msgs).to(model.device)
    inlen = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    new = out[0][inlen:]
    return tok.decode(new, skip_special_tokens=True), int(new.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(REGISTRY), default="ledgar")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    tok, model = load(args.model, args.adapter)
    labels = label_names(args.dataset)
    with_labels = args.adapter is None  # fine-tuned model knows the classes; base/zero-shot needs the list

    n = correct = toks = 0
    preds = []
    t0 = time.time()
    for ex in examples(args.dataset, "test", n=args.limit):
        gen, nt = classify(tok, model, args.dataset, ex["text"], with_labels=with_labels)
        toks += nt
        pred = to_label(gen, labels)
        preds.append({"gold": ex["label"], "pred": pred, "raw": gen.strip()[:120],
                      "correct": pred == ex["label"], "text": ex["text"][:400]})
        if pred == ex["label"]:
            correct += 1
        n += 1
    dt = time.time() - t0

    res = {
        "name": args.name or (("ft:" if args.adapter else "base:") + args.model),
        "dataset": args.dataset,
        "kind": "local",
        "n": n,
        "accuracy": round(correct / n, 4),
        "tok_per_s": round(toks / dt, 1),
        "sec_per_req": round(dt / n, 4),
    }
    os.makedirs("runs", exist_ok=True)
    fn = "runs/" + res["name"].replace("/", "_").replace(":", "_") + ".json"
    with open(fn, "w") as f:
        json.dump(res, f, indent=2)
    with open("runs/preds-local-" + res["name"].replace("/", "_").replace(":", "_") + ".jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
