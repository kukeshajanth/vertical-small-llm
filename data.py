"""Two verticals, one code path. Each is a clean text -> single-label task.

HERO   = ledgar    (legal contract-clause classification, 100 classes) — the marquee
SECOND = banking77 (banking-support intent classification, 77 classes) — the "it travels" proof

Same loader, same prompt builder, same metric (top-1 accuracy). Swap with --dataset.
"""
from functools import lru_cache

from datasets import load_dataset

REGISTRY = {
    "ledgar": {
        "hf": ("coastalcph/lex_glue", "ledgar"),
        "test_split": "test",
        "train_split": "train",
        "text": "text", "label": "label",
        "domain": "commercial contracts",
        "unit": "contract provision",
        "what": "clause type",
    },
    "banking77": {
        "hf": ("PolyAI/banking77",),
        "test_split": "test",
        "train_split": "train",
        "text": "text", "label": "label",
        "domain": "a digital bank's customer support",
        "unit": "customer message",
        "what": "intent",
    },
}


@lru_cache(maxsize=None)
def _ds(name):
    return load_dataset(*REGISTRY[name]["hf"])


@lru_cache(maxsize=None)
def label_names(name):
    cfg = REGISTRY[name]
    return list(_ds(name)[cfg["test_split"]].features[cfg["label"]].names)


def system_prompt(name):
    cfg = REGISTRY[name]
    return (
        f"You are a {cfg['what']} classifier for {cfg['domain']}. "
        f"Given a {cfg['unit']}, choose the single best {cfg['what']} from the allowed list. "
        f"Respond with ONLY the label, exactly as written in the list, and nothing else. No explanation."
    )


@lru_cache(maxsize=None)
def _label_block(name):
    return "\n".join(f"- {n}" for n in label_names(name))


def build_prompt(name, text, with_labels=True):
    """with_labels lists all classes (zero-shot needs the cheat sheet); the
    fine-tuned model learns them, so it runs without the list (shorter, faster)."""
    cfg = REGISTRY[name]
    head = f"Allowed {cfg['what']} labels:\n{_label_block(name)}\n\n" if with_labels else ""
    return (
        head
        + f'{cfg["unit"].capitalize()}: "{text}"\n'
        + f"{cfg['what'].capitalize()}:"
    )


def examples(name, split, n=None, seed=0):
    """split is 'train' or 'test'. Yields {'text', 'label'} dicts."""
    cfg = REGISTRY[name]
    ds = _ds(name)[cfg[f"{split}_split"]]
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    names = label_names(name)
    tf, lf = cfg["text"], cfg["label"]
    for row in ds:
        yield {"text": row[tf], "label": names[row[lf]]}
