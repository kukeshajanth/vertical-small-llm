"""Map a model's free-text answer back to a dataset label.

Frontier models won't always echo the label byte-for-byte, so we normalize
and fall back to closest-match. Keeps the accuracy number fair instead of
punishing a right answer for a stray space or capital.
"""
import difflib
import re


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def to_label(pred, labels):
    p = _norm(pred)
    norm = {_norm(l): l for l in labels}
    if not p:
        return None
    if p in norm:
        return norm[p]
    # Accept a valid label wrapped in a sentence.
    for nl, l in norm.items():
        if nl and nl in p:
            return l
    # Fail closed on unrelated output. A cutoff of 0.0 silently turns any
    # garbage response into a valid prediction and can inflate accuracy.
    m = difflib.get_close_matches(p, list(norm.keys()), n=1, cutoff=0.75)
    return norm[m[0]] if m else None
