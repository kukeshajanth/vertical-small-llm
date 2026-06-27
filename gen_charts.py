"""Generate the article's image assets from the real run (runs in the container)."""
import os, re
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, GRAY, BLUE = "#0C1752", "#D56E48", "#9b9892", "#3266ad"
os.makedirs("img", exist_ok=True)
plt.rcParams.update({"font.size": 11})


def no_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# 1) training loss curve (parse hero.log)
losses, epochs = [], []
for line in open("hero.log", errors="ignore"):
    if "'loss'" in line and "'epoch'" in line:
        ml = re.search(r"'loss':\s*'?([\d.]+)", line)
        me = re.search(r"'epoch':\s*'?([\d.]+)", line)
        if ml and me:
            losses.append(float(ml.group(1)))
            epochs.append(float(me.group(1)))
if losses:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color=ORANGE, lw=2.2, marker="o", ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Fine-tuning SmolLM3-3B on LEDGAR (LoRA, DGX Spark, ~74 min)", color=NAVY, fontsize=12)
    ax.grid(True, alpha=.25)
    no_spines(ax)
    fig.tight_layout(); fig.savefig("img/training_loss.png", dpi=150); plt.close()
    print("training_loss.png", len(losses), "points")

# 2) data distribution (long tail)
from datasets import load_dataset
d = load_dataset("coastalcph/lex_glue", "ledgar")
names = d["train"].features["label"].names
cnt = Counter(d["train"]["label"])
counts = sorted([cnt[i] for i in range(len(names))], reverse=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(len(counts)), counts, color=ORANGE, width=1.0)
ax.set_xlabel("clause type (100 classes, sorted by frequency)")
ax.set_ylabel("# provisions in train")
ax.set_title("LEDGAR is long-tailed: a few common clauses, a long tail of rare ones", color=NAVY, fontsize=11.5)
ax.annotate(f"most common: {counts[0]:,}", xy=(2, counts[0]), xytext=(14, counts[0] * 0.88), fontsize=10, color=NAVY)
ax.annotate(f"rarest: {counts[-1]}", xy=(99, counts[-1]), xytext=(55, max(counts) * 0.22),
            fontsize=10, color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY))
no_spines(ax)
fig.tight_layout(); fig.savefig("img/data_distribution.png", dpi=150); plt.close()
print("data_distribution.png")

# 3) accuracy
rows = [("Base SmolLM3-3B", 39.3, GRAY), ("GPT-5.5", 77.2, BLUE),
        ("Claude Sonnet 4.6", 77.0, BLUE), ("Fine-tuned SmolLM3-3B (ours)", 81.7, ORANGE)]
rows = sorted(rows, key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows])
for i, r in enumerate(rows):
    ax.text(r[1] + 1, i, f"{r[1]:.1f}%", va="center", fontsize=11, color=NAVY)
ax.axvline(70, ls=":", color="#999"); ax.axvline(88, ls=":", color="#999")
ax.text(70, len(rows) - 0.35, "GPT-3.5 (pub.) 70", fontsize=8.5, color="#999", ha="center")
ax.text(88, len(rows) - 0.35, "Legal-BERT 88", fontsize=8.5, color="#999", ha="center")
ax.set_xlim(0, 100)
ax.set_xlabel("LEDGAR accuracy (%), same 300-example test set")
ax.set_title("A fine-tuned 3B vs the frontier (legal clause classification)", color=NAVY, fontsize=12)
no_spines(ax)
fig.tight_layout(); fig.savefig("img/accuracy.png", dpi=150); plt.close()
print("accuracy.png")

# 4) the settings experiment (prompt design -> 2x throughput)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
bars = ax.bar(["with 100-label\nlist in prompt", "no label list\n(model learned them)"],
              [0.89, 1.84], color=[GRAY, ORANGE], width=.6)
for b, v in zip(bars, [0.89, 1.84]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v}/s", ha="center", fontsize=12, color=NAVY)
ax.set_ylabel("training throughput (samples/sec)")
ax.set_title("One prompt-design choice ~doubled training speed", color=NAVY, fontsize=11.5)
no_spines(ax)
fig.tight_layout(); fig.savefig("img/throughput.png", dpi=150); plt.close()
print("throughput.png")
print("ALL CHARTS DONE")
