"""Regenerate article assets from local experiment artifacts."""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NAVY = "#0C1752"
ORANGE = "#D56E48"
GRAY = "#9B9892"
BLUE = "#3266AD"
OUT = Path("assets")
RUNS = Path("runs")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11})


def no_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_training_loss():
    log = Path("hero.log")
    if not log.exists():
        print("skip training_loss.png: hero.log not found")
        return

    losses, epochs = [], []
    for line in log.read_text(errors="ignore").splitlines():
        if "'loss'" not in line or "'epoch'" not in line:
            continue
        loss_match = re.search(r"'loss':\s*'?([\d.]+)", line)
        epoch_match = re.search(r"'epoch':\s*'?([\d.]+)", line)
        if loss_match and epoch_match:
            losses.append(float(loss_match.group(1)))
            epochs.append(float(epoch_match.group(1)))

    if not losses:
        print("skip training_loss.png: no loss records in hero.log")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color=ORANGE, lw=2.2, marker="o", ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Fine-tuning SmolLM3-3B on LEDGAR", color=NAVY, fontsize=12)
    ax.grid(True, alpha=0.25)
    no_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "training_loss.png", dpi=150)
    plt.close(fig)
    print("wrote assets/training_loss.png")


def write_data_distribution():
    from datasets import load_dataset

    dataset = load_dataset("coastalcph/lex_glue", "ledgar")
    names = dataset["train"].features["label"].names
    counts_by_label = Counter(dataset["train"]["label"])
    counts = sorted((counts_by_label[i] for i in range(len(names))), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(counts)), counts, color=ORANGE, width=1.0)
    ax.set_xlabel("clause type (100 classes, sorted by frequency)")
    ax.set_ylabel("provisions in train")
    ax.set_title("LEDGAR is long-tailed", color=NAVY, fontsize=12)
    ax.annotate(
        f"most common: {counts[0]:,}",
        xy=(2, counts[0]),
        xytext=(14, counts[0] * 0.88),
        fontsize=10,
        color=NAVY,
    )
    ax.annotate(
        f"rarest: {counts[-1]}",
        xy=(99, counts[-1]),
        xytext=(55, max(counts) * 0.22),
        fontsize=10,
        color=GRAY,
        arrowprops={"arrowstyle": "->", "color": GRAY},
    )
    no_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "data_distribution.png", dpi=150)
    plt.close(fig)
    print("wrote assets/data_distribution.png")


def ledgar_runs():
    rows = []
    for path in sorted(RUNS.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("dataset") == "ledgar" and "accuracy" in row and "name" in row:
            rows.append(row)
    return sorted(rows, key=lambda row: row["accuracy"])


def write_accuracy():
    rows = ledgar_runs()
    if not rows:
        print("skip accuracy.png: no LEDGAR run summaries found")
        return

    names = [row["name"] for row in rows]
    values = [row["accuracy"] * 100 for row in rows]
    colors = [ORANGE if "fine" in name.lower() or "ours" in name.lower() else BLUE for name in names]

    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.55 * len(rows) + 1.5)))
    ax.barh(names, values, color=colors)
    for index, value in enumerate(values):
        ax.text(value + 0.8, index, f"{value:.1f}%", va="center", fontsize=11, color=NAVY)
    ax.set_xlim(0, 100)
    ax.set_xlabel("LEDGAR accuracy (%)")
    ax.set_title("Measured runs on the same held-out split", color=NAVY, fontsize=12)
    no_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png", dpi=150)
    plt.close(fig)
    print("wrote assets/accuracy.png")


def write_throughput():
    artifact = RUNS / "training-throughput.json"
    if not artifact.exists():
        print("skip throughput.png: runs/training-throughput.json not found")
        return
    data = json.loads(artifact.read_text())
    values = [data["with_label_list"], data["without_label_list"]]
    labels = ["with label list", "without label list"]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = ax.bar(labels, values, color=[GRAY, ORANGE], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}/s", ha="center")
    ax.set_ylabel("training throughput (samples/sec)")
    ax.set_title("Prompt design changes training speed", color=NAVY, fontsize=11.5)
    no_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT / "throughput.png", dpi=150)
    plt.close(fig)
    print("wrote assets/throughput.png")


def main():
    write_training_loss()
    write_data_distribution()
    write_accuracy()
    write_throughput()


if __name__ == "__main__":
    main()
