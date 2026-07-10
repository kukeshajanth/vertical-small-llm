"""Build an auditable table and accuracy chart from runs/*.json."""
import json
from collections import defaultdict
from pathlib import Path


RUNS_DIR = Path("runs")


def load_runs():
    rows = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping {path}: {exc}")
            continue
        if not {"name", "dataset", "accuracy"}.issubset(row):
            print(f"skipping {path}: not a run summary")
            continue
        rows.append(row)
    return rows


def print_table(dataset, rows):
    print(f"\n{dataset.upper()}  n varies by run; inspect JSON before comparing")
    print(f"{'model':46} {'n':>6} {'acc':>7} {'errors':>7} {'sec/req':>9} {'tok/s':>8}")
    print("-" * 91)
    for row in rows:
        tps = row.get("tok_per_s", "-")
        sec = row.get("sec_per_req", "-")
        print(
            f"{row['name']:46} {row.get('n', '-'):>6} {row['accuracy'] * 100:6.1f}% "
            f"{row.get('errors', 0):>7} {sec:>9} {tps:>8}"
        )


def write_chart(dataset, rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("chart skipped:", exc)
        return

    names = [row["name"] for row in rows]
    accuracy = [row["accuracy"] * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
    bars = ax.barh(names[::-1], accuracy[::-1])
    for bar, value in zip(bars, accuracy[::-1]):
        ax.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel(f"{dataset.upper()} accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"Model accuracy on {dataset.upper()}")
    fig.tight_layout()
    output = RUNS_DIR / f"{dataset}-accuracy.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"wrote {output}")


def main():
    rows = load_runs()
    if not rows:
        print("no runs yet; run eval_local.py or a frontier evaluator first")
        return

    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset in sorted(by_dataset):
        dataset_rows = sorted(by_dataset[dataset], key=lambda row: -row["accuracy"])
        print_table(dataset, dataset_rows)
        write_chart(dataset, dataset_rows)


if __name__ == "__main__":
    main()
