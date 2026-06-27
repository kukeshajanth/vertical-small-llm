"""Line up every runs/*.json into one table + a bar chart for the article."""
import glob
import json


def main():
    rows = [json.load(open(f)) for f in glob.glob("runs/*.json")]
    if not rows:
        print("no runs yet — run eval_local.py / eval_frontier.py first")
        return
    rows.sort(key=lambda r: -r.get("accuracy", 0))

    print(f"\n{'model':46} {'acc':>7} {'sec/req':>9} {'tok/s':>8}")
    print("-" * 74)
    for r in rows:
        tps = r.get("tok_per_s", "-")
        print(f"{r['name']:46} {r['accuracy'] * 100:6.1f}% "
              f"{r.get('sec_per_req', '-'):>9} {tps:>8}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["name"] for r in rows]
        acc = [r["accuracy"] * 100 for r in rows]
        plt.figure(figsize=(8, 0.5 * len(rows) + 1.5))
        bars = plt.barh(names[::-1], acc[::-1])
        for b, a in zip(bars, acc[::-1]):
            plt.text(a + 0.5, b.get_y() + b.get_height() / 2,
                     f"{a:.1f}%", va="center", fontsize=9)
        plt.xlabel("Banking77 accuracy (%)")
        plt.xlim(0, 100)
        plt.title("Small fine-tuned 3B vs current frontier models")
        plt.tight_layout()
        plt.savefig("runs/accuracy.png", dpi=150)
        print("\nwrote runs/accuracy.png")
    except Exception as e:
        print("chart skipped:", e)


if __name__ == "__main__":
    main()
