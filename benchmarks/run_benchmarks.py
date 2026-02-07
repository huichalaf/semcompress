"""Benchmark semcompress on real books from Project Gutenberg."""

import os
import time
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from semcompress import Compactor, CompactorConfig
from semcompress.token_counting import count_tokens

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BENCH_DIR, "results.json")

BOOKS = {
    "The Art of War\n(Sun Tzu)": "the_art_of_war.txt",
    "The Prince\n(Machiavelli)": "the_prince.txt",
    "Frankenstein\n(Shelley)": "frankenstein.txt",
    "A Tale of Two Cities\n(Dickens)": "a_tale_of_two_cities.txt",
    "The Republic\n(Plato)": "the_republic.txt",
}

RATIOS = [0.7, 0.5, 0.3]

COLORS = {0.7: "#4CAF50", 0.5: "#2196F3", 0.3: "#FF5722"}
RATIO_LABELS = {0.7: "70%", 0.5: "50%", 0.3: "30%"}


def strip_gutenberg_header(text):
    """Remove Project Gutenberg header/footer."""
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    start = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start = text.find("\n", idx) + 1
            break

    end = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end = idx
            break

    return text[start:end].strip()


def load_book(filename, max_tokens=5000):
    """Load a book and truncate to max_tokens for benchmark speed."""
    path = os.path.join(BENCH_DIR, filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    text = strip_gutenberg_header(raw)
    words = text.split()

    if len(words) > max_tokens:
        words = words[:max_tokens]

    return " ".join(words)


def run_benchmarks():
    print("=" * 60)
    print("SEMCOMPRESS BENCHMARK - Real Books")
    print("=" * 60)
    print()

    results = {}

    for book_label, filename in BOOKS.items():
        book_name = book_label.replace("\n", " ")
        print(f"Loading {book_name}...")
        text = load_book(filename, max_tokens=5000)
        original_tokens = count_tokens(text, "whitespace")
        print(f"  Original: {original_tokens} tokens")

        results[book_name] = {
            "original_tokens": original_tokens,
            "ratios": {},
        }

        for ratio in RATIOS:
            print(f"  Compacting to {ratio:.0%}...", end=" ", flush=True)

            config = CompactorConfig(
                target_ratio=ratio,
                similarity_threshold=0.3,
                batch_removal_fraction=0.1,
                min_chunks_to_keep=1,
            )
            compactor = Compactor(config)

            start = time.perf_counter()
            result = compactor.compact(text)
            elapsed = time.perf_counter() - start

            actual_ratio = result.compacted_tokens / result.original_tokens
            print(
                f"Done in {elapsed:.1f}s | "
                f"{result.original_tokens} -> {result.compacted_tokens} tokens "
                f"({actual_ratio:.0%}) | "
                f"{result.chunks_removed} removed, {result.chunks_kept} kept, "
                f"{result.iterations} iters"
            )

            results[book_name]["ratios"][str(ratio)] = {
                "target_ratio": ratio,
                "actual_ratio": actual_ratio,
                "original_tokens": result.original_tokens,
                "compacted_tokens": result.compacted_tokens,
                "chunks_removed": result.chunks_removed,
                "chunks_kept": result.chunks_kept,
                "iterations": result.iterations,
                "elapsed_seconds": round(elapsed, 2),
            }

        print()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {RESULTS_FILE}")
    return results


def plot_token_comparison(results):
    """Bar chart: original vs compacted tokens per book at each ratio."""
    fig, ax = plt.subplots(figsize=(14, 7))

    books = list(results.keys())
    short_names = [b.replace(" ", "\n", 1) if len(b) > 20 else b for b in books]
    x = np.arange(len(books))
    total_width = 0.7
    bar_width = total_width / (len(RATIOS) + 1)

    # Original bars
    originals = [results[b]["original_tokens"] for b in books]
    ax.bar(x - total_width / 2, originals, bar_width, label="Original", color="#9E9E9E", alpha=0.7)

    for i, ratio in enumerate(RATIOS):
        compacted = []
        for b in books:
            r = results[b]["ratios"].get(str(ratio), {})
            compacted.append(r.get("compacted_tokens", 0))

        offset = -total_width / 2 + (i + 1) * bar_width
        bars = ax.bar(
            x + offset, compacted, bar_width,
            label=f"Target {RATIO_LABELS[ratio]}", color=COLORS[ratio], alpha=0.85
        )

        for bar, val, orig in zip(bars, compacted, originals):
            pct = val / orig * 100 if orig > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_ylabel("Tokens", fontsize=12, fontweight="bold")
    ax.set_title(
        "semcompress — Token Compression on Classic Books\n(5,000 token samples)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(BENCH_DIR, "token_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_compression_efficiency(results):
    """Grouped bar: target ratio vs actual achieved ratio per book."""
    fig, ax = plt.subplots(figsize=(14, 7))

    books = list(results.keys())
    short_names = [b.replace(" ", "\n", 1) if len(b) > 20 else b for b in books]
    x = np.arange(len(books))
    bar_width = 0.22

    for i, ratio in enumerate(RATIOS):
        actual = []
        for b in books:
            r = results[b]["ratios"].get(str(ratio), {})
            actual.append(r.get("actual_ratio", 1.0) * 100)

        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset, actual, bar_width,
            label=f"Target {RATIO_LABELS[ratio]}", color=COLORS[ratio], alpha=0.85
        )

        # Target line
        for j in range(len(books)):
            ax.hlines(
                ratio * 100, x[j] + offset - bar_width / 2, x[j] + offset + bar_width / 2,
                colors="black", linestyles="dashed", linewidth=1.2
            )

        for bar, val in zip(bars, actual):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_ylabel("Achieved Ratio (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "semcompress — Target vs Achieved Compression Ratio\n(dashed = target, solid = achieved)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(BENCH_DIR, "compression_efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_performance(results):
    """Scatter: tokens processed vs time, colored by ratio."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for ratio in RATIOS:
        tokens = []
        times = []
        names = []
        for b in results:
            r = results[b]["ratios"].get(str(ratio), {})
            if r:
                tokens.append(r["original_tokens"])
                times.append(r["elapsed_seconds"])
                names.append(b.split("(")[0].strip())

        ax.scatter(
            tokens, times, s=120, color=COLORS[ratio],
            label=f"Target {RATIO_LABELS[ratio]}", alpha=0.8, edgecolors="white", linewidth=1
        )

        for t, tm, n in zip(tokens, times, names):
            ax.annotate(
                n, (t, tm), textcoords="offset points", xytext=(8, 5),
                fontsize=7, alpha=0.7
            )

    ax.set_xlabel("Input Tokens", fontsize=12, fontweight="bold")
    ax.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "semcompress — Processing Speed\n(CPU, all-MiniLM-L6-v2)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(BENCH_DIR, "performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_chunks_removed(results):
    """Stacked bar: chunks kept vs removed per book at 50% ratio."""
    fig, ax = plt.subplots(figsize=(12, 7))

    books = list(results.keys())
    short_names = [b.replace(" ", "\n", 1) if len(b) > 20 else b for b in books]

    kept = []
    removed = []
    for b in books:
        r = results[b]["ratios"].get("0.5", {})
        kept.append(r.get("chunks_kept", 0))
        removed.append(r.get("chunks_removed", 0))

    x = np.arange(len(books))
    bar_width = 0.5

    bars_kept = ax.bar(x, kept, bar_width, label="Kept (high importance)", color="#4CAF50", alpha=0.85)
    bars_removed = ax.bar(x, removed, bar_width, bottom=kept, label="Removed (low importance)", color="#FF5722", alpha=0.6)

    for bar_k, bar_r, k, r in zip(bars_kept, bars_removed, kept, removed):
        total = k + r
        ax.text(
            bar_k.get_x() + bar_k.get_width() / 2, total + 1,
            f"{k}/{total}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Number of Chunks (sentences)", fontsize=12, fontweight="bold")
    ax.set_title(
        "semcompress — Chunks Kept vs Removed (50% target)\n(numbers show kept/total)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(BENCH_DIR, "chunks_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    results = run_benchmarks()

    print()
    print("Generating charts...")
    plot_token_comparison(results)
    plot_compression_efficiency(results)
    plot_performance(results)
    plot_chunks_removed(results)
    print()
    print("All done!")
