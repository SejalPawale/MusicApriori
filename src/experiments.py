"""
experiments.py: Step 6 - Systematic Parameter Sweep and Runtime Benchmarking.

Performs grid sweep across:
- support ∈ {0.05%, 0.1%, 0.2%, 0.5%}  (0.0005, 0.001, 0.002, 0.005)
- confidence ∈ {0.1, 0.3, 0.5, 0.7}
- lift ∈ {1.0, 2.0, 5.0}
- algorithm comparison: Apriori vs FP-Growth runtime

Records itemset count, rule count, and runtimes; writes results/sweep_results.csv;
generates rule count heatmaps and runtime-vs-support scaling plots.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, SWEEP_RESULTS_CSV
from mining import load_basket_data, mine_frequent_itemsets, generate_rules


def run_parameter_sweep(
    supports=[0.003, 0.005, 0.008, 0.010],
    confidences=[0.1, 0.3, 0.5, 0.7],
    lifts=[1.0, 2.0, 5.0],
    algorithms=["fpgrowth", "apriori"]
):
    print("=" * 70)
    print("TUNETRAIL PARAMETER SWEEP & RUNTIME BENCHMARK")
    print("=" * 70)

    basket_df, num_transactions, items = load_basket_data()
    print(f"Loaded basket with {num_transactions:,} transactions, {len(items):,} items.")

    results = []

    # Cache frequent itemsets per (support, algorithm) to avoid redundant mining
    fi_cache = {}

    for algo in algorithms:
        print(f"\n--- Benchmarking Algorithm: {algo.upper()} ---")
        for supp in supports:
            supp_pct = supp * 100.0
            print(f"  Mining support = {supp_pct:.2f}% ({int(np.ceil(supp * num_transactions))} playlists)...")

            cache_key = (algo, supp)
            t0 = time.perf_counter()
            fi, fi_time = mine_frequent_itemsets(basket_df, min_support=supp, max_len=3, algorithm=algo)
            fi_cache[cache_key] = fi
            num_fi = len(fi)
            print(f"    Itemsets: {num_fi:,} found in {fi_time:.3f}s")

            for conf in confidences:
                for lift_val in lifts:
                    t_rule_0 = time.perf_counter()
                    rules = generate_rules(fi, num_transactions, min_confidence=conf, min_lift=lift_val)
                    rule_time = time.perf_counter() - t_rule_0

                    total_time = fi_time + rule_time
                    results.append({
                        "algorithm": algo,
                        "support": supp,
                        "support_pct": supp_pct,
                        "min_playlists": int(np.ceil(supp * num_transactions)),
                        "confidence": conf,
                        "lift": lift_val,
                        "num_itemsets": num_fi,
                        "num_rules": len(rules),
                        "fi_runtime_sec": round(fi_time, 4),
                        "rule_runtime_sec": round(rule_time, 4),
                        "total_runtime_sec": round(total_time, 4)
                    })

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(SWEEP_RESULTS_CSV, index=False)
    print(f"\nSaved {len(sweep_df)} parameter sweep configurations to: {SWEEP_RESULTS_CSV}")

    # Generate analysis plots
    plot_sweep_heatmaps(sweep_df)
    plot_runtime_comparison(sweep_df)

    return sweep_df


def plot_sweep_heatmaps(sweep_df: pd.DataFrame):
    """
    Plots heatmaps of Rule Counts across Support and Confidence for different Lift thresholds.
    """
    fg_df = sweep_df[sweep_df["algorithm"] == "fpgrowth"]
    if fg_df.empty:
        fg_df = sweep_df

    lifts = sorted(fg_df["lift"].unique())
    fig, axes = plt.subplots(1, len(lifts), figsize=(5 * len(lifts), 4), sharey=True)
    if len(lifts) == 1:
        axes = [axes]

    for idx, l_val in enumerate(lifts):
        sub = fg_df[fg_df["lift"] == l_val]
        pivot = sub.pivot(index="confidence", columns="support_pct", values="num_rules")
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=axes[idx], cbar=idx == len(lifts)-1)
        axes[idx].set_title(f"Rules Count (Lift ≥ {l_val})")
        axes[idx].set_xlabel("Support (%)")
        if idx == 0:
            axes[idx].set_ylabel("Confidence")
        else:
            axes[idx].set_ylabel("")

    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "sweep_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved sweep heatmap to: {heatmap_path}")


def plot_runtime_comparison(sweep_df: pd.DataFrame):
    """
    Plots runtime vs support for Apriori vs FP-Growth.
    """
    agg = sweep_df.groupby(["algorithm", "support_pct"])["fi_runtime_sec"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    for algo in agg["algorithm"].unique():
        sub = agg[agg["algorithm"] == algo].sort_values("support_pct")
        plt.plot(sub["support_pct"], sub["fi_runtime_sec"], marker="o", linewidth=2, label=algo.upper())

    plt.title("Frequent Itemset Mining Runtime vs. Support Floor")
    plt.xlabel("Support Threshold (%)")
    plt.ylabel("Mining Runtime (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    runtime_path = RESULTS_DIR / "sweep_runtime.png"
    plt.savefig(runtime_path, dpi=300)
    plt.close()
    print(f"Saved runtime plot to: {runtime_path}")


if __name__ == "__main__":
    run_parameter_sweep()
