"""
Step 1 – Task 1.3: 8-Fold Cross-Validation
===========================================
Total pool  : 1 600 scenarios (20 wind × 20 price × 4 imbalance)
Each fold   : 200 in-sample  (optimise DA offers)
              1 400 out-of-sample (evaluate the fixed offers)
Goal        : compare averaged in-sample vs out-of-sample expected profits
              for both the one-price and two-price schemes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Re-use data loading, solvers, and profit helpers from previous tasks
from step1_scenario_generation import dataframe_building, generate_combined_scenarios
from step1_task_1_and_2 import (
    solve_one_price, solve_two_price,
    compute_one_price_profits, compute_two_price_profits,
)

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def set_equal_probs(scenarios):
    """Return a new list of scenario dicts with uniform probabilities (1/N)."""
    n = len(scenarios)
    # **s copies all existing keys/values; "prob" is then overwritten with 1/n
    return [{**s, "prob": 1.0 / n} for s in scenarios]


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(all_scenarios, n_folds=8):
    """
    Run n_folds-fold cross-validation over all_scenarios.

    Each fold uses FOLD_SIZE = len(all_scenarios) // n_folds scenarios as
    in-sample (for optimisation) and the remaining scenarios as out-of-sample
    (for evaluation of the fixed offer).

    Returns a dict with per-fold IS and OOS expected profits for both schemes.
    """
    N_TOTAL   = len(all_scenarios)
    fold_size = N_TOTAL // n_folds
    n_oos     = N_TOTAL - fold_size

    print(f"\nTotal scenarios : {N_TOTAL}")
    print(f"Folds           : {n_folds}")
    print(f"In-sample / fold: {fold_size}")
    print(f"Out-of-sample   : {n_oos}")
    print("=" * 60)

    is_one, oos_one = [], []
    is_two, oos_two = [], []

    for fold in range(n_folds):
        print(f"\nFold {fold + 1}/{n_folds}")

        # ── Split ──────────────────────────────────────────────────────────
        start = fold * fold_size
        end   = start + fold_size

        # Current fold block → in-sample; everything else → out-of-sample
        raw_in  = all_scenarios[start:end]
        raw_oos = all_scenarios[:start] + all_scenarios[end:]

        # Re-normalise probabilities so they sum to 1 within each split;
        # the solvers and eval functions expect probabilities that sum to 1.
        in_sample  = set_equal_probs(raw_in)   # prob = 1/fold_size
        out_sample = set_equal_probs(raw_oos)  # prob = 1/n_oos

        # ── One-price scheme ───────────────────────────────────────────────
        # Optimise DA offers on the in-sample scenarios
        q1, ep1_is, _ = solve_one_price(in_sample)
        # Evaluate the same fixed offers on the out-of-sample scenarios
        _, ep1_oos = compute_one_price_profits(q1, out_sample)

        is_one.append(ep1_is)
        oos_one.append(ep1_oos)
        print(f"  One-price  |  IS: {ep1_is:>12,.0f} EUR  |  OOS: {ep1_oos:>12,.0f} EUR")

        # ── Two-price scheme ───────────────────────────────────────────────
        # Optimise DA offers on the in-sample scenarios (MILP)
        q2, ep2_is, _ = solve_two_price(in_sample)
        # Evaluate the fixed offers on the out-of-sample scenarios
        _, ep2_oos = compute_two_price_profits(q2, out_sample)

        is_two.append(ep2_is)
        oos_two.append(ep2_oos)
        print(f"  Two-price  |  IS: {ep2_is:>12,.0f} EUR  |  OOS: {ep2_oos:>12,.0f} EUR")

    results = {
        "is_one":  is_one,
        "oos_one": oos_one,
        "is_two":  is_two,
        "oos_two": oos_two,
    }

    # Print summary table
    avg_is_one  = np.mean(is_one)
    avg_oos_one = np.mean(oos_one)
    avg_is_two  = np.mean(is_two)
    avg_oos_two = np.mean(oos_two)

    print("\n" + "=" * 65)
    print("8-Fold Cross-Validation – Averaged Results")
    print("=" * 65)
    print(f"{'Scheme':<12} {'Avg IS E[profit]':>20} {'Avg OOS E[profit]':>20} {'Gap (IS−OOS)':>12}")
    print("-" * 65)
    print(f"{'One-price':<12} {avg_is_one:>20,.0f} {avg_oos_one:>20,.0f} {avg_is_one - avg_oos_one:>12,.0f}")
    print(f"{'Two-price':<12} {avg_is_two:>20,.0f} {avg_oos_two:>20,.0f} {avg_is_two - avg_oos_two:>12,.0f}")

    return results


# ---------------------------------------------------------------------------
# Plot functions (callable from main)
# ---------------------------------------------------------------------------

def plot_cv_per_fold(results, n_folds=8):
    """
    Line plot showing in-sample vs out-of-sample expected profit per fold
    for both the one-price and two-price schemes.
    """
    folds = np.arange(1, n_folds + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, is_vals, oos_vals, title, color in zip(
        axes,
        [results["is_one"],  results["is_two"]],
        [results["oos_one"], results["oos_two"]],
        ["One-price scheme", "Two-price scheme"],
        ["steelblue", "darkorange"],
    ):
        ax.plot(folds, is_vals,  marker="o", linewidth=2, color=color,
                label="In-sample E[profit]")
        ax.plot(folds, oos_vals, marker="s", linewidth=2, color=color,
                linestyle="--", label="Out-of-sample E[profit]")
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Expected Profit (EUR)")
        ax.set_xticks(folds)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Task 1.3 – Per-fold In-sample vs Out-of-sample Expected Profits",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/task_1_3_per_fold.png", dpi=150)
    plt.show()


def plot_cv_avg_comparison(results):
    """
    Bar chart comparing averaged in-sample vs out-of-sample expected profits
    for one-price and two-price schemes across all 8 folds.
    """
    avg_is_one  = np.mean(results["is_one"])
    avg_oos_one = np.mean(results["oos_one"])
    avg_is_two  = np.mean(results["is_two"])
    avg_oos_two = np.mean(results["oos_two"])

    x     = np.array([0, 1])   # group positions: one-price, two-price
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_is = ax.bar(
        x - width / 2,
        [avg_is_one,  avg_is_two],
        width, label="Avg In-sample",
        color=["steelblue", "darkorange"], alpha=0.9,
    )
    bars_oos = ax.bar(
        x + width / 2,
        [avg_oos_one, avg_oos_two],
        width, label="Avg Out-of-sample",
        color=["steelblue", "darkorange"], alpha=0.5,
        edgecolor="black", linewidth=1.2,
    )

    # Annotate each bar with its value
    for bar in list(bars_is) + list(bars_oos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{bar.get_height():,.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["One-price", "Two-price"])
    ax.set_ylabel("Expected Profit (EUR)")
    ax.set_title(
        "Task 1.3 – Averaged In-sample vs Out-of-sample Expected Profits\n"
        "(8-Fold Cross-Validation)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task_1_3_avg_comparison.png", dpi=150)
    plt.show()
