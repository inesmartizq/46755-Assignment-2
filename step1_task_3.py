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
import matplotlib.ticker as mtick

# Format y-axis as thousands (e.g. 350000 → "350")
_K_FORMATTER = mtick.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
_PROFIT_LABEL = "Expected Profit [k€]"

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
# Vary in-sample size – fixed OOS block
# ---------------------------------------------------------------------------

def run_vary_is_fixed_oos(all_scenarios, is_sizes=None, n_folds=8,
                           oos_start=1000):
    """
    Study IS sizes around 200 with 8-fold CV per size, all evaluated on the
    SAME fixed OOS block so the comparison across IS sizes stays fair.

    Design
    ------
    - Fixed OOS block : all_scenarios[oos_start:]   (default scenarios 1000–1599,
                        i.e. 600 scenarios — same set is reused for every IS size
                        and every fold).
    - IS pool         : all_scenarios[:oos_start]   (default 1000 scenarios — the
                        only scenarios from which IS folds are sampled).
    - For each IS size: take `n_folds` *contiguous* IS blocks of length `is_size`,
                        starting positions spread evenly across the IS pool.
                        Each block = one fold. Blocks may overlap when
                        n_folds × is_size > len(IS pool); that is fine — we just
                        want n_folds different windows to average over.
    - No shuffling: keeps the time structure of the underlying wind / price data.

    Parameters
    ----------
    all_scenarios : list of scenario dicts (length 1 600, time-ordered)
    is_sizes      : IS sizes to test (default [100, 150, 200, 250, 300] —
                    a window AROUND the original 200 used in the main CV)
    n_folds       : number of folds per IS size (default 8)
    oos_start     : index where the fixed OOS block starts (default 1000)

    Returns
    -------
    dict with keys
        "is_sizes": list of IS sizes
        "n_folds" : int
        "is_one"  : list (per IS size) of lists (per fold) of IS expected profits
        "oos_one" : same structure for OOS expected profits, one-price scheme
        "is_two", "oos_two": same for two-price
    """
    if is_sizes is None:
        is_sizes = [50, 100, 150, 200, 250, 300, 350, 400]

    # Fixed OOS block — same scenarios for every IS size and every fold.
    fixed_oos  = all_scenarios[oos_start:]
    out_sample = set_equal_probs(fixed_oos)

    # IS pool — every IS fold is sampled exclusively from here, so OOS and IS
    # are guaranteed disjoint.
    is_pool = all_scenarios[:oos_start]

    results = {
        "is_sizes": is_sizes,
        "n_folds":  n_folds,
        "is_one":   [],
        "oos_one":  [],
        "is_two":   [],
        "oos_two":  [],
    }

    print(f"\nFixed OOS block : scenarios [{oos_start} → {len(all_scenarios)}] "
          f"({len(fixed_oos)} scenarios)")
    print(f"IS pool         : scenarios [0 → {oos_start}] ({len(is_pool)} scenarios)")
    print(f"Folds per size  : {n_folds}")
    print("=" * 70)

    for is_size in is_sizes:
        if is_size > len(is_pool):
            print(f"Skipping IS={is_size}: bigger than IS pool ({len(is_pool)}).")
            continue

        # n_folds contiguous IS blocks, starts evenly spread across the IS pool.
        max_start = len(is_pool) - is_size
        if n_folds == 1 or max_start == 0:
            starts = [0]
        else:
            starts = np.linspace(0, max_start, n_folds, dtype=int).tolist()

        is_one_folds,  oos_one_folds = [], []
        is_two_folds,  oos_two_folds = [], []

        print(f"\nIS size = {is_size} | block starts = {starts}")
        for fold, start in enumerate(starts):
            end       = start + is_size
            in_sample = set_equal_probs(is_pool[start:end])

            # One-price LP
            q1, ep1_is, _ = solve_one_price(in_sample)
            _, ep1_oos     = compute_one_price_profits(q1, out_sample)

            # Two-price MILP
            q2, ep2_is, _ = solve_two_price(in_sample)
            _, ep2_oos     = compute_two_price_profits(q2, out_sample)

            is_one_folds.append(ep1_is)
            oos_one_folds.append(ep1_oos)
            is_two_folds.append(ep2_is)
            oos_two_folds.append(ep2_oos)

            print(f"  Fold {fold+1}/{n_folds} (start={start:>3}) | "
                  f"1P IS: {ep1_is:>10,.0f} OOS: {ep1_oos:>10,.0f} | "
                  f"2P IS: {ep2_is:>10,.0f} OOS: {ep2_oos:>10,.0f}")

        # Store per-fold lists (one list per IS size).
        results["is_one"].append(is_one_folds)
        results["oos_one"].append(oos_one_folds)
        results["is_two"].append(is_two_folds)
        results["oos_two"].append(oos_two_folds)

    # Summary table — average across the n_folds for each IS size.
    print("\n" + "=" * 75)
    print(f"{'IS size':<10} {'1P IS μ':>12} {'1P OOS μ':>12} "
          f"{'2P IS μ':>12} {'2P OOS μ':>12}")
    print("-" * 75)
    for i, is_size in enumerate(results["is_sizes"]):
        print(f"{is_size:<10} "
              f"{np.mean(results['is_one'][i]):>12,.0f} "
              f"{np.mean(results['oos_one'][i]):>12,.0f} "
              f"{np.mean(results['is_two'][i]):>12,.0f} "
              f"{np.mean(results['oos_two'][i]):>12,.0f}")

    return results


# ---------------------------------------------------------------------------
# Plot functions
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
        ax.set_ylabel(_PROFIT_LABEL)
        ax.set_xticks(folds)
        ax.yaxis.set_major_formatter(_K_FORMATTER)
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

    # Annotate each bar with its value (in k€)
    for bar in list(bars_is) + list(bars_oos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{bar.get_height() / 1000:,.1f}k",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["One-price", "Two-price"])
    ax.set_ylabel(_PROFIT_LABEL)
    ax.yaxis.set_major_formatter(_K_FORMATTER)
    ax.set_title(
        "Task 1.3 – Averaged In-sample vs Out-of-sample Expected Profits\n"
        "(8-Fold Cross-Validation)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task_1_3_avg_comparison.png", dpi=150)
    plt.show()


def plot_vary_is_fixed_oos(results):
    """
    Line plot of average IS and OOS profit vs in-sample size for both
    schemes. Each IS size has n_folds values (from 8-fold CV with the SAME
    fixed OOS block); we plot only the per-fold mean as a single line.

    A vertical dashed line marks IS=200, the size used in the main CV.
    """
    is_sizes = results["is_sizes"]
    n_folds  = results.get("n_folds", None)

    # Convert per-fold lists to (n_sizes, n_folds) arrays and take the
    # across-fold mean for each IS size.
    is_one  = np.array(results["is_one"]).mean(axis=1)
    oos_one = np.array(results["oos_one"]).mean(axis=1)
    is_two  = np.array(results["is_two"]).mean(axis=1)
    oos_two = np.array(results["oos_two"]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # One-price (blue): solid IS, dashed OOS
    ax.plot(is_sizes, is_one,  marker="o", markersize=8,
            color="#1f77b4", linewidth=2.2, label="One-price IS")
    ax.plot(is_sizes, oos_one, marker="D", markersize=8,
            color="#1f77b4", linewidth=2.2, linestyle="--",
            markerfacecolor="white", markeredgewidth=1.8,
            label="One-price OOS")

    # Two-price (orange): solid IS, dashed OOS
    ax.plot(is_sizes, is_two,  marker="s", markersize=8,
            color="#d35400", linewidth=2.2, label="Two-price IS")
    ax.plot(is_sizes, oos_two, marker="^", markersize=8,
            color="#d35400", linewidth=2.2, linestyle="--",
            markerfacecolor="white", markeredgewidth=1.8,
            label="Two-price OOS")

    # Reference line: the IS size used in the main 8-fold CV
    ax.axvline(x=200, color="gray", linestyle=":", linewidth=1.5,
               label="Original IS = 200")

    title_folds = f"{n_folds}-fold CV per size" if n_folds else "CV per size"
    ax.set_xlabel("In-sample size")
    ax.set_ylabel(_PROFIT_LABEL)
    ax.set_xticks(is_sizes)
    ax.yaxis.set_major_formatter(_K_FORMATTER)
    ax.set_title(
        f"Task 1.3 – Avg IS / OOS Profit vs In-sample Size\n"
        f"({title_folds}, fixed OOS = scenarios 1000–1600)"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task_1_3_vary_is_fixed_oos.png", dpi=150)
    plt.show()