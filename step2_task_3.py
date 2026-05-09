"""
Step 2 – Task 2.3: Energinet Perspective – Sensitivity Analysis
===============================================================
Sweeps reliability thresholds from P80 to P100, solving the ALSO-X
reserve bid problem at each level, and plots the trade-off between
reliability and capacity provision against out-of-sample performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MIN_LOAD = 220                              # kW
P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]  # default reliability thresholds


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_alsox(flexibility, epsilon):
    """
    ALSO-X MILP: maximise reserve bid subject to a violation budget.

    Parameters
    ----------
    flexibility : np.array (W, M)
    epsilon     : float  — allowed violation fraction (= 1 - P)

    Returns
    -------
    float — optimal reserve bid [kW]
    """
    W, M = flexibility.shape

    m = gp.Model("alsox_sensitivity")
    m.Params.OutputFlag = 0

    R = m.addVar(lb=0, name="R")
    y = m.addVars(W, M, vtype=GRB.BINARY, name="y")

    m.setObjective(R, GRB.MAXIMIZE)

    for w in range(W):
        for t in range(M):
            m.addConstr(R <= flexibility[w, t] + 10000 * y[w, t])

    m.addConstr(
        gp.quicksum(y[w, t] for w in range(W) for t in range(M))
        <= epsilon * W * M
    )

    m.optimize()
    return R.X


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(in_flex, out_flex, p_levels=None):
    """
    Sweep reliability thresholds, solve ALSO-X at each level, and evaluate
    out-of-sample expected shortfall. Produces a dual-axis trade-off plot.

    Parameters
    ----------
    in_flex   : np.array (W_in, M)   — in-sample flexibility
    out_flex  : np.array (W_out, M)  — out-of-sample flexibility
    p_levels  : list of floats       — reliability thresholds (default P80–P100)

    Saves
    -----
    results/task_2_3_tradeoff.png
    """
    if p_levels is None:
        p_levels = P_LEVELS

    results = []
    for p in p_levels:
        epsilon = max(1.0 - p, 1e-6)
        bid     = solve_alsox(in_flex, epsilon)
        sf_amt  = np.mean(np.maximum(bid - out_flex, 0))
        results.append([p, bid, sf_amt])
        print(f"  P={p:.0%}  |  Bid: {bid:.2f} kW  |  Exp. shortfall: {sf_amt:.4f} kW")

    df = pd.DataFrame(results, columns=["P_requirement", "Reserve_Bid_kW",
                                         "Expected_Shortfall_kW"])

    _plot_sensitivity(df)
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_sensitivity(df):
    """Dual-axis plot: reserve bid and expected shortfall vs. P-requirement."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_sf  = "tab:red"
    color_bid = "steelblue"

    ax1.set_xlabel("P-Requirement (Reliability Threshold [%])")
    ax1.set_ylabel("Expected Reserve Shortfall [kW]", color=color_sf)
    line1 = ax1.plot(df["P_requirement"] * 100, df["Expected_Shortfall_kW"],
                     color=color_sf, marker="s", linewidth=2.2,
                     label="Expected shortfall (out-of-sample)")
    ax1.tick_params(axis="y", labelcolor=color_sf)
    ax1.grid(linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Optimal Reserve Bid [kW]", color=color_bid)
    line2 = ax2.plot(df["P_requirement"] * 100, df["Reserve_Bid_kW"],
                     color=color_bid, marker="o", linewidth=2.2,
                     label="Reserve bid (in-sample)")
    ax2.tick_params(axis="y", labelcolor=color_bid)

    lns  = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=2)

    plt.title("Task 2.3 – Reliability vs. Capacity Provision Trade-off")
    plt.tight_layout()
    plt.savefig("results/task_2_3_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved results/task_2_3_tradeoff.png")