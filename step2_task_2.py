"""
Step 2 – Task 2.2: Out-of-Sample Verification
==============================================
Verifies the P90 reserve bids obtained in Task 2.1 against the
200 out-of-sample load profiles, and plots the shortfall comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MIN_LOAD = 220   # kW
EPSILON  = 0.10  # P90 requirement


# ---------------------------------------------------------------------------
# Data loading and pre-processing
# ---------------------------------------------------------------------------

def load_profiles(path):
    """
    Load load profiles from a CSV file.

    Returns
    -------
    np.array of shape (N, 60)
    """
    return pd.read_csv(path).values


def compute_flexibility(profiles):
    """
    Compute available FCR-D UP flexibility per minute per scenario.

    Returns
    -------
    np.array — same shape as profiles
    """
    return profiles - MIN_LOAD


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_p90_out_of_sample(reserve_bid, profiles, method_name):
    """
    Verify a reserve bid against out-of-sample profiles.

    Prints the shortfall rate and expected shortfall magnitude,
    and returns the per-minute shortfall array.

    Parameters
    ----------
    reserve_bid : float        — bid to verify [kW]
    profiles    : np.array     — out-of-sample load profiles (N, 60)
    method_name : str          — label for printing ("ALSO-X" or "CVaR")

    Returns
    -------
    np.array — per-minute shortfall magnitudes max(0, bid - flexibility)
    """
    flex           = compute_flexibility(profiles)
    shortfall_rate = np.mean(flex < reserve_bid)
    shortfall_amt  = np.maximum(reserve_bid - flex, 0)

    print(f"\n--- Task 2.2 Verification: {method_name} ---")
    print(f"  Reserve bid    : {reserve_bid:.2f} kW")
    print(f"  Shortfall rate : {shortfall_rate:.4f}  (target <= {EPSILON})")
    print(f"  Exp. shortfall : {np.mean(shortfall_amt):.4f} kW")

    return shortfall_amt


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_shortfall_comparison(sf_alsox, sf_cvar):
    """
    Histogram comparing out-of-sample shortfall magnitudes for both methods.
    Only positive shortfall events are shown.

    Saves to results/task2_2_comparison.png.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(sf_alsox.flatten()[sf_alsox.flatten() > 0], bins=30,
            color="steelblue", alpha=0.6, edgecolor="black",
            label="ALSO-X shortfalls")
    ax.hist(sf_cvar.flatten()[sf_cvar.flatten() > 0], bins=30,
            color="darkorange", alpha=0.6, edgecolor="black",
            label="CVaR shortfalls")

    ax.set_title("Task 2.2 – Out-of-Sample Shortfall Magnitude Comparison")
    ax.set_xlabel("Shortfall Magnitude [kW]")
    ax.set_ylabel("Frequency [minutes]")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("results/task_2_2_comparison.png", dpi=150)
    #plt.show()