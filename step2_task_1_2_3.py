import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# =========================================================
# STEP 2 - TASK 2.1 + TASK 2.2 + TASK 2.3
# Participation in Ancillary Service Markets
# =========================================================

# =========================================================
# PARAMETERS
# =========================================================

MIN_LOAD = 220              # kW
EPSILON = 0.10              # P90 => max 10% violation
BIG_M = 10000               # Big-M for ALSO-X
NUM_MINUTES = 60

# Task 2.3 range:
# threshold between 80% and 100%
P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]


# =========================================================
# LOAD IN-SAMPLE DATA
# =========================================================

def load_in_sample_profiles():
    """
    Load the 100 in-sample profiles

    Shape:
    (100 scenarios, 60 minutes)
    """
    df = pd.read_csv("data/in_sample_profiles.csv")
    return df.values


# =========================================================
# LOAD OUT-OF-SAMPLE DATA
# =========================================================

def load_out_sample_profiles():
    """
    Load the 200 out-of-sample profiles

    Shape:
    (200 scenarios, 60 minutes)
    """
    df = pd.read_csv("data/out_sample_profiles.csv")
    return df.values


# =========================================================
# FLEXIBILITY CALCULATION
# =========================================================

def compute_flexibility(profiles):
    """
    For FCR-D UP:
    upward reserve is provided by reducing consumption

    available reserve = load - minimum technical load

    flexibility[w, m] = profile[w, m] - MIN_LOAD
    """
    flexibility = profiles - MIN_LOAD
    return flexibility


# =========================================================
# ALSO-X MODEL
# =========================================================

def solve_alsox(flexibility, epsilon=EPSILON):
    """
    ALSO-X sample-based MILP formulation

    Decision:
        R = reserve bid [kW]

    Constraint:
        Total violations <= epsilon * total samples
    """

    W, M = flexibility.shape

    model = gp.Model("Task2_ALSOX")
    model.setParam("OutputFlag", 0)

    # Decision variable
    R = model.addVar(lb=0, name="ReserveBid")

    # Binary violation variables
    y = model.addVars(W, M, vtype=GRB.BINARY, name="Violation")

    # Objective: maximize reserve bid
    model.setObjective(R, GRB.MAXIMIZE)

    # Big-M constraints
    for w in range(W):
        for m in range(M):
            model.addConstr(
                R <= flexibility[w, m] + BIG_M * y[w, m],
                name=f"BigM_{w}_{m}"
            )

    # Violation budget
    model.addConstr(
        gp.quicksum(y[w, m] for w in range(W) for m in range(M))
        <= epsilon * W * M,
        name="P90_budget"
    )

    model.optimize()

    reserve_bid = R.X

    return reserve_bid


# =========================================================
# CVaR MODEL
# =========================================================

def solve_cvar(flexibility, epsilon=EPSILON):
    """
    CVaR approximation (LP)

    More conservative than ALSO-X
    """

    W, M = flexibility.shape
    N = W * M

    model = gp.Model("Task2_CVaR")
    model.setParam("OutputFlag", 0)

    # Decision variables
    R = model.addVar(lb=0, name="ReserveBid")

    # IMPORTANT:
    # beta must be free (can be negative)
    beta = model.addVar(lb=-GRB.INFINITY, name="Beta")

    # Slack variables
    z = model.addVars(W, M, lb=0, name="Slack")

    # Objective: maximize reserve bid
    model.setObjective(R, GRB.MAXIMIZE)

    # CVaR constraints
    for w in range(W):
        for m in range(M):
            shortfall = R - flexibility[w, m]

            model.addConstr(
                z[w, m] >= shortfall - beta,
                name=f"Slack_{w}_{m}"
            )

    model.addConstr(
        beta + (1 / (epsilon * N)) *
        gp.quicksum(z[w, m] for w in range(W) for m in range(M))
        <= 0,
        name="CVaR_constraint"
    )

    model.optimize()

    reserve_bid = R.X

    return reserve_bid


# =========================================================
# TASK 2.1 VALIDATION (IN-SAMPLE)
# =========================================================

def validate_solution(reserve_bid, flexibility, method_name):
    """
    Check in-sample performance
    """

    success_matrix = flexibility >= reserve_bid

    success_rate = np.mean(success_matrix)
    shortfall_rate = 1 - success_rate

    print(f"\n========== {method_name} VALIDATION ==========")
    print(f"Reserve bid: {reserve_bid:.2f} kW")
    print(f"Success rate: {success_rate:.4f}")
    print(f"Shortfall rate: {shortfall_rate:.4f}")
    print("Target shortfall <= 0.10")


# =========================================================
# TASK 2.2
# OUT-OF-SAMPLE VERIFICATION
# =========================================================

def verify_p90_out_of_sample(reserve_bid, profiles, method_name):
    """
    Verify whether the reserve bid satisfies
    the P90 requirement on out-of-sample data
    """

    flexibility = compute_flexibility(profiles)

    # Minute-level verification
    success_matrix = flexibility >= reserve_bid

    success_rate = np.mean(success_matrix)
    shortfall_rate = 1 - success_rate

    # Reserve shortfall magnitude
    shortfall_amount = np.maximum(
        reserve_bid - flexibility,
        0
    )

    expected_shortfall = np.mean(shortfall_amount)
    max_shortfall = np.max(shortfall_amount)

    # Scenario-level verification
    scenario_success = np.all(
        flexibility >= reserve_bid,
        axis=1
    )

    scenario_success_rate = np.mean(scenario_success)
    scenario_failure_rate = 1 - scenario_success_rate

    # Print results
    print(f"\n========== TASK 2.2 : {method_name} ==========")

    print(f"Reserve bid used: {reserve_bid:.2f} kW")

    print("\n--- Minute-level verification ---")
    print(f"Success rate      : {success_rate:.4f}")
    print(f"Shortfall rate    : {shortfall_rate:.4f}")
    print("Target shortfall <= 0.10")

    if shortfall_rate <= 0.10:
        print("P90 requirement satisfied.")
    else:
        print("P90 requirement NOT satisfied.")

    print("\n--- Reserve shortfall magnitude ---")
    print(f"Expected shortfall: {expected_shortfall:.4f} kW")
    print(f"Maximum shortfall : {max_shortfall:.4f} kW")

    print("\n--- Scenario-level verification ---")
    print("(all 60 minutes must satisfy reserve bid)")
    print(f"Scenario success rate : {scenario_success_rate:.4f}")
    print(f"Scenario failure rate : {scenario_failure_rate:.4f}")

    return {
        "success_rate": success_rate,
        "shortfall_rate": shortfall_rate,
        "expected_shortfall": expected_shortfall,
        "max_shortfall": max_shortfall,
        "scenario_success_rate": scenario_success_rate
    }


# =========================================================
# TASK 2.3
# ENERGINET P90 SENSITIVITY ANALYSIS
# =========================================================

def task_23_sensitivity_analysis(in_profiles, out_profiles):
    """
    Analyze how modifying the P90 requirement
    (80% to 100%) affects:

    1. Optimal reserve bid (in-sample)
    2. Expected reserve shortfall (out-of-sample)

    using ALSO-X only
    """

    print("\n=================================================")
    print("TASK 2.3 - ENERGINET P90 SENSITIVITY ANALYSIS")
    print("=================================================")

    in_flex = compute_flexibility(in_profiles)
    out_flex = compute_flexibility(out_profiles)

    results = []

    for p_level in P_LEVELS:
        # Example:
        # P = 0.90 -> epsilon = 0.10
        epsilon = 1 - p_level

        # Avoid division issue for 100%
        if epsilon == 0:
            epsilon = 1e-6

        reserve_bid = solve_alsox(
            in_flex,
            epsilon=epsilon
        )

        # Out-of-sample shortfall
        shortfall_amount = np.maximum(
            reserve_bid - out_flex,
            0
        )

        expected_shortfall = np.mean(shortfall_amount)

        shortfall_rate = np.mean(
            out_flex < reserve_bid
        )

        results.append([
            p_level,
            reserve_bid,
            shortfall_rate,
            expected_shortfall
        ])

        print(f"\nP-level: {int(p_level * 100)}%")
        print(f"Reserve bid           : {reserve_bid:.2f} kW")
        print(f"Out-of-sample shortfall rate : {shortfall_rate:.4f}")
        print(f"Expected shortfall    : {expected_shortfall:.4f} kW")

    results_df = pd.DataFrame(
        results,
        columns=[
            "P_requirement",
            "Reserve_Bid_kW",
            "Shortfall_Rate",
            "Expected_Shortfall_kW"
        ]
    )

    print("\n========== TASK 2.3 SUMMARY TABLE ==========")
    print(results_df)

    # Save results for report
    results_df.to_csv(
        "data/task_23_sensitivity_results.csv",
        index=False
    )

    print("\nTask 2.3 results saved to:")
    print("data/task_23_sensitivity_results.csv")

    print("\nDiscussion for report:")
    print("- Higher P requirement => lower reserve bid")
    print("- Higher reliability => lower shortfall")
    print("- Clear trade-off between reliability and reserve provision")

    return results_df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # =====================================================
    # TASK 2.1
    # =====================================================

    print("\nSTEP 2 - TASK 2.1")
    print("Offering Strategy Under P90 Requirement\n")

    # Load in-sample data
    profiles = load_in_sample_profiles()
    flexibility = compute_flexibility(profiles)

    print("Profiles shape:", profiles.shape)
    print("Flexibility shape:", flexibility.shape)

    # -----------------------------------------------------
    # ALSO-X
    # -----------------------------------------------------

    print("\nSolving ALSO-X model...")

    reserve_alsox = solve_alsox(flexibility)

    print(f"ALSO-X Optimal Reserve Bid: {reserve_alsox:.2f} kW")

    validate_solution(
        reserve_alsox,
        flexibility,
        "ALSO-X"
    )

    # -----------------------------------------------------
    # CVaR
    # -----------------------------------------------------

    print("\nSolving CVaR model...")

    reserve_cvar = solve_cvar(flexibility)

    print(f"CVaR Optimal Reserve Bid: {reserve_cvar:.2f} kW")

    validate_solution(
        reserve_cvar,
        flexibility,
        "CVaR"
    )

    # -----------------------------------------------------
    # FINAL COMPARISON
    # -----------------------------------------------------

    print("\n========== FINAL COMPARISON ==========")
    print(f"ALSO-X Reserve Bid : {reserve_alsox:.2f} kW")
    print(f"CVaR Reserve Bid   : {reserve_cvar:.2f} kW")

    if reserve_cvar <= reserve_alsox:
        print("\nAs expected:")
        print("CVaR is more conservative than ALSO-X.")
    else:
        print("\nCheck formulation:")
        print("CVaR should usually be more conservative.")

    # =====================================================
    # TASK 2.2
    # =====================================================

    print("\n=================================================")
    print("TASK 2.2 - OUT-OF-SAMPLE VERIFICATION")
    print("=================================================")

    # Load out-of-sample data
    out_profiles = load_out_sample_profiles()

    print("\nOut-of-sample profiles shape:", out_profiles.shape)

    # Verify ALSO-X solution
    verify_p90_out_of_sample(
        reserve_alsox,
        out_profiles,
        "ALSO-X"
    )

    # Verify CVaR solution
    verify_p90_out_of_sample(
        reserve_cvar,
        out_profiles,
        "CVaR"
    )

    # =====================================================
    # TASK 2.3
    # =====================================================

    task_23_sensitivity_analysis(
        profiles,
        out_profiles
    )