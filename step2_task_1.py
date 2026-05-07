import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# PARAMETERS

MIN_LOAD = 220              # kW 
EPSILON = 0.10              # P90 requirement 
BIG_M = 10000               # Big-M for ALSO-X
NUM_MINUTES = 60            # Minute-level resolution 


# DATA LOADING & PRE-PROCESSING


def load_in_sample_profiles():
    """
    Load the 100 in-sample profiles.
    Shape: (100 scenarios, 60 minutes)
    """
    df = pd.read_csv("data/in_sample_profiles.csv")
    return df.values

def compute_flexibility(profiles):
    """
    For FCR-D UP, reserve is provided by reducing consumption.
    Available reserve = current load - minimum technical load.
    """
    flexibility = profiles - MIN_LOAD
    return flexibility


# OPTIMIZATION MODELS


def solve_alsox(flexibility, epsilon=EPSILON):
    """
    ALSO-X sample-based MILP formulation.
    Determines the optimal FCR-D UP reserve bid (kW) satisfying P90.
    """
    W, M = flexibility.shape
    model = gp.Model("Task2_1_ALSOX")
    model.setParam("OutputFlag", 0)

    # Decision variable: Reserve Bid (kW)
    R = model.addVar(lb=0, name="ReserveBid")

    # Binary violation variables: 1 if bid > flexibility
    y = model.addVars(W, M, vtype=GRB.BINARY, name="Violation")

    # Objective: Maximize the reserve bid R
    model.setObjective(R, GRB.MAXIMIZE)

    # Constraints: R <= flexibility + BigM * y
    for w in range(W):
        for m in range(M):
            model.addConstr(
                R <= flexibility[w, m] + BIG_M * y[w, m],
                name=f"BigM_{w}_{m}"
            )

    # P90 violation budget constraint (Total violations <= epsilon * Total samples)
    model.addConstr(
        gp.quicksum(y[w, m] for w in range(W) for m in range(M))
        <= epsilon * W * M,
        name="P90_budget"
    )

    model.optimize()
    return R.X

def solve_cvar(flexibility, epsilon=EPSILON):
    """
    CVaR approximation (LP)[cite: 83].
    Alternative method to satisfy the P90 requirement[cite: 82, 83].
    """
    W, M = flexibility.shape
    N = W * M
    model = gp.Model("Task2_1_CVaR")
    model.setParam("OutputFlag", 0)

    R = model.addVar(lb=0, name="ReserveBid")
    beta = model.addVar(lb=-GRB.INFINITY, name="Beta")
    z = model.addVars(W, M, lb=0, name="Slack")

    model.setObjective(R, GRB.MAXIMIZE)

    # CVaR formulation constraints
    for w in range(W):
        for m in range(M):
            shortfall = R - flexibility[w, m]
            model.addConstr(z[w, m] >= shortfall - beta)

    # CVaR constraint representing the 1-epsilon (P90) tail
    model.addConstr(
        beta + (1 / (epsilon * N)) * gp.quicksum(z[w, m] for w in range(W) for m in range(M))
        <= 0
    )

    model.optimize()
    return R.X


# MAIN EXECUTION


if __name__ == "__main__":
    print("--- STEP 2: TASK 2.1 (In-sample Decision Making) ---")

    # Load and process in-sample data (100 profiles) 
    try:
        in_profiles = load_in_sample_profiles()
        in_flex = compute_flexibility(in_profiles)
        print(f"Loaded {in_profiles.shape[0]} profiles with {in_profiles.shape[1]} minutes each.")

        # Solve using ALSO-X 
        res_alsox = solve_alsox(in_flex)
        print(f"ALSO-X Optimal Reserve Bid: {res_alsox:.2f} kW")

        # Solve using CVaR 
        res_cvar = solve_cvar(in_flex)
        print(f"CVaR Optimal Reserve Bid:    {res_cvar:.2f} kW")

        # 4. Final summary
        print("\nResults Summary:")
        print(f"Difference: {res_alsox - res_cvar:.2f} kW")
        
    except FileNotFoundError:
        print("Error: 'data/in_sample_profiles.csv' not found. Please ensure data exists.")