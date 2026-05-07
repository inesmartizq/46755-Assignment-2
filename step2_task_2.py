import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


# PARAMETERS

MIN_LOAD = 220 
EPSILON = 0.10  # P90 Requirement

def load_profiles(path):
    return pd.read_csv(path).values

def compute_flexibility(profiles):
    return profiles - MIN_LOAD


# OPTIMIZATION MODELS (FROM TASK 2.1)


def solve_alsox(flexibility):
    """MILP formulation for P90"""
    W, M = flexibility.shape
    model = gp.Model(); model.setParam("OutputFlag", 0)
    R = model.addVar(lb=0); y = model.addVars(W, M, vtype=GRB.BINARY)
    model.setObjective(R, GRB.MAXIMIZE)
    for w in range(W):
        for m in range(M): 
            model.addConstr(R <= flexibility[w, m] + 10000 * y[w, m])
    model.addConstr(gp.quicksum(y) <= EPSILON * W * M)
    model.optimize(); return R.X

def solve_cvar(flexibility):
    """LP CVaR approximation for P90"""
    W, M = flexibility.shape
    N = W * M
    model = gp.Model(); model.setParam("OutputFlag", 0)
    R = model.addVar(lb=0); beta = model.addVar(lb=-GRB.INFINITY)
    z = model.addVars(W, M, lb=0)
    model.setObjective(R, GRB.MAXIMIZE)
    for w in range(W):
        for m in range(M):
            model.addConstr(z[w, m] >= (R - flexibility[w, m]) - beta)
    model.addConstr(beta + (1 / (EPSILON * N)) * gp.quicksum(z) <= 0)
    model.optimize(); return R.X


# VERIFICATION FUNCTION


def verify_p90_out_of_sample(reserve_bid, profiles, method_name):
    flex = compute_flexibility(profiles)
    # Check how many minutes the bid was higher than actual flexibility
    shortfall_rate = np.mean(flex < reserve_bid)
    # Calculate magnitude: max(0, Bid - Actual)
    shortfall_amt = np.maximum(reserve_bid - flex, 0)
    
    print(f"\n--- TASK 2.2 VERIFICATION: {method_name} ---")
    print(f"Reserve Bid:    {reserve_bid:.2f} kW")
    print(f"Shortfall Rate: {shortfall_rate:.4f} (Target <= {EPSILON})")
    print(f"Exp. Shortfall: {np.mean(shortfall_amt):.4f} kW")
    
    return shortfall_amt

# MAIN EXECUTION

if __name__ == "__main__":
    # Load data
    in_flex = compute_flexibility(load_profiles("data/in_sample_profiles.csv"))
    out_profiles = load_profiles("data/out_sample_profiles.csv")
    out_flex = compute_flexibility(out_profiles)
    
    # Determine bids using in-sample data 
    r_alsox = solve_alsox(in_flex)
    r_cvar = solve_cvar(in_flex)
    
    # Verify against out-of-sample data 
    sf_alsox = verify_p90_out_of_sample(r_alsox, out_profiles, "ALSO-X")
    sf_cvar = verify_p90_out_of_sample(r_cvar, out_profiles, "CVaR")
    
    # 3. Combined Visualization 
    plt.figure(figsize=(10, 6))
    
    # Plotting only positive shortfall events
    plt.hist(sf_alsox.flatten()[sf_alsox.flatten() > 0], bins=30, 
             color='blue', alpha=0.5, label='ALSO-X Shortfalls', edgecolor='black')
    plt.hist(sf_cvar.flatten()[sf_cvar.flatten() > 0], bins=30, 
             color='red', alpha=0.5, label='CVaR Shortfalls', edgecolor='black')
    
    plt.title("Task 2.2: Out-of-Sample Shortfall Magnitude Comparison")
    plt.xlabel("Shortfall Magnitude (kW)")
    plt.ylabel("Frequency (Minutes)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/task2_2_comparison.png")
    plt.show()