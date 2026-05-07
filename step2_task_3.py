import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# Constants 
MIN_LOAD = 220  # kW 
P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00] # Thresholds for Task 2.3 

def load_profiles(path): 
    return pd.read_csv(path).values

def compute_flexibility(profiles): 
    # Available reserve for FCR-D UP = Consumption - MIN_LOAD 
    return profiles - MIN_LOAD

def solve_alsox(flexibility, epsilon):
    """MILP formulation for P90 using ALSO-X"""
    W, M = flexibility.shape
    model = gp.Model(); model.setParam("OutputFlag", 0)
    R = model.addVar(lb=0, name="ReserveBid")
    y = model.addVars(W, M, vtype=GRB.BINARY, name="Violation")
    model.setObjective(R, GRB.MAXIMIZE)
    
    for w in range(W):
        for m in range(M): 
            # Constraint for ALSO-X
            model.addConstr(R <= flexibility[w, m] + 10000 * y[w, m])
    
    # Violation budget constraint (epsilon = 1 - P) 
    model.addConstr(gp.quicksum(y) <= epsilon * W * M)
    model.optimize()
    return R.X

def run_sensitivity_analysis(in_flex, out_flex, p_levels):
    """
    Wraps the logic into a function so main.py can call it.
    """
    results = []
    for p in p_levels:
        # epsilon is the allowed violation rate
        eps = max(1 - p, 1e-6) 
        bid = solve_alsox(in_flex, eps)
        # Evaluate possible shortfalls against out-of-sample profiles
        sf_amt = np.mean(np.maximum(bid - out_flex, 0))
        results.append([p, bid, sf_amt])
    
    df = pd.DataFrame(results, columns=["P_requirement", "Reserve_Bid_kW", "Expected_Shortfall_kW"])
    
    # Plotting 
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # RED AXIS: Expected Reserve Shortfall 
    color_sf = 'tab:red'
    ax1.set_xlabel('P-Requirement (Reliability Threshold %)')
    ax1.set_ylabel('Expected Reserve Shortfall (kW)', color=color_sf)
    line1 = ax1.plot(df['P_requirement'] * 100, df['Expected_Shortfall_kW'], 'r-s', label='Expected Shortfall (Out-of-Sample)')
    ax1.tick_params(axis='y', labelcolor=color_sf)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # BLUE AXIS: Optimal Reserve Bid 
    ax2 = ax1.twinx()
    color_bid = 'tab:blue'
    ax2.set_ylabel('Optimal Reserve Bid (kW)', color=color_bid)
    line2 = ax2.plot(df['P_requirement'] * 100, df['Reserve_Bid_kW'], 'b-o', label='Reserve Bid (In-Sample)')
    ax2.tick_params(axis='y', labelcolor=color_bid)

    plt.title("Task 2.3: Trade-off between Reliability and Capacity Provision")
    fig.tight_layout() 
    
    # Legend
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.savefig("results/task2_3_tradeoff.png", bbox_inches='tight')
    print("Task 2.3 Plot saved to results/task2_3_tradeoff.png")
    # plt.show() # Can be toggled if you want to see the plot immediately

if __name__ == "__main__":
    # This keeps the script working as a standalone file too
    in_flex = compute_flexibility(load_profiles("data/in_sample_profiles.csv"))
    out_profiles = load_profiles("data/out_sample_profiles.csv")
    out_flex = compute_flexibility(out_profiles)
    
    run_sensitivity_analysis(in_flex, out_flex, P_LEVELS)