import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# Constants from assignment description [cite: 69, 75, 87]
MIN_LOAD = 220  # kW [cite: 75]
P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00] # Thresholds for Task 2.3 [cite: 87]

def load_profiles(path): 
    return pd.read_csv(path).values

def compute_flexibility(profiles): 
    # Available reserve for FCR-D UP = Consumption - MIN_LOAD [cite: 69, 75]
    return profiles - MIN_LOAD

def solve_alsox(flexibility, epsilon):
    """MILP formulation for P90 using ALSO-X [cite: 81, 83]"""
    W, M = flexibility.shape
    model = gp.Model(); model.setParam("OutputFlag", 0)
    R = model.addVar(lb=0, name="ReserveBid")
    y = model.addVars(W, M, vtype=GRB.BINARY, name="Violation")
    model.setObjective(R, GRB.MAXIMIZE)
    
    for w in range(W):
        for m in range(M): 
            # Big-M constraint for ALSO-X
            model.addConstr(R <= flexibility[w, m] + 10000 * y[w, m])
    
    # Violation budget constraint (epsilon = 1 - P) [cite: 81, 87]
    model.addConstr(gp.quicksum(y) <= epsilon * W * M)
    model.optimize()
    return R.X

if __name__ == "__main__":
    # Load in-sample (100) and out-of-sample (200) profiles [cite: 78, 79]
    in_flex = compute_flexibility(load_profiles("data/in_sample_profiles.csv"))
    out_flex = compute_flexibility(load_profiles("data/out_sample_profiles.csv"))
    
    results = []
    for p in P_LEVELS:
        # epsilon is the allowed violation rate [cite: 87]
        eps = max(1 - p, 1e-6) 
        bid = solve_alsox(in_flex, eps)
        # Evaluate possible shortfalls against out-of-sample profiles [cite: 84, 85]
        sf_amt = np.mean(np.maximum(bid - out_flex, 0))
        results.append([p, bid, sf_amt])
    
    df = pd.DataFrame(results, columns=["P_requirement", "Reserve_Bid_kW", "Expected_Shortfall_kW"])
    
    # Plotting with Swapped Axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # RED AXIS: Expected Reserve Shortfall (Now on the LEFT)
    color_sf = 'tab:red'
    ax1.set_xlabel('P-Requirement (Reliability Threshold %)')
    ax1.set_ylabel('Expected Reserve Shortfall (kW)', color=color_sf)
    line1 = ax1.plot(df['P_requirement'] * 100, df['Expected_Shortfall_kW'], 'r-s', label='Expected Shortfall (Out-of-Sample)')
    ax1.tick_params(axis='y', labelcolor=color_sf)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # BLUE AXIS: Optimal Reserve Bid (Now on the RIGHT)
    ax2 = ax1.twinx()
    color_bid = 'tab:blue'
    ax2.set_ylabel('Optimal Reserve Bid (kW)', color=color_bid)
    line2 = ax2.plot(df['P_requirement'] * 100, df['Reserve_Bid_kW'], 'b-o', label='Reserve Bid (In-Sample)')
    ax2.tick_params(axis='y', labelcolor=color_bid)

    plt.title("Task 2.3: Trade-off between Reliability and Capacity Provision")
    fig.tight_layout() 
    
    # Unified legend positioned below the plot
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.savefig("data/task2_3_swapped_axes.png", bbox_inches='tight')
    plt.show()