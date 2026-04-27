import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

os.makedirs("results", exist_ok=True)

def compute_one_price_profits(q_opt, scenarios):
    """Return per-scenario profits and expected profit for a fixed one-price offer."""
    profits = []
    for scen in scenarios:
        pi = sum(
            scen["price"][t] * q_opt[t] + scen["bp"][t] * (scen["wind"][t] - q_opt[t])
            for t in range(24)
        )
        profits.append(pi)
    exp_profit = sum(s["prob"] * p for s, p in zip(scenarios, profits))
    return profits, exp_profit


def compute_two_price_profits(q_opt, scenarios):
    """Return per-scenario profits and expected profit for a fixed two-price offer."""
    profits = []
    for scen in scenarios:
        pi = 0.0
        for t in range(24):
            da = float(scen["price"][t])
            bp = float(scen["bp"][t])
            si = int(scen["imbalance"][t])
            dev = scen["wind"][t] - q_opt[t]
            pi += da * q_opt[t]
            if si == 1:
                pi += da * max(dev, 0) - bp * max(-dev, 0)
            else:
                pi += bp * max(dev, 0) - da * max(-dev, 0)
        profits.append(pi)
    exp_profit = sum(s["prob"] * p for s, p in zip(scenarios, profits))
    return profits, exp_profit


# For the one-price scheme, we have:
def solve_one_price(scenarios, capacity=500):
    T = 24
    prob = scenarios[0]["prob"]

    m = gp.Model("one_price")
    m.setParam("OutputFlag", 0)

    q = m.addVars(T, lb=0, ub=capacity, name="q")

    obj = gp.LinExpr()
    for scen in scenarios:
        for t in range(T):
            obj += prob * (scen["price"][t] - scen["bp"][t]) * q[t]
            obj += prob * scen["bp"][t] * scen["wind"][t]

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price model not optimal. Status: {m.status}")

    q_opt = np.array([q[t].X for t in range(T)])
    profits, exp_profit = compute_one_price_profits(q_opt, scenarios)
    return q_opt, exp_profit, profits

# For the two-price scenario we have: 
def solve_two_price(scenarios, capacity=500):
    T = 24 #number of hours
    S = len(scenarios) #number of scenarios
    prob = scenarios[0]["prob"] # Assuming all scenarios have the same probability, we can take it from the first one.

    m = gp.Model("two_price")
    m.setParam("OutputFlag", 0)

    q = m.addVars(T, lb=0, ub=capacity, name="q") #DA offers for each hour
    dev_plus = m.addVars(S, T, lb=0, ub=capacity, name="dev_plus") #Positive deviations (wind > DA offer)
    dev_minus = m.addVars(S, T, lb=0, ub=capacity, name="dev_minus") #Negative deviations (wind < DA offer)
    z = m.addVars(S, T, vtype=GRB.BINARY, name="z") #Binary AUXILIARY variable to indicate if we are in a positive or negative deviation scenario

    for s, scen in enumerate(scenarios):
        for t in range(T):
            m.addConstr(dev_plus[s, t] - dev_minus[s, t] == scen["wind"][t] - q[t]) #Deviation definition: dev_plus - dev_minus = actual wind - DA offer
            m.addConstr(dev_plus[s, t] <= capacity * z[s, t]) #If z[s, t] = 0, then dev_plus[s, t] must be 0 (no positive deviation)
            m.addConstr(dev_minus[s, t] <= capacity * (1 - z[s, t])) #If z[s, t] = 1, then dev_minus[s, t] must be 0 (no negative deviation)

    obj = gp.LinExpr()

    for s, scen in enumerate(scenarios):
        for t in range(T):
            obj += prob * scen["price"][t] * q[t] #Revenue from DA offers

    for s, scen in enumerate(scenarios):
        for t in range(T):
            da = float(scen["price"][t])   #DA price for scenario s at time t
            bp = float(scen["bp"][t])      #Balancing price for scenario s at time t
            si = int(scen["imbalance"][t]) #Imbalance indicator for scenario s at time t (1 if positive imbalance, 0 if negative imbalance)

            if si == 1:
                obj += prob * da * dev_plus[s, t]  #Revenue from positive deviations (wind > DA offer) at DA price
                obj -= prob * bp * dev_minus[s, t] #Cost from negative deviations (wind < DA offer) at balancing price
            else:
                obj += prob * bp * dev_plus[s, t]  #Revenue from positive deviations (wind > DA offer) at balancing price
                obj -= prob * da * dev_minus[s, t] #Cost from negative deviations (wind < DA offer) at DA price

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price model not optimal. Status: {m.status}")

    q_opt = np.array([q[t].X for t in range(T)])
    profits, exp_profit = compute_two_price_profits(q_opt, scenarios)
    return q_opt, exp_profit, profits

def plot_task_results(q, profits, exp_profit, scenarios, task_name, offer_label, color, linestyle="-"):
    hours = np.arange(1, 25)
    avg_wind = np.mean([s["wind"] for s in scenarios], axis=0)

    plt.figure(figsize=(10, 4))
    plt.step(hours, q, where="mid", label=offer_label, linewidth=2, color=color, linestyle=linestyle)
    plt.step(hours, avg_wind, where="mid", label="Avg wind", linewidth=1.5, linestyle=":", color="gray")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title(f"{task_name} - Optimal DA Offers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{task_name.lower().replace('.', '_').replace(' ', '_')}_da_offers.png", dpi=150)
    plt.show()

    sorted_profits = np.sort(profits)
    cumulative_profit = np.cumsum(sorted_profits)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(profits, bins=50, color=color, edgecolor="white", alpha=0.85)
    axes[0].axvline(exp_profit, color="red", linestyle="--", linewidth=2,
                    label=f"E[profit] = {exp_profit:,.0f} EUR")
    axes[0].set_title(f"{task_name} - Profit Distribution")
    axes[0].set_xlabel("Profit (EUR)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(np.arange(1, len(sorted_profits) + 1), cumulative_profit, color=color, linewidth=1.8)
    axes[1].set_title(f"{task_name} - Cumulative Profit")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Cumulative Profit (EUR)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{task_name.lower().replace('.', '_').replace(' ', '_')}_profit_analysis.png", dpi=150)
    plt.show()


def plot_offer_comparison(q1, q2, scenarios):
    hours = np.arange(1, 25)
    avg_wind = np.mean([s["wind"] for s in scenarios], axis=0)

    plt.figure(figsize=(10, 4))
    plt.step(hours, q1, where="mid", label="Task 1.1", linewidth=2, color="steelblue")
    plt.step(hours, q2, where="mid", label="Task 1.2", linewidth=2, linestyle="--", color="darkorange")
    plt.step(hours, avg_wind, where="mid", label="Avg wind", linewidth=1.5, linestyle=":", color="gray")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("DA Offer Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/da_offers_comparison.png", dpi=150)
    plt.show()