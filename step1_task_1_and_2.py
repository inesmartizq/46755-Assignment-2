import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

os.makedirs("results", exist_ok=True)

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

    profits = []
    for scen in scenarios:
        pi = 0
        for t in range(T):
            pi += scen["price"][t] * q_opt[t]
            pi += scen["bp"][t] * (scen["wind"][t] - q_opt[t])
        profits.append(pi)

    exp_profit = sum(s["prob"] * p for s, p in zip(scenarios, profits))
    return q_opt, exp_profit, profits

# For the two-price scenario we have: 
def solve_two_price(scenarios, capacity=500):
    T = 24
    S = len(scenarios)
    prob = scenarios[0]["prob"]

    m = gp.Model("two_price")
    m.setParam("OutputFlag", 0)

    q = m.addVars(T, lb=0, ub=capacity, name="q")
    dev_plus = m.addVars(S, T, lb=0, ub=capacity, name="dev_plus")
    dev_minus = m.addVars(S, T, lb=0, ub=capacity, name="dev_minus")
    z = m.addVars(S, T, vtype=GRB.BINARY, name="z")

    for s, scen in enumerate(scenarios):
        for t in range(T):
            m.addConstr(dev_plus[s, t] - dev_minus[s, t] == scen["wind"][t] - q[t])
            m.addConstr(dev_plus[s, t] <= capacity * z[s, t])
            m.addConstr(dev_minus[s, t] <= capacity * (1 - z[s, t]))

    obj = gp.LinExpr()

    for s, scen in enumerate(scenarios):
        for t in range(T):
            obj += prob * scen["price"][t] * q[t]

    for s, scen in enumerate(scenarios):
        for t in range(T):
            da = float(scen["price"][t])
            bp = float(scen["bp"][t])
            si = int(scen["imbalance"][t])

            if si == 1:
                obj += prob * da * dev_plus[s, t]
                obj -= prob * bp * dev_minus[s, t]
            else:
                obj += prob * bp * dev_plus[s, t]
                obj -= prob * da * dev_minus[s, t]

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price model not optimal. Status: {m.status}")

    q_opt = np.array([q[t].X for t in range(T)])

    profits = []
    for scen in scenarios:
        pi = 0
        for t in range(T):
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
    return q_opt, exp_profit, profits
def get_wind_stats(scenarios):
    wind_data = np.array([s["wind"] for s in scenarios])

    avg_wind = np.mean(wind_data, axis=0)
    std_wind = np.std(wind_data, axis=0)
    wind_min = np.min(wind_data, axis=0)
    wind_max = np.max(wind_data, axis=0)

    return avg_wind, std_wind, wind_min, wind_max


def plot_task_results(q, profits, exp_profit, scenarios, task_name, offer_label, color, linestyle="-"):
    hours = np.arange(1, 25)
    avg_wind, std_wind, wind_min, wind_max = get_wind_stats(scenarios)

    plt.figure(figsize=(10, 4))

    plt.fill_between(hours, wind_min, wind_max, step="mid", color="gray", alpha=0.10, label="Wind min-max")
    plt.fill_between(hours, avg_wind - std_wind, avg_wind + std_wind,
                     step="mid", color="gray", alpha=0.20, label="Wind mean ± std")
    plt.step(hours, avg_wind, where="mid", label="Avg wind", linewidth=1.5, linestyle=":", color="gray")
    plt.step(hours, q, where="mid", label=offer_label, linewidth=2, color=color, linestyle=linestyle)

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
    axes[0].set_ylabel("Number of scenarios")
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
    avg_wind, std_wind, wind_min, wind_max = get_wind_stats(scenarios)

    plt.figure(figsize=(10, 4))

    plt.fill_between(hours, wind_min, wind_max, step="mid", color="gray", alpha=0.10, label="Wind min-max")
    plt.fill_between(hours, avg_wind - std_wind, avg_wind + std_wind,
                     step="mid", color="gray", alpha=0.20, label="Wind mean ± std")
    plt.step(hours, avg_wind, where="mid", label="Avg wind", linewidth=1.5, linestyle=":", color="gray")

    plt.step(hours, q1, where="mid", label="Task 1.1", linewidth=2, color="steelblue")
    plt.step(hours, q2, where="mid", label="Task 1.2", linewidth=2, linestyle="--", color="darkorange")

    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("DA Offer Comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/da_offers_comparison.png", dpi=150)
    plt.show()
    
def plot_profit_comparison(profits1, profits2):
    sorted_1 = np.sort(profits1)
    sorted_2 = np.sort(profits2)

    cum_1 = np.cumsum(sorted_1)
    cum_2 = np.cumsum(sorted_2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Same bins for fair comparison
    all_profits = np.concatenate([profits1, profits2])
    bins = np.linspace(all_profits.min(), all_profits.max(), 40)

    # Distribution comparison
    axes[0].hist(profits1, bins=bins, alpha=0.6, color="steelblue", edgecolor="white", label="Task 1.1")
    axes[0].hist(profits2, bins=bins, alpha=0.6, color="darkorange", edgecolor="white", label="Task 1.2")
    axes[0].set_title("Profit Distribution Comparison")
    axes[0].set_xlabel("Profit (EUR)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Cumulative comparison
    axes[1].plot(np.arange(1, len(sorted_1) + 1), cum_1, linewidth=2, color="steelblue", label="Task 1.1")
    axes[1].plot(np.arange(1, len(sorted_2) + 1), cum_2, linewidth=2, color="darkorange", linestyle="--", label="Task 1.2")
    axes[1].set_title("Cumulative Profit Comparison")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Cumulative Profit (EUR)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/profit_comparison.png", dpi=150)
    plt.show()