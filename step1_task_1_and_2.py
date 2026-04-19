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


def plot_results(q1, q2, profits1, profits2, ep1, ep2, scenarios):
    hours = np.arange(1, 25)
    avg_wind = np.mean([s["wind"] for s in scenarios], axis=0)

    plt.figure(figsize=(11, 4))
    plt.step(hours, q1, where="mid", label="One-price", linewidth=2)
    plt.step(hours, q2, where="mid", label="Two-price", linewidth=2, linestyle="--")
    plt.step(hours, avg_wind, where="mid", label="Avg wind", linewidth=1.5, linestyle=":")
    plt.xlabel("Hour")
    plt.ylabel("MW")
    plt.title("Optimal DA Offers — One-price vs Two-price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/da_offers.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    items = [
        (axes[0], profits1, ep1, "One-price"),
        (axes[1], profits2, ep2, "Two-price"),
    ]

    for ax, profits, ep, title in items:
        color = "steelblue" if title == "One-price" else "darkorange"
        ax.hist(profits, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(ep, color="red", linestyle="--", linewidth=2, label=f"E[π] = {ep:,.0f} EUR")
        ax.set_title(f"{title} profit distribution")
        ax.set_xlabel("Daily profit (EUR)")
        ax.set_ylabel("Scenarios")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Profit Distribution — 1,600 Scenarios")
    plt.tight_layout()
    plt.savefig("results/profit_distributions.png", dpi=150)
    plt.show()