"""
Step 1 - Task 1.4: Risk-Averse Offering Strategy (CVaR)
========================================================
Two dashboards:

Dashboard 1 — beta sweep, offering strategy and profit variability
  Panel A: E[profit] and CVaR vs beta (one + two)
  Panel B: 24-hour offer schedule q*[t] for each beta (one + two)
  Panel C: Per-scenario profit boxplot vs beta (one + two)
  Panel D: Profit CDF for each beta (one + two)
  Panel E: Profit histogram for each beta (one + two)

Dashboard 2 — in-sample sensitivity (temporal contiguous subsets, NO shuffling)
  Panel A: E[profit] vs beta overlaid across subsets (one + two)
  Panel B: CVaR vs beta overlaid across subsets (one + two)
  Panel C: Optimal offer q*[t] at beta=1 across subsets (one + two)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

os.makedirs("results", exist_ok=True)

CAPACITY  = 500
ALPHA     = 0.90
BETA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_equal_probs(scenarios):
    n = len(scenarios)
    return [{**s, "prob": 1.0 / n} for s in scenarios]


def contiguous_subsets(scenarios, n_subsets=3, subset_size=200):
    """
    Take n_subsets non-overlapping contiguous blocks of subset_size scenarios.
    Preserves temporal order — no shuffling.
    """
    N = len(scenarios)
    if n_subsets * subset_size > N:
        raise ValueError(f"Need {n_subsets*subset_size} scenarios, only have {N}")
    # Spread the blocks evenly across the timeline
    starts = np.linspace(0, N - subset_size, n_subsets, dtype=int)
    return [set_equal_probs(scenarios[s : s + subset_size]) for s in starts]


# ── Solvers ───────────────────────────────────────────────────────────────────

def solve_one_price_cvar(scenarios, beta, alpha=ALPHA):
    T  = 24
    S  = len(scenarios)
    pi = np.array([s["prob"] for s in scenarios])
    coeff = np.array([s["price"] - s["bp"] for s in scenarios])
    const = np.array([s["bp"] @ s["wind"]  for s in scenarios])

    m = gp.Model(); m.Params.OutputFlag = 0
    q   = m.addVars(T, lb=0.0, ub=CAPACITY)
    p   = m.addVars(S, lb=-GRB.INFINITY)
    eta = m.addVar(lb=-GRB.INFINITY)
    xi  = m.addVars(S, lb=0.0)

    for s in range(S):
        m.addConstr(p[s] == gp.quicksum(coeff[s,t]*q[t] for t in range(T)) + const[s])
        m.addConstr(xi[s] >= eta - p[s])

    ep   = gp.quicksum(pi[s]*p[s]  for s in range(S))
    cvar = eta - (1/(1-alpha)) * gp.quicksum(pi[s]*xi[s] for s in range(S))
    m.setObjective((1-beta)*ep + beta*cvar, GRB.MAXIMIZE)
    m.optimize()

    q_opt    = np.array([q[t].X for t in range(T)])
    ep_val   = float(sum(pi[s]*p[s].X for s in range(S)))
    cvar_val = float(eta.X - (1/(1-alpha)) * sum(pi[s]*xi[s].X for s in range(S)))
    profits  = np.array([p[s].X for s in range(S)])
    return q_opt, ep_val, cvar_val, profits


def solve_two_price_cvar(scenarios, beta, alpha=ALPHA):
    T  = 24
    S  = len(scenarios)
    pi = np.array([s["prob"] for s in scenarios])

    m = gp.Model(); m.Params.OutputFlag = 0
    q  = m.addVars(T, lb=0.0, ub=CAPACITY)
    dp = m.addVars(S, T, lb=0.0, ub=CAPACITY)
    dm = m.addVars(S, T, lb=0.0, ub=CAPACITY)
    z  = m.addVars(S, T, vtype=GRB.BINARY)
    p  = m.addVars(S, lb=-GRB.INFINITY)
    eta = m.addVar(lb=-GRB.INFINITY)
    xi  = m.addVars(S, lb=0.0)

    for s, sc in enumerate(scenarios):
        expr = gp.LinExpr()
        for t in range(T):
            da = float(sc["price"][t]); bp = float(sc["bp"][t])
            si = int(sc["imbalance"][t]); w = float(sc["wind"][t])
            m.addConstr(dp[s,t] - dm[s,t] == w - q[t])
            m.addConstr(dp[s,t] <= CAPACITY * z[s,t])
            m.addConstr(dm[s,t] <= CAPACITY * (1 - z[s,t]))
            expr += da * q[t]
            if si == 1:
                expr += da*dp[s,t] - bp*dm[s,t]
            else:
                expr += bp*dp[s,t] - da*dm[s,t]
        m.addConstr(p[s] == expr)
        m.addConstr(xi[s] >= eta - p[s])

    ep   = gp.quicksum(pi[s]*p[s]  for s in range(S))
    cvar = eta - (1/(1-alpha)) * gp.quicksum(pi[s]*xi[s] for s in range(S))
    m.setObjective((1-beta)*ep + beta*cvar, GRB.MAXIMIZE)
    m.optimize()

    q_opt    = np.array([q[t].X for t in range(T)])
    ep_val   = float(sum(pi[s]*p[s].X for s in range(S)))
    cvar_val = float(eta.X - (1/(1-alpha)) * sum(pi[s]*xi[s].X for s in range(S)))
    profits  = np.array([p[s].X for s in range(S)])
    return q_opt, ep_val, cvar_val, profits


# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep_beta(scenarios, scheme="one"):
    solver = solve_one_price_cvar if scheme == "one" else solve_two_price_cvar
    results = []
    print(f"\nbeta sweep -- {scheme}-price  (alpha={ALPHA})")
    print(f"{'beta':>6}  {'E[profit]':>15}  {'CVaR':>15}")
    for beta in BETA_GRID:
        q_opt, ep, cvar, profits = solver(scenarios, beta)
        results.append({"beta": beta, "q": q_opt, "ep": ep,
                        "cvar": cvar, "profits": profits})
        print(f"{beta:>6.2f}  {ep:>15,.0f}  {cvar:>15,.0f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD 1 — beta sweep, offering strategy, profit variability
# ════════════════════════════════════════════════════════════════════════════

def plot_dashboard1(results_one, results_two):
    """
    5 rows x 2 cols. Left = one-price (blue), right = two-price (orange).
    Rows:
      A) E[profit] and CVaR vs beta
      B) 24-hour offer schedule q*[t] for each beta
      C) Per-scenario profit boxplot vs beta
      D) Profit CDF for each beta
      E) Profit histogram for each beta
    """
    fig, axes = plt.subplots(5, 2, figsize=(15, 22))

    schemes = [
        ("One-price", "#1f77b4", plt.cm.Blues,   results_one),
        ("Two-price", "#d35400", plt.cm.Oranges, results_two),
    ]

    for col, (label, color, cmap, results) in enumerate(schemes):
        betas    = [r["beta"] for r in results]
        eps      = np.array([r["ep"]   for r in results]) / 1000
        cvars    = np.array([r["cvar"] for r in results]) / 1000
        profits  = [r["profits"] / 1000 for r in results]
        qs       = [r["q"]       for r in results]
        shades   = np.linspace(0.4, 0.9, len(betas))
        hours    = np.arange(1, 25)

        # ── A: E[profit] and CVaR vs beta ────────────────────────────────────
        ax = axes[0, col]
        ax.plot(betas, eps,   color=color, lw=2.2, marker="o", ms=7, label="E[profit]")
        ax.plot(betas, cvars, color=color, lw=2.2, marker="s", ms=7,
                linestyle="--", markerfacecolor="white",
                markeredgewidth=1.8, label="CVaR_0.90")
        ax.set_xlabel("beta"); ax.set_ylabel("Value [kEUR]")
        ax.set_xticks(betas)
        ax.set_title(f"{label} — A) E[Profit] and CVaR vs beta")
        ax.legend(); ax.grid(alpha=0.3)

        # ── B: 24-hour offer schedule per beta ───────────────────────────────
        ax = axes[1, col]
        for shade, b, q in zip(shades, betas, qs):
            ax.step(hours, q, where="mid", color=cmap(shade),
                    lw=1.8, label=f"beta={b}")
        ax.set_xlabel("Hour"); ax.set_ylabel("DA Offer q* [MW]")
        ax.set_xticks(hours[::2])
        ax.set_title(f"{label} — B) Offer schedule q*[t] vs beta")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

        # ── C: Per-scenario profit boxplot vs beta ───────────────────────────
        ax = axes[2, col]
        bp = ax.boxplot(profits, positions=betas, widths=0.08,
                        patch_artist=True, showfliers=False)
        for patch, shade in zip(bp["boxes"], shades):
            patch.set_facecolor(cmap(shade)); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.5)
        ax.set_xlabel("beta"); ax.set_ylabel("Profit [kEUR]")
        ax.set_xticks(betas); ax.set_xticklabels([f"{b}" for b in betas])
        ax.set_title(f"{label} — C) Per-scenario profit spread vs beta")
        ax.grid(alpha=0.3, axis="y")

        # ── D: CDF of profits per beta ───────────────────────────────────────
        ax = axes[3, col]
        for shade, b, pr in zip(shades, betas, profits):
            sorted_p = np.sort(pr)
            cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
            ax.plot(sorted_p, cdf, color=cmap(shade), lw=2.0, label=f"beta={b}")
        ax.set_xlabel("Profit [kEUR]"); ax.set_ylabel("Cumulative probability")
        ax.set_title(f"{label} — D) Profit CDF per beta")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── E: Histogram of profits per beta ─────────────────────────────────
        ax = axes[4, col]
        all_p = np.concatenate(profits)
        bins  = np.linspace(all_p.min(), all_p.max(), 40)
        for shade, b, pr in zip(shades, betas, profits):
            ax.hist(pr, bins=bins, color=cmap(shade), alpha=0.45,
                    label=f"beta={b}", histtype="stepfilled", edgecolor=cmap(shade))
        ax.set_xlabel("Profit [kEUR]"); ax.set_ylabel("Count")
        ax.set_title(f"{label} — E) Profit histogram per beta")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        "Task 1.4 — Dashboard 1: Risk aversion (beta), offering strategy, profit variability",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig("results/task_1_4_dashboard1.png", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD 2 — in-sample sensitivity (temporal contiguous subsets)
# ════════════════════════════════════════════════════════════════════════════

def plot_dashboard2(scenarios, n_subsets=3, subset_size=200):
    """
    3 rows x 2 cols. Left = one-price, right = two-price.
    Rows:
      A) E[profit] vs beta overlaid across contiguous subsets
      B) CVaR vs beta overlaid across contiguous subsets
      C) Optimal offer q*[t] at beta=1 across subsets
    """
    subsets = contiguous_subsets(scenarios, n_subsets=n_subsets, subset_size=subset_size)
    N = len(scenarios)
    starts = np.linspace(0, N - subset_size, n_subsets, dtype=int)
    subset_labels = [f"Subset {i+1}: [{s}:{s+subset_size}]"
                     for i, s in enumerate(starts)]

    # Solve all subsets for both schemes
    print(f"\n--- Dashboard 2: solving {n_subsets} contiguous subsets x 2 schemes ---")
    all_results = {"one": [], "two": []}
    for scheme in ("one", "two"):
        for i, sub in enumerate(subsets):
            print(f"  scheme={scheme}  {subset_labels[i]}")
            all_results[scheme].append(sweep_beta(sub, scheme=scheme))

    fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    schemes = [
        ("One-price", "#1f77b4", plt.cm.Blues,   "one"),
        ("Two-price", "#d35400", plt.cm.Oranges, "two"),
    ]
    styles = ["-", "--", ":", "-."]

    for col, (label, color, cmap, key) in enumerate(schemes):
        results_per_subset = all_results[key]
        shades = np.linspace(0.45, 0.9, n_subsets)

        # ── A: E[profit] vs beta overlaid ────────────────────────────────────
        ax = axes[0, col]
        for i, results in enumerate(results_per_subset):
            betas = [r["beta"] for r in results]
            eps   = np.array([r["ep"] for r in results]) / 1000
            ax.plot(betas, eps, color=cmap(shades[i]), lw=2.0, marker="o",
                    ms=6, linestyle=styles[i % len(styles)],
                    label=subset_labels[i])
        ax.set_xlabel("beta"); ax.set_ylabel("E[profit] [kEUR]")
        ax.set_xticks(BETA_GRID)
        ax.set_title(f"{label} — A) E[Profit] vs beta across subsets")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── B: CVaR vs beta overlaid ─────────────────────────────────────────
        ax = axes[1, col]
        for i, results in enumerate(results_per_subset):
            betas = [r["beta"] for r in results]
            cvars = np.array([r["cvar"] for r in results]) / 1000
            ax.plot(betas, cvars, color=cmap(shades[i]), lw=2.0, marker="s",
                    ms=6, linestyle=styles[i % len(styles)],
                    markerfacecolor="white", label=subset_labels[i])
        ax.set_xlabel("beta"); ax.set_ylabel("CVaR_0.90 [kEUR]")
        ax.set_xticks(BETA_GRID)
        ax.set_title(f"{label} — B) CVaR vs beta across subsets")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # ── C: Optimal q*[t] at beta=1 across subsets ────────────────────────
        ax = axes[2, col]
        hours = np.arange(1, 25)
        idx_b1 = BETA_GRID.index(1.0)
        for i, results in enumerate(results_per_subset):
            q_b1 = results[idx_b1]["q"]
            ax.step(hours, q_b1, where="mid", color=cmap(shades[i]),
                    lw=2.0, linestyle=styles[i % len(styles)],
                    label=subset_labels[i])
        ax.set_xlabel("Hour"); ax.set_ylabel("DA Offer q* [MW]")
        ax.set_xticks(hours[::2])
        ax.set_title(f"{label} — C) Risk-averse offer (beta=1) across subsets")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        f"Task 1.4 — Dashboard 2: In-sample sensitivity "
        f"({n_subsets} contiguous temporal subsets, no shuffling)",
        fontsize=14, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig("results/task_1_4_dashboard2.png", dpi=150)
    plt.close()