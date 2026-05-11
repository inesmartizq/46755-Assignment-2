"""
Step 1 - Task 1.4: Risk-Averse Offering Strategy (CVaR) — COMPLETE
==================================================================
This file contains every function imported by main.py. It is the merge
of the original task_1_4 script and the extension that adds:

  Plot 3 (pareto):  E[Π] vs CVaR Pareto frontier (both schemes)
          → fulfils the assignment's explicit "plot expected profit
            versus CVaR" requirement.

  Table A' (metrics_vs_beta extended): adds sigma[Π] and 5th-percentile
          profit per beta, both schemes.

  Table C (hours_14_19): per-beta offer at hours 14 and 19, plus mean
          across remaining 22 hours, both schemes.

Existing functions (kept unchanged from the original script):
  plot_offers_beta, plot_profit_distribution, plot_sensitivity_offer,
  plot_sensitivity_metrics, print_table_metrics, print_table_sensitivity,
  run_sensitivity, sweep_beta, set_equal_probs.

Tables are printed to console in plain + LaTeX format.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

plt.rcParams.update({
    "axes.titlesize": 21,
    "figure.titlesize": 27,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})

os.makedirs("results", exist_ok=True)

CAPACITY  = 500
ALPHA     = 0.90
BETA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_equal_probs(scenarios):
    n = len(scenarios)
    return [{**s, "prob": 1.0 / n} for s in scenarios]


def contiguous_subsets(scenarios, n_subsets=3, subset_size=200):
    """Non-overlapping contiguous blocks spread across the timeline (no shuffling)."""
    N = len(scenarios)
    if n_subsets * subset_size > N:
        raise ValueError(f"Need {n_subsets*subset_size} scenarios, only have {N}")
    starts = np.linspace(0, N - subset_size, n_subsets, dtype=int)
    subsets = [set_equal_probs(scenarios[s : s + subset_size]) for s in starts]
    labels  = [f"Subset {i+1}: [{s}:{s+subset_size}]" for i, s in enumerate(starts)]
    return subsets, labels


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

def sweep_beta(scenarios, scheme="one", verbose=True):
    solver = solve_one_price_cvar if scheme == "one" else solve_two_price_cvar
    results = []
    if verbose:
        print(f"\nbeta sweep -- {scheme}-price  (alpha={ALPHA})")
        print(f"{'beta':>6}  {'E[profit]':>15}  {'CVaR':>15}")
    for beta in BETA_GRID:
        q_opt, ep, cvar, profits = solver(scenarios, beta)
        results.append({"beta": beta, "q": q_opt, "ep": ep,
                        "cvar": cvar, "profits": profits})
        if verbose:
            print(f"{beta:>6.2f}  {ep:>15,.0f}  {cvar:>15,.0f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Helper: distributional statistics for a results dict
# ════════════════════════════════════════════════════════════════════════════

def profit_stats(results):
    """Return arrays of (beta, E[Π], CVaR, sigma[Π], Π_5%) across the sweep.
    All money values in EUR (not kEUR)."""
    betas  = np.array([r["beta"]  for r in results])
    eps    = np.array([r["ep"]    for r in results])
    cvars  = np.array([r["cvar"]  for r in results])
    sigmas = np.array([np.std(r["profits"], ddof=0)            for r in results])
    p5     = np.array([np.percentile(r["profits"], 5)          for r in results])
    return betas, eps, cvars, sigmas, p5


# ════════════════════════════════════════════════════════════════════════════
# PLOT 1 — DA offer profile vs beta (one figure per scheme, single panel)
# ════════════════════════════════════════════════════════════════════════════

def plot_offers_beta(results, scheme):
    """Single-panel: DA Offer profile for each beta."""
    label = "One-price" if scheme == "one" else "Two-price"
    cmap  = plt.cm.Blues if scheme == "one" else plt.cm.Oranges

    betas  = [r["beta"] for r in results]
    qs     = [r["q"] for r in results]
    shades = np.linspace(0.4, 0.9, len(betas))
    hours  = np.arange(1, 25)

    fig, ax = plt.subplots(figsize=(10, 5))
    for shade, b, q in zip(shades, betas, qs):
        ax.step(hours, q, where="mid", color=cmap(shade),
                lw=1.9, marker="o", ms=4, label=f"beta={b}")
    ax.set_xlabel("Hour")
    ax.set_ylabel(r"DA Offer $p^{DA,*}_t$ [MW]")
    ax.set_xticks(hours[::2])
    ax.axhline(CAPACITY, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_title(f"{label}: DA offer profile",
                 fontsize=21)
    ax.legend(fontsize=15, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/task_1_4_offers_beta_{scheme}.png", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Profit distribution: boxplot vs beta (both schemes side by side)
# ════════════════════════════════════════════════════════════════════════════

def plot_profit_distribution(results_one, results_two):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    schemes = [
        (axes[0], results_one, "One-price", plt.cm.Blues),
        (axes[1], results_two, "Two-price", plt.cm.Oranges),
    ]
    for ax, results, label, cmap in schemes:
        betas    = [r["beta"] for r in results]
        profits  = [r["profits"] / 1000 for r in results]
        shades   = np.linspace(0.4, 0.9, len(betas))

        bp = ax.boxplot(
            profits, positions=betas, widths=0.08,
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                            markerfacecolor="gray", markeredgecolor="none"),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
        )
        for patch, shade in zip(bp["boxes"], shades):
            patch.set_facecolor(cmap(shade))
            patch.set_alpha(0.75)

        ax.set_xlabel("beta")
        ax.set_ylabel("Per-scenario profit [kEUR]")
        ax.set_xticks(betas)
        ax.set_xticklabels([f"{b}" for b in betas])
        ax.set_xlim(-0.1, 1.1)
        ax.set_title(f"{label} — profit distribution vs beta")
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Profit distribution vs risk aversion",
                 fontsize=26)
    plt.tight_layout()
    plt.savefig("results/task_1_4_profit_dist.png", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Pareto frontier: E[Π] vs CVaR (both schemes)
# ════════════════════════════════════════════════════════════════════════════

def plot_pareto(results_one, results_two):
    """
    Risk-return Pareto frontier: x = CVaR, y = E[Π].
    Arrows indicate the direction of increasing beta.
    """

    fig, ax = plt.subplots(figsize=(8.5, 6))

    for results, label, cmap, marker in [
        (results_one, "One-price", plt.cm.Blues,   "o"),
        (results_two, "Two-price", plt.cm.Oranges, "s"),
    ]:
        betas, eps, cvars, _, _ = profit_stats(results)

        # plot in kEUR for readability
        x = cvars / 1000.0
        y = eps   / 1000.0

        # connecting line
        ax.plot(
            x, y,
            color=cmap(0.7),
            lw=1.8,
            alpha=0.9,
            zorder=2
        )

        # points with beta-shaded markers
        shades = np.linspace(0.4, 0.95, len(betas))

        for i, (xi_, yi_, sh, b) in enumerate(zip(x, y, shades, betas)):

            ax.scatter(
                xi_, yi_,
                s=20,
                color=cmap(sh),
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
                label=label if i == 0 else None,
            )

            # ---- smarter annotations to avoid overlap ----
            if label == "One-price":
                # left-most point: beta = 0
                if np.isclose(b, 0.0):
                    offset = (6, 8)
                    text = r"$\beta = 0$"

                # right-most point: collapse all others
                else:
                    offset = (6, 8)
                    text = r"$\beta \neq 0$"

            else:
                # spread labels vertically for readability
                if np.isclose(b, 0.0):
                    offset = (6, 8)
                elif np.isclose(b, 0.25):
                    offset = (6, 4)
                elif np.isclose(b, 0.5):
                    offset = (6, -2)
                elif np.isclose(b, 0.75):
                    offset = (6, -8)
                else:  # beta = 1
                    offset = (6, -14)

                text = rf"$\beta = {b}$"

            ax.annotate(
                text,
                (xi_, yi_),
                xytext=offset,
                textcoords="offset points",
                fontsize=8.5,
                color="black",
                alpha=0.9,
            )

        # arrow on the connecting line (direction of increasing beta)
        if len(x) >= 2:
            mid = len(x) // 2

            ax.annotate(
                "",
                xy=(
                    x[mid + 1] if mid + 1 < len(x) else x[mid],
                    y[mid + 1] if mid + 1 < len(y) else y[mid],
                ),
                xytext=(x[mid], y[mid]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=cmap(0.85),
                    lw=2,
                ),
                zorder=4,
            )

    ax.set_xlabel(r"$\mathrm{CVaR}_{0.90}[\Pi]$  [kEUR]")
    ax.set_ylabel(r"Expected profit $\mathbb{E}[\Pi]$  [kEUR]")

    ax.set_title(
        r"Risk-return Pareto frontier: "
        r"$\mathbb{E}[\Pi]$ vs $\mathrm{CVaR}_{0.90}$",
        fontsize=21,
    )

    # cleaner legend
    ax.legend(loc="best", fontsize=15, frameon=True)

    # slightly softer grid
    ax.grid(alpha=0.3)

    # add a bit of padding around data
    ax.margins(x=0.08, y=0.08)

    plt.tight_layout()
    plt.savefig("results/task_1_4_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PLOT 4 — In-sample sensitivity: risk-averse DA Offer (beta=1) across subsets
# ════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_offer(all_results, sub_labels):
    idx_b1 = BETA_GRID.index(1.0)
    hours  = np.arange(1, 25)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    schemes = [
        (axes[0], "one", "One-price", plt.cm.Blues),
        (axes[1], "two", "Two-price", plt.cm.Oranges),
    ]
    styles = ["-", "--", ":"]

    for ax, key, label, cmap in schemes:
        shades = np.linspace(0.5, 0.9, len(all_results[key]))
        for i, results in enumerate(all_results[key]):
            q_b1 = results[idx_b1]["q"]
            ax.step(hours, q_b1, where="mid", color=cmap(shades[i]),
                    lw=2.0, linestyle=styles[i % len(styles)],
                    label=sub_labels[i])
        ax.set_xlabel("Hour")
        ax.set_ylabel(r"DA Offer $p^{DA,*}_t$ [MW]")
        ax.set_xticks(hours[::2])
        ax.axhline(CAPACITY, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(label)
        ax.legend(fontsize=15); ax.grid(alpha=0.3)

    plt.suptitle(
        r"Risk-averse DA offer (β=1)",
        fontsize=26
    )
    plt.tight_layout()
    plt.savefig("results/task_1_4_sensitivity_q.png", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PLOT 5 — In-sample sensitivity: metrics vs beta across subsets
# ════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_metrics(all_results, sub_labels):
    """
    Single figure, 1x2 grid (schemes side by side).
    Per panel: E[profit] (solid) and CVaR (dashed), one curve per subset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    schemes = [
        (axes[0], "one", "One-price", plt.cm.Blues),
        (axes[1], "two", "Two-price", plt.cm.Oranges),
    ]

    for ax, key, label, cmap in schemes:
        n_sub  = len(all_results[key])
        shades = np.linspace(0.5, 0.92, n_sub)

        for i, results in enumerate(all_results[key]):
            betas = [r["beta"] for r in results]
            eps   = np.array([r["ep"]   for r in results]) / 1000
            cvars = np.array([r["cvar"] for r in results]) / 1000
            color = cmap(shades[i])

            ax.plot(betas, eps,   color=color, lw=2.0, marker="o", ms=6,
                    linestyle="-",  label=f"Subset {i+1} — E[Π]")
            ax.plot(betas, cvars, color=color, lw=2.0, marker="s", ms=6,
                    linestyle="--", markerfacecolor="white",
                    markeredgewidth=1.5, label=f"Subset {i+1} — CVaR")

        ax.set_xlabel("beta")
        ax.set_ylabel("Value [kEUR]")
        ax.set_xticks(BETA_GRID)
        ax.set_title(f"{label}")
        ax.legend(fontsize=15, ncol=1, loc="best")
        ax.grid(alpha=0.3)

    plt.suptitle(
        "E[Π] and CVaR vs β",
        fontsize=26
    )
    plt.tight_layout()
    plt.savefig("results/task_1_4_sensitivity_metrics.png", dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# TABLE A — Metrics vs beta (original; kept for backward compatibility)
# ════════════════════════════════════════════════════════════════════════════

def print_table_metrics(results_one, results_two):
    """E[profit] and CVaR_0.90 per beta, both schemes. Plain + LaTeX."""
    print("\n" + "="*72)
    print("TABLE A — E[profit] and CVaR_0.90 per beta  (kEUR, alpha=0.90)")
    print("="*72)
    print(f"{'beta':>6} | {'One E[π]':>12} {'One CVaR':>12} | "
          f"{'Two E[π]':>12} {'Two CVaR':>12}")
    print("-"*72)
    for r1, r2 in zip(results_one, results_two):
        print(f"{r1['beta']:>6.2f} | "
              f"{r1['ep']/1000:>12,.1f} {r1['cvar']/1000:>12,.1f} | "
              f"{r2['ep']/1000:>12,.1f} {r2['cvar']/1000:>12,.1f}")

    # LaTeX
    print("\n--- LaTeX ---")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{E[$\pi$] and CVaR$_{0.90}$ vs.\ $\beta$ for both schemes (kEUR).}")
    print(r"\begin{tabular}{c|cc|cc}")
    print(r"\toprule")
    print(r"$\beta$ & One E[$\pi$] & One CVaR & Two E[$\pi$] & Two CVaR \\")
    print(r"\midrule")
    for r1, r2 in zip(results_one, results_two):
        print(f"{r1['beta']:.2f} & "
              f"{r1['ep']/1000:,.1f} & {r1['cvar']/1000:,.1f} & "
              f"{r2['ep']/1000:,.1f} & {r2['cvar']/1000:,.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ════════════════════════════════════════════════════════════════════════════
# EXTENDED TABLE A' — Metrics vs beta with sigma and 5th percentile
# ════════════════════════════════════════════════════════════════════════════

def print_table_metrics_extended(results_one, results_two):
    """E[profit], CVaR_0.90, sigma[Π], and 5th-percentile profit per beta,
    both schemes. Plain + LaTeX."""
    b1, ep1, cv1, sg1, p5_1 = profit_stats(results_one)
    b2, ep2, cv2, sg2, p5_2 = profit_stats(results_two)

    print("\n" + "="*100)
    print("TABLE A (extended) — E[Π], CVaR_0.90, sigma[Π], Π_5% per beta  "
          "(kEUR, alpha=0.90)")
    print("="*100)
    print(f"{'beta':>6} | "
          f"{'One E[Π]':>10} {'One CVaR':>10} {'One sigma':>10} {'One Π_5%':>10} | "
          f"{'Two E[Π]':>10} {'Two CVaR':>10} {'Two sigma':>10} {'Two Π_5%':>10}")
    print("-"*100)
    for i in range(len(b1)):
        print(f"{b1[i]:>6.2f} | "
              f"{ep1[i]/1000:>10,.1f} {cv1[i]/1000:>10,.1f} "
              f"{sg1[i]/1000:>10,.1f} {p5_1[i]/1000:>10,.1f} | "
              f"{ep2[i]/1000:>10,.1f} {cv2[i]/1000:>10,.1f} "
              f"{sg2[i]/1000:>10,.1f} {p5_2[i]/1000:>10,.1f}")

    # Headline % changes (β=0 → β=1) for the discussion paragraph
    print("\n--- Headline changes (beta=0 -> beta=1) ---")
    for name, ep, cv, sg, p5 in [("One-price", ep1, cv1, sg1, p5_1),
                                  ("Two-price", ep2, cv2, sg2, p5_2)]:
        d_ep = (ep[-1]-ep[0])/ep[0]*100 if ep[0] != 0 else float("nan")
        d_cv = (cv[-1]-cv[0])/cv[0]*100 if cv[0] != 0 else float("nan")
        d_sg = (sg[-1]-sg[0])/sg[0]*100 if sg[0] != 0 else float("nan")
        d_p5 = (p5[-1]-p5[0])/p5[0]*100 if p5[0] != 0 else float("nan")
        print(f"  {name:>10}:  E[Π] {d_ep:+6.1f}%   "
              f"CVaR {d_cv:+6.1f}%   sigma {d_sg:+6.1f}%   "
              f"Π_5% {d_p5:+6.1f}%")

    # Marginal substitution rate |ΔCVaR / ΔE[Π]| between adjacent β points
    print("\n--- Two-price marginal substitution rate |ΔCVaR / ΔE[Π]| ---")
    for i in range(1, len(b2)):
        d_ep = ep2[i] - ep2[i-1]
        d_cv = cv2[i] - cv2[i-1]
        msr  = abs(d_cv / d_ep) if d_ep != 0 else float("inf")
        print(f"  beta {b2[i-1]:.2f} -> {b2[i]:.2f}:   "
              f"dE[Π] = {d_ep/1000:+8,.1f} kEUR   "
              f"dCVaR = {d_cv/1000:+8,.1f} kEUR   "
              f"MSR = {msr:6.2f}")

    # LaTeX
    print("\n--- LaTeX ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{E[$\Pi$], CVaR$_{0.90}$, profit standard deviation "
          r"$\sigma[\Pi]$, and 5\textsuperscript{th}-percentile profit "
          r"$\Pi_{5\%}$ vs.\ $\beta$ for both schemes (kEUR).}")
    print(r"\label{tab:1_4_metrics_extended}")
    print(r"\small")
    print(r"\begin{tabular}{c|cccc|cccc}")
    print(r"\toprule")
    print(r" & \multicolumn{4}{c|}{\textbf{One-price}} "
          r"& \multicolumn{4}{c}{\textbf{Two-price}} \\")
    print(r"$\beta$ & E[$\Pi$] & CVaR & $\sigma[\Pi]$ & $\Pi_{5\%}$ "
          r"& E[$\Pi$] & CVaR & $\sigma[\Pi]$ & $\Pi_{5\%}$ \\")
    print(r"\midrule")
    for i in range(len(b1)):
        print(f"{b1[i]:.2f} & "
              f"{ep1[i]/1000:,.1f} & {cv1[i]/1000:,.1f} & "
              f"{sg1[i]/1000:,.1f} & {p5_1[i]/1000:,.1f} & "
              f"{ep2[i]/1000:,.1f} & {cv2[i]/1000:,.1f} & "
              f"{sg2[i]/1000:,.1f} & {p5_2[i]/1000:,.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ════════════════════════════════════════════════════════════════════════════
# TABLE B — In-sample sensitivity: metrics across subsets
# ════════════════════════════════════════════════════════════════════════════

def print_table_sensitivity(all_results, sub_labels):
    """E[profit] and CVaR per beta across subsets, both schemes. Plain + LaTeX."""
    print("\n" + "="*78)
    print("TABLE B — In-sample sensitivity: metrics across contiguous subsets (kEUR)")
    print("="*78)

    for scheme_key, scheme_label in [("one", "One-price"), ("two", "Two-price")]:
        print(f"\n  {scheme_label}:")
        header = f"  {'beta':>6} | " + " | ".join(
            f"{'S'+str(i+1)+' E[π]':>10} {'S'+str(i+1)+' CVaR':>10}"
            for i in range(len(sub_labels))
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for bi, beta in enumerate(BETA_GRID):
            row = f"  {beta:>6.2f} | "
            cells = []
            for results in all_results[scheme_key]:
                ep   = results[bi]["ep"]   / 1000
                cvar = results[bi]["cvar"] / 1000
                cells.append(f"{ep:>10,.1f} {cvar:>10,.1f}")
            print(row + " | ".join(cells))

        # cross-subset spread for stability commentary
        print(f"\n  {scheme_label} — across-subset stats:")
        print(f"  {'beta':>6} | {'std E[π]':>10} {'std CVaR':>10} "
              f"{'rng E[π]':>10} {'rng CVaR':>10}")
        print("  " + "-" * 54)
        for bi, beta in enumerate(BETA_GRID):
            eps   = np.array([r[bi]["ep"]   for r in all_results[scheme_key]]) / 1000
            cvars = np.array([r[bi]["cvar"] for r in all_results[scheme_key]]) / 1000
            print(f"  {beta:>6.2f} | "
                  f"{eps.std():>10,.2f} {cvars.std():>10,.2f} "
                  f"{(eps.max()-eps.min()):>10,.2f} "
                  f"{(cvars.max()-cvars.min()):>10,.2f}")

    # LaTeX (compact: one block per scheme)
    print("\n--- LaTeX ---")
    for scheme_key, scheme_label in [("one", "One-price"), ("two", "Two-price")]:
        n = len(sub_labels)
        col_spec = "c|" + "|".join(["cc"] * n)
        print(r"\begin{table}[h]")
        print(r"\centering")
        print(rf"\caption{{In-sample sensitivity ({scheme_label}): "
              r"E[$\pi$] and CVaR$_{0.90}$ across contiguous subsets (kEUR).}}")
        print(rf"\begin{{tabular}}{{{col_spec}}}")
        print(r"\toprule")
        head = r"$\beta$"
        for i in range(n):
            head += f" & S{i+1} E[$\\pi$] & S{i+1} CVaR"
        print(head + r" \\")
        print(r"\midrule")
        for bi, beta in enumerate(BETA_GRID):
            row = f"{beta:.2f}"
            for results in all_results[scheme_key]:
                ep   = results[bi]["ep"]   / 1000
                cvar = results[bi]["cvar"] / 1000
                row += f" & {ep:,.1f} & {cvar:,.1f}"
            print(row + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")


# ════════════════════════════════════════════════════════════════════════════
# TABLE C — Hours 14 and 19 de-risking
# ════════════════════════════════════════════════════════════════════════════

def print_table_hours_14_19(results_one, results_two):
    """Per-beta offer at hours 14 and 19, plus mean of remaining 22 hours,
    both schemes. Plain + LaTeX."""
    HOUR_A, HOUR_B = 14, 19
    print("\n" + "="*90)
    print(f"TABLE C — De-risking of high-price hours "
          f"({HOUR_A} and {HOUR_B}) across beta  (MW)")
    print("="*90)

    for name, results in [("One-price", results_one), ("Two-price", results_two)]:
        print(f"\n  {name}:")
        print(f"  {'beta':>6} | {'q_14':>8} {'q_19':>8} "
              f"{'mean all':>10} {'mean rest':>11}")
        print("  " + "-"*46)
        for r in results:
            q = r["q"]
            q14 = q[HOUR_A - 1]
            q19 = q[HOUR_B - 1]
            mean_all = float(np.mean(q))
            mask = np.ones(24, dtype=bool); mask[HOUR_A-1] = False; mask[HOUR_B-1] = False
            mean_rest = float(np.mean(q[mask]))
            print(f"  {r['beta']:>6.2f} | {q14:>8.1f} {q19:>8.1f} "
                  f"{mean_all:>10.1f} {mean_rest:>11.1f}")

    # LaTeX (two-price; one-price is typically degenerate)
    print("\n--- LaTeX (Two-price; one-price typically degenerate) ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{De-risking of the two highest-price hours under "
          r"increasing risk aversion (two-price scheme, "
          r"$p^{DA,*}_t$ in MW). \emph{Mean rest} is the mean offer over "
          r"the 22 hours other than 14 and 19.}")
    print(r"\label{tab:hours_14_19}")
    print(r"\small")
    print(r"\begin{tabular}{c|cccc}")
    print(r"\toprule")
    print(r"$\beta$ & $p^{DA,*}_{14}$ & $p^{DA,*}_{19}$ "
          r"& Mean offer & Mean of remaining 22 hr \\")
    print(r"\midrule")
    for r in results_two:
        q = r["q"]
        q14 = q[HOUR_A - 1]
        q19 = q[HOUR_B - 1]
        mean_all = float(np.mean(q))
        mask = np.ones(24, dtype=bool); mask[HOUR_A-1] = False; mask[HOUR_B-1] = False
        mean_rest = float(np.mean(q[mask]))
        print(f"{r['beta']:.2f} & {q14:.1f} & {q19:.1f} & "
              f"{mean_all:.1f} & {mean_rest:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ════════════════════════════════════════════════════════════════════════════
# Driver for sensitivity (solves once, used by Plot 4 + Plot 5 + Table B)
# ════════════════════════════════════════════════════════════════════════════

def run_sensitivity(scenarios, n_subsets=3, subset_size=200):
    subsets, sub_labels = contiguous_subsets(scenarios, n_subsets, subset_size)
    print(f"\n--- Sensitivity: solving {n_subsets} contiguous subsets x 2 schemes ---")
    all_results = {"one": [], "two": []}
    for scheme in ("one", "two"):
        for i, sub in enumerate(subsets):
            print(f"  scheme={scheme}  {sub_labels[i]}")
            all_results[scheme].append(sweep_beta(sub, scheme=scheme, verbose=False))
    return all_results, sub_labels