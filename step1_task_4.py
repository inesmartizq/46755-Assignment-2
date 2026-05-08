"""
Step 1 – Task 1.4: Risk-Averse Offering Strategy (CVaR)
========================================================
Formulate and solve the risk-averse offering strategy under both one- and
two-price schemes using CVaR with α = 0.90.

Objective : maximise  (1 - β) · E[profit] + β · CVaR_α[profit]
β = 0  →  pure profit maximisation  (reproduces Tasks 1.1 / 1.2)
β = 1  →  pure CVaR maximisation    (most risk-averse)

Scenario dict keys  (must match step1_scenario_generation.py)
-------------------------------------------------------------
  wind      : np.array shape (24,)  wind power [MW]
  price     : np.array shape (24,)  DA price   [EUR/MWh]
  imbalance : np.array shape (24,)  0=short system, 1=long system
  bp        : np.array shape (24,)  balancing price [EUR/MWh]
  prob      : float                 scenario probability

Profit formulas  (consistent with step1_task_1_and_2.py)
---------------------------------------------------------
One-price (per hour t):
  profit_t = price[t]*q[t] + bp[t]*(wind[t] - q[t])

Two-price (per hour t):
  dev = wind[t] - q[t]
  if imbalance[t] == 1:   # long system
      profit_t = price[t]*q[t] + price[t]*max(dev,0) - bp[t]*max(-dev,0)
  else:                    # short system
      profit_t = price[t]*q[t] + bp[t]*max(dev,0)   - price[t]*max(-dev,0)

Three deliverables
------------------
1. Efficient frontier  : E[profit] vs CVaR as β increases, both schemes.
2. Offer evolution     : how the 24-hour DA offer q* changes with β.
3. In-sample sensitivity: re-solve on three different 200-scenario subsets
   and overlay the frontiers to check solution stability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import gaussian_kde
import gurobipy as gp
from gurobipy import GRB

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Formatting helpers  (same style as Tasks 1.1–1.3)
# ---------------------------------------------------------------------------
# NOTE: all monetary values are divided by 1000 before plotting, so axes
# are already in k€. We use a plain comma formatter — NOT the /1000
# formatter — to avoid double-scaling.
_K_FORMATTER  = mtick.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")  # kept for non-monetary axes only
_COMMA_FMT    = mtick.FuncFormatter(lambda v, _: f"{v:,.0f}")          # for pre-divided k€ values
_PROFIT_LABEL = "Expected Profit [k€]"
_CVAR_LABEL   = "CVaR₀.₉₀ [k€]"
_BETA_GRID    = [0.0, 0.25, 0.5, 0.75, 1.0]

CAPACITY = 500   # MW — must match Tasks 1.1 / 1.2


# ---------------------------------------------------------------------------
# Helper: uniform probabilities  (same as Task 1.3)
# ---------------------------------------------------------------------------

def set_equal_probs(scenarios):
    """Return a new list of scenario dicts with uniform probabilities (1/N)."""
    n = len(scenarios)
    return [{**s, "prob": 1.0 / n} for s in scenarios]


# ---------------------------------------------------------------------------
# CVaR solvers
# ---------------------------------------------------------------------------

def solve_one_price_cvar(scenarios, beta, alpha=0.90):
    """
    Risk-averse one-price DA offering (LP).

    Per-scenario profit (summed over 24 hours):
        profit_s = Σ_t [ price_s[t]*q[t] + bp_s[t]*(wind_s[t] - q[t]) ]

    CVaR reformulation (Rockafellar & Uryasev 2000):
        CVaR_α = η - 1/(1-α) · Σ_s π_s · ξ_s
        s.t.  ξ_s ≥ η - profit_s,  ξ_s ≥ 0

    Objective:
        max  (1-β)·E[profit] + β·CVaR_α

    Parameters
    ----------
    scenarios : list of scenario dicts
    beta      : CVaR weight in [0, 1]
    alpha     : confidence level (default 0.90)

    Returns
    -------
    q_opt    : np.array (24,) optimal DA offer [MW]
    ep_val   : float  expected profit [EUR]
    cvar_val : float  CVaR_α [EUR]
    profits  : np.array (S,) per-scenario profit [EUR]
    """
    T  = 24
    S  = len(scenarios)
    pi = np.array([s["prob"] for s in scenarios])

    coeff = np.array([s["price"] - s["bp"] for s in scenarios])   # (S, T)
    const = np.array([s["bp"] @ s["wind"]  for s in scenarios])   # (S,)

    m = gp.Model("one_price_cvar")
    m.Params.OutputFlag = 0

    q   = m.addVars(T, lb=0.0, ub=CAPACITY, name="q")
    p   = m.addVars(S, lb=-GRB.INFINITY,    name="p")
    eta = m.addVar(lb=-GRB.INFINITY,         name="eta")
    xi  = m.addVars(S, lb=0.0,              name="xi")

    for s in range(S):
        m.addConstr(
            p[s] == gp.quicksum(coeff[s, t] * q[t] for t in range(T)) + const[s]
        )

    for s in range(S):
        m.addConstr(xi[s] >= eta - p[s])

    ep   = gp.quicksum(pi[s] * p[s]  for s in range(S))
    cvar = eta - (1.0 / (1.0 - alpha)) * gp.quicksum(pi[s] * xi[s] for s in range(S))

    m.setObjective((1.0 - beta) * ep + beta * cvar, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price CVaR model infeasible (β={beta})")

    q_opt    = np.array([q[t].X for t in range(T)])
    ep_val   = float(sum(pi[s] * p[s].X for s in range(S)))
    cvar_val = float(eta.X - (1.0 / (1.0 - alpha)) * sum(pi[s] * xi[s].X for s in range(S)))
    profits  = np.array([p[s].X for s in range(S)])

    return q_opt, ep_val, cvar_val, profits


def solve_two_price_cvar(scenarios, beta, alpha=0.90):
    """
    Risk-averse two-price DA offering (MILP).

    Per-scenario profit (summed over 24 hours):
        dev_t = wind_s[t] - q[t]
        if imbalance_s[t] == 1:
            profit_t = price_s[t]*q[t] + price_s[t]*max(dev,0) - bp_s[t]*max(-dev,0)
        else:
            profit_t = price_s[t]*q[t] + bp_s[t]*max(dev,0)   - price_s[t]*max(-dev,0)

    Binary variable z[s,t] = 1 if dev_t >= 0  (surplus).

    Parameters
    ----------
    scenarios : list of scenario dicts
    beta      : CVaR weight in [0, 1]
    alpha     : confidence level (default 0.90)

    Returns
    -------
    q_opt, ep_val, cvar_val, profits  (same as solve_one_price_cvar)
    """
    T  = 24
    S  = len(scenarios)
    pi = np.array([s["prob"] for s in scenarios])

    m = gp.Model("two_price_cvar")
    m.Params.OutputFlag = 0

    q         = m.addVars(T, lb=0.0, ub=CAPACITY, name="q")
    dev_plus  = m.addVars(S, T, lb=0.0, ub=CAPACITY, name="dp")
    dev_minus = m.addVars(S, T, lb=0.0, ub=CAPACITY, name="dm")
    z         = m.addVars(S, T, vtype=GRB.BINARY,    name="z")
    p         = m.addVars(S, lb=-GRB.INFINITY,        name="p")
    eta       = m.addVar(lb=-GRB.INFINITY,             name="eta")
    xi        = m.addVars(S, lb=0.0,                  name="xi")

    for s, sc in enumerate(scenarios):
        profit_expr = gp.LinExpr()
        for t in range(T):
            da  = float(sc["price"][t])
            bp  = float(sc["bp"][t])
            si  = int(sc["imbalance"][t])
            w   = float(sc["wind"][t])

            m.addConstr(dev_plus[s, t] - dev_minus[s, t] == w - q[t])
            m.addConstr(dev_plus[s, t]  <= CAPACITY * z[s, t])
            m.addConstr(dev_minus[s, t] <= CAPACITY * (1 - z[s, t]))

            profit_expr += da * q[t]

            if si == 1:   # long system
                profit_expr += da * dev_plus[s, t] - bp * dev_minus[s, t]
            else:         # short system
                profit_expr += bp * dev_plus[s, t] - da * dev_minus[s, t]

        m.addConstr(p[s] == profit_expr)

    for s in range(S):
        m.addConstr(xi[s] >= eta - p[s])

    ep   = gp.quicksum(pi[s] * p[s]  for s in range(S))
    cvar = eta - (1.0 / (1.0 - alpha)) * gp.quicksum(pi[s] * xi[s] for s in range(S))

    m.setObjective((1.0 - beta) * ep + beta * cvar, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price CVaR model infeasible (β={beta})")

    q_opt    = np.array([q[t].X for t in range(T)])
    ep_val   = float(sum(pi[s] * p[s].X for s in range(S)))
    cvar_val = float(eta.X - (1.0 / (1.0 - alpha)) * sum(pi[s] * xi[s].X for s in range(S)))
    profits  = np.array([p[s].X for s in range(S)])

    return q_opt, ep_val, cvar_val, profits


# ---------------------------------------------------------------------------
# Sweep β → efficient frontier data
# ---------------------------------------------------------------------------

def sweep_beta(scenarios, scheme="one", alpha=0.90, beta_grid=None):
    """
    Solve the CVaR offering problem for each β in beta_grid.

    Parameters
    ----------
    scenarios  : list of scenario dicts
    scheme     : "one" or "two"
    alpha      : CVaR confidence level (default 0.90)
    beta_grid  : list of β values

    Returns
    -------
    dict with keys: beta, ep, cvar, q, profits, scheme, alpha
    """
    if beta_grid is None:
        beta_grid = _BETA_GRID

    solver = solve_one_price_cvar if scheme == "one" else solve_two_price_cvar

    out = {"beta": [], "ep": [], "cvar": [], "q": [], "profits": [],
           "scheme": scheme, "alpha": alpha}

    print(f"\n{'='*60}")
    print(f"Sweeping β — {scheme}-price scheme  (α={alpha})")
    print(f"{'='*60}")
    print(f"{'β':>6} {'E[profit]':>14} {'CVaR':>14} {'q* mean [MW]':>14}")
    print("-" * 55)

    for beta in beta_grid:
        q, ep, cvar, profits = solver(scenarios, beta, alpha)
        out["beta"].append(beta)
        out["ep"].append(ep)
        out["cvar"].append(cvar)
        out["q"].append(q)
        out["profits"].append(profits)
        print(f"{beta:>6.2f} {ep:>14,.0f} {cvar:>14,.0f} {q.mean():>14.2f}")

    return out


# ---------------------------------------------------------------------------
# In-sample sensitivity
# ---------------------------------------------------------------------------

def assess_in_sample_sensitivity(subsets, scheme="one", alpha=0.90,
                                  beta_grid=None):
    """
    Run sweep_beta on each subset to check whether the efficient frontier
    changes significantly when different in-sample sets are used.

    Parameters
    ----------
    subsets   : list of scenario lists (e.g. three 200-scenario subsets)
    scheme    : "one" or "two"
    alpha     : CVaR confidence level
    beta_grid : β values to sweep

    Returns
    -------
    list of sweep result dicts (one per subset)
    """
    results = []
    for i, subset in enumerate(subsets):
        print(f"\n--- Subset {i+1}/{len(subsets)} ({len(subset)} scenarios) ---")
        normalised = set_equal_probs(subset)
        results.append(sweep_beta(normalised, scheme=scheme,
                                  alpha=alpha, beta_grid=beta_grid))
    return results


# ---------------------------------------------------------------------------
# Plot 1 – Efficient frontier: E[profit] vs CVaR
# ---------------------------------------------------------------------------

def plot_efficient_frontier(sweep_one, sweep_two):
    """
    Risk-return frontier for both schemes on the same axes.
    Each point = one β value. Moving right = more risk-averse.

    Values are pre-divided by 1000 before plotting → axes are in k€.
    _COMMA_FMT is used (no further division) to show clean numbers.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for sweep, color, label in [
        (sweep_one, "#1f77b4", "One-price"),
        (sweep_two, "#d35400", "Two-price"),
    ]:
        cvars = np.array(sweep["cvar"]) / 1000   # → k€
        eps   = np.array(sweep["ep"])   / 1000   # → k€
        betas = sweep["beta"]

        ax.plot(cvars, eps, marker="o", color=color, linewidth=2.2,
                markersize=7, label=label)

        ax.annotate("β=0",  (cvars[0],  eps[0]),
                    textcoords="offset points", xytext=(6,  4),
                    fontsize=8, color=color)
        ax.annotate("β=1",  (cvars[-1], eps[-1]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color=color)

        for i, b in enumerate(betas):
            if b in (0.25, 0.75):
                ax.annotate(f"β={b}", (cvars[i], eps[i]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7, color=color, alpha=0.8)

    ax.set_xlabel(_CVAR_LABEL)
    ax.set_ylabel(_PROFIT_LABEL)
    ax.xaxis.set_major_formatter(_COMMA_FMT)   # values already in k€
    ax.yaxis.set_major_formatter(_COMMA_FMT)
    ax.set_title(
        "Task 1.4 – Efficient Frontier: E[Profit] vs CVaR₀.₉₀\n"
        "(increasing β = increasing risk-aversion)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task_1_4_frontier.png", dpi=150)
    #plt.show()


# ---------------------------------------------------------------------------
# Plot 2 – Offer schedule and profit metrics vs β
# ---------------------------------------------------------------------------

def plot_offers_vs_beta(sweep):
    """
    Two-panel plot:
      Left  : 24-hour DA offer profile q*[t] for each β (step lines)
      Right : E[profit] and CVaR vs β

    Values pre-divided by 1000 on right panel → _COMMA_FMT, no re-division.
    """
    scheme = sweep["scheme"].capitalize()
    color  = "#1f77b4" if sweep["scheme"] == "one" else "#d35400"
    betas  = sweep["beta"]
    hours  = np.arange(1, 25)

    cmap   = plt.cm.Blues if sweep["scheme"] == "one" else plt.cm.Oranges
    shades = np.linspace(0.35, 0.85, len(betas))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: offer profiles per β (MW, no scaling needed)
    for shade, b, q in zip(shades, betas, sweep["q"]):
        ax1.step(hours, q, where="mid", color=cmap(shade),
                 linewidth=1.8, label=f"β={b}")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("DA Offer q* [MW]")
    ax1.set_title(f"{scheme}-price – Offer Profile vs β")
    ax1.legend(fontsize=7, ncol=2, loc="best")
    ax1.grid(alpha=0.3)

    # Right: E[profit] and CVaR vs β — pre-divide to k€ then use _COMMA_FMT
    eps   = np.array(sweep["ep"])   / 1000   # → k€
    cvars = np.array(sweep["cvar"]) / 1000   # → k€

    ax2.plot(betas, eps,   marker="o", color=color, linewidth=2.2,
             markersize=7, label="E[profit]")
    ax2.plot(betas, cvars, marker="s", color=color, linewidth=2.2,
             markersize=7, linestyle="--", markerfacecolor="white",
             markeredgewidth=1.8, label="CVaR₀.₉₀")
    ax2.set_xlabel("β (risk-aversion weight)")
    ax2.set_ylabel(_PROFIT_LABEL)
    ax2.yaxis.set_major_formatter(_COMMA_FMT)   # values already in k€
    ax2.set_title(f"{scheme}-price – E[Profit] and CVaR vs β")
    ax2.set_xticks(betas)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Task 1.4 – {scheme}-price Offering Strategy vs Risk Aversion",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"results/task_1_4_offers_beta_{sweep['scheme']}.png", dpi=150)
    #plt.show()


# ---------------------------------------------------------------------------
# Plot 3 – Profit distribution for selected β values
# ---------------------------------------------------------------------------

def plot_profit_distributions(sweep, betas_to_show=None):
    """
    KDE of per-scenario profit distribution for a few β values.
    Dotted vertical lines mark the corresponding CVaR level.

    Profits pre-divided to k€ → _COMMA_FMT on x-axis.
    """
    if betas_to_show is None:
        betas_to_show = [0.0, 0.5, 1.0]

    scheme = sweep["scheme"].capitalize()
    cmap   = plt.cm.Blues   if sweep["scheme"] == "one" else plt.cm.Oranges
    shades = np.linspace(0.4, 0.85, len(betas_to_show))

    fig, ax = plt.subplots(figsize=(9, 5))

    for shade, b in zip(shades, betas_to_show):
        if b not in sweep["beta"]:
            continue
        idx      = sweep["beta"].index(b)
        profits  = sweep["profits"][idx] / 1000   # → k€
        cvar_val = sweep["cvar"][idx]    / 1000   # → k€

        kde = gaussian_kde(profits, bw_method=0.3)
        x   = np.linspace(profits.min() - 10, profits.max() + 10, 300)
        ax.plot(x, kde(x), color=cmap(shade), linewidth=2.2, label=f"β={b}")
        ax.fill_between(x, kde(x), alpha=0.15, color=cmap(shade))
        ax.axvline(cvar_val, color=cmap(shade), linestyle=":", linewidth=1.4)

    ax.set_xlabel("Profit [k€]")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(_COMMA_FMT)   # values already in k€
    ax.set_title(
        f"Task 1.4 – {scheme}-price Profit Distribution for Selected β\n"
        "(dotted lines = CVaR₀.₉₀)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/task_1_4_distributions_{sweep['scheme']}.png", dpi=150)
    #plt.show()


# ---------------------------------------------------------------------------
# Plot 4 – In-sample sensitivity: overlay frontiers for different subsets
# ---------------------------------------------------------------------------

def plot_sensitivity_frontiers(sensitivity_results, scheme_label):
    """
    Overlay efficient frontiers from different in-sample subsets.
    A tight cluster = robust; wide spread = sensitive to IS choice.

    Values pre-divided to k€ → _COMMA_FMT on both axes.
    """
    color  = "#1f77b4" if "One" in scheme_label else "#d35400"
    styles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (sweep, style) in enumerate(zip(sensitivity_results, styles)):
        cvars = np.array(sweep["cvar"]) / 1000   # → k€
        eps   = np.array(sweep["ep"])   / 1000   # → k€
        ax.plot(cvars, eps, marker="o", color=color, linewidth=2.2,
                markersize=6, linestyle=style, label=f"Subset {i+1}")

    ax.set_xlabel(_CVAR_LABEL)
    ax.set_ylabel(_PROFIT_LABEL)
    ax.xaxis.set_major_formatter(_COMMA_FMT)   # values already in k€
    ax.yaxis.set_major_formatter(_COMMA_FMT)
    ax.set_title(
        f"Task 1.4 – {scheme_label} In-sample Sensitivity\n"
        "(each line = different 200-scenario IS subset)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"results/task_1_4_sensitivity_{scheme_label.split('-')[0].strip().lower()}.png"
    plt.savefig(fname, dpi=150)
    #plt.show()