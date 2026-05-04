"""
Step 1 – Task 1.4: Risk-Averse Offering Strategy (CVaR)
========================================================
Goal       : Reformulate the one-price and two-price offering problems
             as risk-averse stochastic programs, trading off expected
             profit against CVaR_alpha. Sweep the trade-off weight beta
             and plot the efficient frontier (E[profit] vs CVaR).

CVaR primer
-----------
For a profit r.v. pi with scenarios s = 1..S and probabilities p_s:
  - VaR_alpha(pi)  = alpha-quantile of profit from below; with alpha = 0.9
                     it is the threshold below which the worst 10% of
                     scenarios fall.
  - CVaR_alpha(pi) = average profit in the worst (1 - alpha) fraction of
                     scenarios. Higher CVaR  =>  better worst-case mass.

Rockafellar-Uryasev linear formulation
--------------------------------------
Introduce:
  zeta  in R          (free; equals VaR_alpha at the optimum)
  eta_s >= 0          (shortfall of scenario s below zeta)
Constraints:
  eta_s >= zeta - pi_s          for every scenario s
Then:
  CVaR_alpha(pi) = zeta - (1 / (1 - alpha)) * sum_s p_s * eta_s

Risk-averse objective (with weight beta in [0, 1]):
  max   (1 - beta) * E[pi]  +  beta * CVaR_alpha(pi)

beta = 0  -> pure expected-profit max (= Task 1.1 / 1.2)
beta = 1  -> pure CVaR max (most risk-averse)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Shared formatting (same style as step1_task_3.py)
# ---------------------------------------------------------------------------

# Y-axis formatter: turns 350000 into "350" so labels stay short
_K_FORMATTER  = mtick.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
_PROFIT_LABEL = "Expected Profit [k€]"
_CVAR_LABEL   = "CVaR_0.9 [k€]"

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def compute_cvar(profits, probs, alpha=0.90):
    """
    Sample CVaR_alpha from a realized profit vector.

    Used AFTER solving to verify / report the CVaR of the resulting profit
    distribution (the model's CVaR via zeta is an LP value; this is an
    independent re-computation directly from the profits).

    Parameters
    ----------
    profits : array-like (length S) of profit per scenario
    probs   : array-like (length S) of scenario probabilities (sum to 1)
    alpha   : confidence level (e.g. 0.90 -> worst 10% tail)

    Returns
    -------
    cvar  : conditional expected profit in the worst (1 - alpha) tail
    var   : the VaR (alpha-quantile of profit from below)
    """
    profits = np.asarray(profits, dtype=float)
    probs   = np.asarray(probs,   dtype=float)

    # Sort scenarios from worst to best profit (ascending).
    order   = np.argsort(profits)
    p_sorted = profits[order]
    w_sorted = probs[order]

    # Walk the sorted scenarios and accumulate probability mass until we
    # cross the (1 - alpha) tail boundary -> that gives us the VaR.
    tail_mass = 1.0 - alpha
    cum       = np.cumsum(w_sorted)
    # First scenario whose cumulative weight reaches the tail boundary
    idx       = np.searchsorted(cum, tail_mass)
    idx       = min(idx, len(p_sorted) - 1)
    var       = p_sorted[idx]

    # CVaR = expected profit conditional on being in the worst tail.
    # We average the tail scenarios weighted by their (rescaled) probabilities.
    tail_profits = p_sorted[: idx + 1]
    tail_weights = w_sorted[: idx + 1]
    # Re-normalise tail weights so they sum to 1 (conditional expectation).
    cvar = float(np.sum(tail_profits * tail_weights) / tail_weights.sum())

    return cvar, float(var)


# ---------------------------------------------------------------------------
# Risk-averse one-price scheme
# ---------------------------------------------------------------------------

def solve_one_price_cvar(scenarios, alpha=0.90, beta=0.0, capacity=500):
    """
    One-price LP with the CVaR-augmented Rockafellar-Uryasev objective.

    Decision variables
    ------------------
    q[t]    : DA offer at hour t                       (0 <= q[t] <= capacity)
    zeta    : VaR proxy (free)
    eta[s]  : shortfall of scenario s below zeta       (>= 0)

    Per-scenario profit (one-price scheme):
        pi_s = sum_t [ DA_t * q_t + BP^s_t * (W^s_t - q_t) ]

    Risk-averse objective:
        max  (1 - beta) * sum_s p_s * pi_s
           + beta * [ zeta  -  (1/(1-alpha)) * sum_s p_s * eta_s ]
    Constraints:
        eta_s >= zeta - pi_s   for every s
        eta_s >= 0

    Returns
    -------
    q_opt      : ndarray of length 24, optimal DA offers
    profits    : list of length S, realised profit per scenario
    exp_profit : float, sample mean of profits (= sum_s p_s * pi_s)
    cvar_val   : float, sample CVaR_alpha of profits
    var_val    : float, sample VaR_alpha (zeta at optimum, ish)
    """
    T    = 24
    S    = len(scenarios)
    prob = scenarios[0]["prob"]   # uniform; we assume all p_s equal

    m = gp.Model("one_price_cvar")
    m.setParam("OutputFlag", 0)

    # --- Decision variables ------------------------------------------------
    q    = m.addVars(T, lb=0.0, ub=capacity, name="q")
    # zeta is unbounded (free) - it is allowed to be negative.
    zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")
    eta  = m.addVars(S, lb=0.0, name="eta")

    # --- Profit expression per scenario (linear in q) ----------------------
    # We build a Gurobi LinExpr for each scenario so we can re-use it both in
    # the expected-profit term AND in the CVaR shortfall constraints.
    pi_expr = []
    for scen in scenarios:
        e = gp.LinExpr()
        for t in range(T):
            # Profit decomposes as DA revenue + balancing settlement:
            #   DA_t * q_t + BP_t * (W_t - q_t)
            # = (DA_t - BP_t) * q_t + BP_t * W_t
            e += (scen["price"][t] - scen["bp"][t]) * q[t]
            e += scen["bp"][t] * scen["wind"][t]
        pi_expr.append(e)

    # --- CVaR shortfall constraints: eta_s >= zeta - pi_s ------------------
    for s in range(S):
        m.addConstr(eta[s] >= zeta - pi_expr[s], name=f"cvar_short_{s}")

    # --- Objective ---------------------------------------------------------
    # Expected profit: sum_s p_s * pi_s
    exp_term = gp.LinExpr()
    for s in range(S):
        exp_term += prob * pi_expr[s]

    # CVaR term: zeta - (1/(1-alpha)) * sum_s p_s * eta_s
    cvar_term = gp.LinExpr()
    cvar_term += zeta
    coef = 1.0 / (1.0 - alpha)
    for s in range(S):
        cvar_term -= coef * prob * eta[s]

    m.setObjective((1.0 - beta) * exp_term + beta * cvar_term, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price CVaR model not optimal. Status: {m.status}")

    # --- Recover decisions and compute sample statistics -------------------
    q_opt = np.array([q[t].X for t in range(T)])

    # Re-compute realised profit per scenario from the fixed q_opt (this is
    # what would actually be earned scenario-by-scenario).
    profits = []
    for scen in scenarios:
        pi = 0.0
        for t in range(T):
            pi += scen["price"][t] * q_opt[t]
            pi += scen["bp"][t] * (scen["wind"][t] - q_opt[t])
        profits.append(pi)

    probs_arr  = np.full(S, prob)
    exp_profit = float(np.sum(probs_arr * profits))
    cvar_val, var_val = compute_cvar(profits, probs_arr, alpha=alpha)

    return q_opt, profits, exp_profit, cvar_val, var_val


# ---------------------------------------------------------------------------
# Risk-averse two-price scheme
# ---------------------------------------------------------------------------

def solve_two_price_cvar(scenarios, alpha=0.90, beta=0.0, capacity=500):
    """
    Two-price MILP with the CVaR-augmented objective.

    The MILP structure (binary z_{s,t} that splits deviations into positive /
    negative parts under the imbalance-state-dependent settlement) is identical
    to solve_two_price; we just bolt the CVaR constraints + objective on top.
    """
    T    = 24
    S    = len(scenarios)
    prob = scenarios[0]["prob"]

    m = gp.Model("two_price_cvar")
    m.setParam("OutputFlag", 0)

    # --- Decisions ---------------------------------------------------------
    q         = m.addVars(T, lb=0.0, ub=capacity, name="q")
    dev_plus  = m.addVars(S, T, lb=0.0, ub=capacity, name="dev_plus")
    dev_minus = m.addVars(S, T, lb=0.0, ub=capacity, name="dev_minus")
    z         = m.addVars(S, T, vtype=GRB.BINARY, name="z")

    zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")
    eta  = m.addVars(S, lb=0.0, name="eta")

    # --- Standard two-price physical / disjunctive constraints -------------
    # dev_plus_s,t - dev_minus_s,t = W_s,t - q_t  (signed deviation split)
    # z = 1  =>  dev_plus  active   (over-production)
    # z = 0  =>  dev_minus active   (under-production)
    for s, scen in enumerate(scenarios):
        for t in range(T):
            m.addConstr(dev_plus[s, t] - dev_minus[s, t]
                        == scen["wind"][t] - q[t])
            m.addConstr(dev_plus[s, t]  <= capacity * z[s, t])
            m.addConstr(dev_minus[s, t] <= capacity * (1 - z[s, t]))

    # --- Profit expression per scenario ------------------------------------
    # pi_s = sum_t [ DA*q_t + (state-dependent settlement on dev_plus/minus) ]
    #
    # imbalance == 1 (system DEFICIT, surplus from us is rewarded at DA):
    #     gain  = DA * dev_plus
    #     loss  = BP * dev_minus
    # imbalance == 0 (system SURPLUS, surplus from us only earns BP):
    #     gain  = BP * dev_plus
    #     loss  = DA * dev_minus
    pi_expr = []
    for s, scen in enumerate(scenarios):
        e = gp.LinExpr()
        for t in range(T):
            da = float(scen["price"][t])
            bp = float(scen["bp"][t])
            si = int(scen["imbalance"][t])

            # DA revenue from the offer
            e += da * q[t]

            if si == 1:   # system deficit
                e += da * dev_plus[s, t]
                e -= bp * dev_minus[s, t]
            else:         # system surplus
                e += bp * dev_plus[s, t]
                e -= da * dev_minus[s, t]
        pi_expr.append(e)

    # --- CVaR shortfall constraints ----------------------------------------
    for s in range(S):
        m.addConstr(eta[s] >= zeta - pi_expr[s], name=f"cvar_short_{s}")

    # --- Objective: (1-beta)*E[pi] + beta * CVaR ---------------------------
    exp_term = gp.LinExpr()
    for s in range(S):
        exp_term += prob * pi_expr[s]

    cvar_term = gp.LinExpr()
    cvar_term += zeta
    coef = 1.0 / (1.0 - alpha)
    for s in range(S):
        cvar_term -= coef * prob * eta[s]

    m.setObjective((1.0 - beta) * exp_term + beta * cvar_term, GRB.MAXIMIZE)
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price CVaR model not optimal. Status: {m.status}")

    q_opt = np.array([q[t].X for t in range(T)])

    # --- Recompute realised profits from fixed q_opt -----------------------
    # Mirrors compute_two_price_profits in step1_task_1_and_2.py
    profits = []
    for scen in scenarios:
        pi = 0.0
        for t in range(T):
            da  = float(scen["price"][t])
            bp  = float(scen["bp"][t])
            si  = int(scen["imbalance"][t])
            dev = scen["wind"][t] - q_opt[t]

            pi += da * q_opt[t]
            if si == 1:
                pi += da * max(dev, 0) - bp * max(-dev, 0)
            else:
                pi += bp * max(dev, 0) - da * max(-dev, 0)
        profits.append(pi)

    probs_arr  = np.full(S, prob)
    exp_profit = float(np.sum(probs_arr * profits))
    cvar_val, var_val = compute_cvar(profits, probs_arr, alpha=alpha)

    return q_opt, profits, exp_profit, cvar_val, var_val


# ---------------------------------------------------------------------------
# Beta sweep -> efficient frontier
# ---------------------------------------------------------------------------

def sweep_beta(scenarios, scheme="one", alpha=0.90, betas=None):
    """
    Solve the risk-averse problem for a sequence of beta values.

    Parameters
    ----------
    scenarios : list of scenario dicts with keys {price, bp, wind, imbalance, prob}
    scheme    : "one" or "two"
    alpha     : CVaR confidence level (e.g. 0.90)
    betas     : iterable of beta values in [0, 1]; default = a sensible grid

    Returns
    -------
    dict with arrays:
        betas, exp_profits, cvars, vars, q_offers (S, 24), profits (S, n_scen)
    where the leading dimension is over beta values.
    """
    if betas is None:
        # Coarse grid -- 5 points evenly spaced in [0, 1]. Cheap to run and
        # already enough to sketch the frontier shape. Increase density later
        # if a smoother curve is needed.
        betas = [0.0, 0.25, 0.5, 0.75, 1.0]

    if scheme == "one":
        solver = solve_one_price_cvar
        label  = "One-price"
    elif scheme == "two":
        solver = solve_two_price_cvar
        label  = "Two-price"
    else:
        raise ValueError(f"Unknown scheme '{scheme}', expected 'one' or 'two'.")

    print(f"\n{'='*65}")
    print(f"Beta sweep ({label}) | alpha = {alpha} | {len(betas)} values of beta")
    print(f"{'='*65}")
    print(f"{'beta':>6} {'E[profit]':>14} {'CVaR_a':>14} {'VaR_a':>14}")
    print("-" * 50)

    exp_profits = []
    cvar_vals   = []
    var_vals    = []
    q_offers    = []
    profits_all = []

    for beta in betas:
        q, profits, ep, cvar, var = solver(scenarios, alpha=alpha, beta=beta)
        exp_profits.append(ep)
        cvar_vals.append(cvar)
        var_vals.append(var)
        q_offers.append(q)
        profits_all.append(profits)
        print(f"{beta:>6.2f} {ep:>14,.0f} {cvar:>14,.0f} {var:>14,.0f}")

    return {
        "betas":       np.array(betas),
        "exp_profits": np.array(exp_profits),
        "cvars":       np.array(cvar_vals),
        "vars":        np.array(var_vals),
        "q_offers":    np.array(q_offers),       # shape (n_betas, 24)
        "profits":     np.array(profits_all),    # shape (n_betas, n_scen)
        "scheme":      scheme,
        "alpha":       alpha,
    }


# ---------------------------------------------------------------------------
# In-sample sensitivity
# ---------------------------------------------------------------------------

def assess_in_sample_sensitivity(scenario_subsets, scheme="one",
                                 alpha=0.90, betas=None):
    """
    Run the beta sweep on several different in-sample subsets and return
    the list of sweep results, so we can compare the frontiers and the
    offer schedules across in-sample draws.

    Parameters
    ----------
    scenario_subsets : list of list-of-scenarios; each inner list is one
                       in-sample subset to sweep beta on.
    """
    results = []
    for i, subset in enumerate(scenario_subsets):
        print(f"\n##### In-sample subset {i + 1}/{len(scenario_subsets)} "
              f"(N = {len(subset)}) #####")
        res = sweep_beta(subset, scheme=scheme, alpha=alpha, betas=betas)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_efficient_frontier(sweep_one, sweep_two,
                            save_path="results/task_1_4_frontier.png"):
    """
    Plot expected profit vs CVaR for both schemes on a single figure.
    Each point is one beta value; beta increases as we move down-left along
    the frontier (more risk aversion -> higher CVaR but lower E[profit]).
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))

    for sweep, color, marker, label in [
        (sweep_one, "#1f77b4", "o", "One-price"),
        (sweep_two, "#d35400", "s", "Two-price"),
    ]:
        ax.plot(sweep["cvars"], sweep["exp_profits"],
                color=color, linewidth=2.0, alpha=0.7)
        sc = ax.scatter(sweep["cvars"], sweep["exp_profits"],
                        c=sweep["betas"], cmap="viridis",
                        s=70, marker=marker, edgecolor=color, linewidth=1.5,
                        label=label, zorder=3)

    # Single shared colour bar -> shows which beta each point corresponds to.
    cb = fig.colorbar(sc, ax=ax, label=r"$\beta$ (risk aversion)")

    ax.set_xlabel(_CVAR_LABEL)
    ax.set_ylabel(_PROFIT_LABEL)
    ax.set_title("Task 1.4 – Efficient Frontier: Expected Profit vs CVaR_0.9\n"
                 r"(sweep $\beta \in [0,1]$, $\alpha = 0.90$)")
    ax.xaxis.set_major_formatter(_K_FORMATTER)
    ax.yaxis.set_major_formatter(_K_FORMATTER)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_offers_vs_beta(sweep, save_path=None):
    """
    Plot how the optimal hourly DA offer schedule q_t evolves as beta grows.
    Selects ~4 representative beta values to keep the plot readable.
    """
    if save_path is None:
        save_path = f"results/task_1_4_offers_{sweep['scheme']}.png"

    betas    = sweep["betas"]
    q_offers = sweep["q_offers"]

    # Pick a sparse subset of betas to plot (first, ~33%, ~67%, last).
    idx_sel = sorted({0, len(betas) // 3, 2 * len(betas) // 3, len(betas) - 1})
    cmap    = plt.cm.viridis(np.linspace(0.05, 0.95, len(idx_sel)))

    fig, ax = plt.subplots(figsize=(10, 5))
    hours = np.arange(1, 25)

    for color, idx in zip(cmap, idx_sel):
        ax.step(hours, q_offers[idx], where="mid",
                color=color, linewidth=2,
                label=fr"$\beta = {betas[idx]:.2f}$")

    label = "One-price" if sweep["scheme"] == "one" else "Two-price"
    ax.set_xlabel("Hour")
    ax.set_ylabel("DA offer q [MW]")
    ax.set_title(f"Task 1.4 – {label}: Optimal Offer Schedule vs Risk Aversion")
    ax.set_xticks(hours[::2])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_profit_distributions(sweep, save_path=None):
    """
    Overlay the realised profit histograms for a few representative betas.
    Visualises how the left tail gets clipped (and the right tail too) as
    risk aversion grows.
    """
    if save_path is None:
        save_path = f"results/task_1_4_profit_dist_{sweep['scheme']}.png"

    betas       = sweep["betas"]
    profits_all = sweep["profits"]

    idx_sel = sorted({0, len(betas) // 3, 2 * len(betas) // 3, len(betas) - 1})
    cmap    = plt.cm.viridis(np.linspace(0.05, 0.95, len(idx_sel)))

    # Common bins so the histograms are comparable across betas
    flat = np.concatenate([profits_all[i] for i in idx_sel])
    bins = np.linspace(flat.min(), flat.max(), 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, idx in zip(cmap, idx_sel):
        ax.hist(profits_all[idx], bins=bins, alpha=0.45,
                color=color,
                label=fr"$\beta = {betas[idx]:.2f}$  "
                      fr"(E={sweep['exp_profits'][idx]/1000:.0f}k, "
                      fr"CVaR={sweep['cvars'][idx]/1000:.0f}k)")

    label = "One-price" if sweep["scheme"] == "one" else "Two-price"
    ax.set_xlabel("Realised profit [€]")
    ax.set_ylabel("# scenarios")
    ax.set_title(f"Task 1.4 – {label}: Profit Distribution vs Risk Aversion")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sensitivity_frontiers(sensitivity_results, scheme_label,
                               save_path=None):
    """
    Overlay the efficient frontiers obtained from several different
    in-sample subsets. If the curves stack closely, the risk-averse
    solution is robust to the in-sample draw; if they spread, it isn't.
    """
    if save_path is None:
        save_path = f"results/task_1_4_sensitivity_{scheme_label}.png"

    fig, ax = plt.subplots(figsize=(8.5, 6))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(sensitivity_results)))

    for color, res in zip(cmap, sensitivity_results):
        ax.plot(res["cvars"], res["exp_profits"],
                marker="o", linewidth=1.8, color=color, alpha=0.8,
                label=f"N = {res['profits'].shape[1]}")

    ax.set_xlabel(_CVAR_LABEL)
    ax.set_ylabel(_PROFIT_LABEL)
    ax.set_title(f"Task 1.4 – {scheme_label}: Frontier Sensitivity to "
                 "In-sample Subset")
    ax.xaxis.set_major_formatter(_K_FORMATTER)
    ax.yaxis.set_major_formatter(_K_FORMATTER)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()