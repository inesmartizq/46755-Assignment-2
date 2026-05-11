"""Entry point for Assignment 2: Runs all Step 1 and Step 2 tasks."""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves only, never shows

# ── Timing helper ─────────────────────────────────────────────────────────

TIMINGS = []   # list of (label, seconds)

def timed(label, func, *args, **kwargs):
    """Run func(*args, **kwargs), record wall-clock time under `label`."""
    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    TIMINGS.append((label, time.perf_counter() - t0))
    return out

# ── Step 1 imports ────────────────────────────────────────────────────────

from step1_scenario_generation import (
    dataframe_building,
    generate_combined_scenarios,
    plot_scenarios,
    analyze_scenarios,
)

from step1_task_1_and_2 import (
    solve_one_price,
    solve_two_price,
    plot_task_results,
    plot_offer_comparison,
    plot_profit_comparison,
)

from step1_task_3 import (
    run_cross_validation,
    run_vary_is_fixed_oos,
    plot_cv_per_fold,
    plot_cv_avg_comparison,
    plot_vary_is_fixed_oos,
)


from step1_task_4 import (
    set_equal_probs, sweep_beta,
    plot_offers_beta, plot_profit_distribution, plot_sensitivity_offer,
    print_table_metrics, print_table_sensitivity, run_sensitivity,
    plot_sensitivity_metrics,
    # ── NEW: Pareto plot, extended metrics table, hours 14/19 table ──
    plot_pareto,
    print_table_metrics_extended,
    print_table_hours_14_19,
)

# ── Step 2 imports ────────────────────────────────────────────────────────

from step2_scenario_generation import generate_and_save_profiles

from step2_task_1 import (
    load_in_sample_profiles,
    compute_flexibility,
    solve_alsox,
    solve_cvar,
)

from step2_task_2 import (
    load_profiles,
    compute_flexibility as compute_flexibility_oos,
    verify_p90_out_of_sample,
    plot_shortfall_comparison,
)

from step2_task_3 import run_sensitivity_analysis


# =========================================================================
# ── STEP 1 ───────────────────────────────────────────────────────────────
# =========================================================================

df = dataframe_building(
    price_file="data/energy_prices.csv",
    wind_file="data/wind_power_data.csv",
    price_date_col="MTU (CET/CEST)",
    wind_date_col="timestamp",
    price_col="Day-ahead Price (EUR/MWh)",
    wind_col="wind_power_mw",
)

scenarios = generate_combined_scenarios(df)
analyze_scenarios(scenarios)
plot_scenarios(scenarios)

# ── Tasks 1.1 and 1.2 ────────────────────────────────────────────────────

q1, ep1, profits1 = timed("Task 1.1 — One-price LP",  solve_one_price, scenarios)
print(f"\nOne-price  |  E[profit] = {ep1:,.2f} EUR")
print(f"Offers (MW): {q1.round(1)}")

q2, ep2, profits2 = timed("Task 1.2 — Two-price MILP", solve_two_price, scenarios)
print(f"\nTwo-price  |  E[profit] = {ep2:,.2f} EUR")
print(f"Offers (MW): {q2.round(1)}")

plot_task_results(q1, profits1, ep1, scenarios, "Task 1.1", "One-price offer", "steelblue")
plot_task_results(q2, profits2, ep2, scenarios, "Task 1.2", "Two-price offer", "darkorange", "--")
plot_offer_comparison(q1, q2, scenarios)
plot_profit_comparison(profits1, profits2)

# ── Task 1.3 – 8-fold cross-validation ───────────────────────────────────

cv_results = timed("Task 1.3 — 8-fold cross-validation",
                   run_cross_validation, scenarios, n_folds=8)
plot_cv_per_fold(cv_results)
plot_cv_avg_comparison(cv_results)

vary_results = timed(
    "Task 1.3 — Vary IS size (fixed OOS)",
    run_vary_is_fixed_oos,
    scenarios,
    is_sizes=[50, 100, 150, 200, 250, 300, 350, 400],
    n_folds=8,
)
plot_vary_is_fixed_oos(vary_results)

# ── Task 1.4 – Risk-averse offering (CVaR) ───────────────────────────────

scenarios_eq = set_equal_probs(scenarios)

# Sweep beta for both schemes (reused by every plot/table below)
results_one = timed("Task 1.4 — CVaR sweep (one-price)",
                    sweep_beta, scenarios_eq, scheme="one")
results_two = timed("Task 1.4 — CVaR sweep (two-price)",
                    sweep_beta, scenarios_eq, scheme="two")

# Existing offer-profile and profit-distribution plots
plot_offers_beta(results_one, scheme="one")
plot_offers_beta(results_two, scheme="two")
plot_profit_distribution(results_one, results_two)

# ── NEW: Pareto frontier plot (E[Π] vs CVaR) ────────────────────────────
# Fulfils the assignment's explicit "plot expected profit versus CVaR".
plot_pareto(results_one, results_two)

# ── NEW: extended metric table with sigma[Π] and Π_5% ───────────────────
# Replaces print_table_metrics. Prints plain + LaTeX, plus headline % changes
# and the two-price marginal substitution rates needed by the discussion.
print_table_metrics_extended(results_one, results_two)

# ── NEW: hours 14/19 de-risking table ───────────────────────────────────
# Per-beta offers at the two highest-price hours, plus mean of the rest.
print_table_hours_14_19(results_one, results_two)

# In-sample sensitivity (decision and metric level)
sens_results, sub_labels = timed(
    "Task 1.4 — IS sensitivity (3 subsets x 2 schemes)",
    run_sensitivity, scenarios_eq, n_subsets=3, subset_size=200,
)
plot_sensitivity_offer(sens_results, sub_labels)
print_table_sensitivity(sens_results, sub_labels)

sens_results, sub_labels = timed(
    "Task 1.4 — IS sensitivity (default subsets)",
    run_sensitivity, scenarios_eq,
)
plot_sensitivity_metrics(sens_results, sub_labels)
plot_sensitivity_offer(sens_results, sub_labels)   # keep this too

# =========================================================================
# ── STEP 2 ───────────────────────────────────────────────────────────────
# =========================================================================

print("\n" + "=" * 60)
print("RUNNING STEP 2: FCR-D UP CAPACITY PROVISION")
print("=" * 60)

# ── Scenario generation: generate CSVs if they don't exist yet ───────────

generate_and_save_profiles(seed=42)

# ── Task 2.1 – In-sample decision making ─────────────────────────────────

in_profiles = load_in_sample_profiles()
in_flex     = compute_flexibility(in_profiles)

res_alsox = timed("Task 2.1 — ALSO-X MILP", solve_alsox, in_flex, epsilon=0.10)
res_cvar  = timed("Task 2.1 — CVaR LP",     solve_cvar,  in_flex, epsilon=0.10)

print(f"\n--- Task 2.1: Optimal Reserve Bids (P90) ---")
print(f"  ALSO-X bid : {res_alsox:.2f} kW")
print(f"  CVaR bid   : {res_cvar:.2f} kW")

# ── Task 2.2 – Out-of-sample verification ────────────────────────────────

print(f"\n--- Task 2.2: Out-of-Sample Verification ---")

out_profiles = load_profiles("data/out_sample_profiles.csv")
out_flex     = compute_flexibility_oos(out_profiles)

sf_alsox = timed("Task 2.2 — OOS verification (ALSO-X)",
                 verify_p90_out_of_sample, res_alsox, out_profiles, "ALSO-X")
sf_cvar  = timed("Task 2.2 — OOS verification (CVaR)",
                 verify_p90_out_of_sample, res_cvar,  out_profiles, "CVaR")
plot_shortfall_comparison(sf_alsox, sf_cvar)

# ── Task 2.3 – Energinet perspective (sensitivity analysis) ──────────────

print(f"\n--- Task 2.3: Reliability vs. Capacity Trade-off ---")

P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]
timed("Task 2.3 — Reliability sensitivity sweep",
      run_sensitivity_analysis, in_flex, out_flex, P_LEVELS)

print("\nStep 2 execution complete. All figures saved to /results.")

# =========================================================================
# ── COMPUTATIONAL TIME TABLE ─────────────────────────────────────────────
# =========================================================================

def _fmt_time(s):
    if s < 1.0:
        return f"{s * 1000:>10,.1f} ms"
    if s < 60.0:
        return f"{s:>10,.3f} s "
    m, sec = divmod(s, 60)
    return f"{int(m):>5d}m {sec:>4.1f}s"

label_w = max(len(lbl) for lbl, _ in TIMINGS)
total   = sum(t for _, t in TIMINGS)

print("\n" + "=" * (label_w + 18))
print("COMPUTATIONAL TIME SUMMARY")
print("=" * (label_w + 18))
print(f"{'Task / Algorithm':<{label_w}}   {'Wall time':>12}")
print("-" * (label_w + 18))
for label, secs in TIMINGS:
    print(f"{label:<{label_w}}   {_fmt_time(secs)}")
print("-" * (label_w + 18))
print(f"{'TOTAL':<{label_w}}   {_fmt_time(total)}")
print("=" * (label_w + 18))