"""Entry point for Assignment 2: Runs all Step 1 and Step 2 tasks."""

# ── Step 1 imports ────────────────────────────────────────────────────

from step1_scenario_generation import dataframe_building, generate_combined_scenarios, plot_scenarios, analyze_scenarios
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
    sweep_beta,
    assess_in_sample_sensitivity,
    plot_efficient_frontier,
    plot_offers_vs_beta,
    plot_profit_distributions,
    plot_sensitivity_frontiers,
)

# ── Step 2 imports ────────────────────────────────────────────────────

from step2_task_1 import (
    load_in_sample_profiles, 
    compute_flexibility, 
    solve_alsox, 
    solve_cvar
)

from step2_task_2 import (
    load_profiles, 
    verify_p90_out_of_sample
)

from step2_task_3 import run_sensitivity_analysis


df = dataframe_building(
    price_file="data/energy_prices.csv",
    wind_file="data/wind_power_data.csv",
    price_date_col="MTU (CET/CEST)",
    wind_date_col="timestamp",
    price_col="Day-ahead Price (EUR/MWh)",
    wind_col="wind_power_mw"
)


scenarios = generate_combined_scenarios(df)
analyze_scenarios(scenarios)
plot_scenarios(scenarios)

# ── Tasks 1.1 and 1.2 ────────────────────────────────────────────────────
q1, ep1, profits1 = solve_one_price(scenarios)
print(f"\nOne-price  |  E[profit] = {ep1:,.2f} EUR")
print(f"Offers (MW): {q1.round(1)}")

q2, ep2, profits2 = solve_two_price(scenarios)
print(f"\nTwo-price  |  E[profit] = {ep2:,.2f} EUR")
print(f"Offers (MW): {q2.round(1)}")

plot_task_results(q1, profits1, ep1, scenarios, "Task 1.1", "One-price offer", "steelblue")
plot_task_results(q2, profits2, ep2, scenarios, "Task 1.2", "Two-price offer", "darkorange", "--")
plot_offer_comparison(q1, q2, scenarios)
plot_profit_comparison(profits1, profits2)

# ── Task 1.3 – 8-fold cross-validation ───────────────────────────────────
cv_results = run_cross_validation(scenarios, n_folds=8)
plot_cv_per_fold(cv_results)
plot_cv_avg_comparison(cv_results)

# IS sizes from 50 to 400 with 8-fold CV per size, evaluated on the same
# fixed OOS block (scenarios 1000–1600).
vary_results = run_vary_is_fixed_oos(
    scenarios,
    is_sizes=[50, 100, 150, 200, 250, 300, 350, 400],
    n_folds=8,
)
plot_vary_is_fixed_oos(vary_results)

# ── Task 1.4 – Risk-averse offering (CVaR) ───────────────────────────────
# 1) Build the efficient frontier on the full 1,600-scenario set for both
#    schemes. Default beta grid is [0, 0.25, 0.5, 0.75, 1.0] (coarse).
sweep_one = sweep_beta(scenarios, scheme="one", alpha=0.90)
sweep_two = sweep_beta(scenarios, scheme="two", alpha=0.90)

plot_efficient_frontier(sweep_one, sweep_two)

# 2) How the optimal offer schedule and profit distribution evolve with β.
plot_offers_vs_beta(sweep_one)
plot_offers_vs_beta(sweep_two)
plot_profit_distributions(sweep_one)
plot_profit_distributions(sweep_two)

# 3) In-sample sensitivity: re-sweep on three contiguous 200-scenario
#    subsets and overlay the frontiers. Small subsets keep the two-price
#    MILP fast.
subsets = [scenarios[0:200], scenarios[700:900], scenarios[1400:1600]]
sens_one = assess_in_sample_sensitivity(subsets, scheme="one", alpha=0.90)
sens_two = assess_in_sample_sensitivity(subsets, scheme="two", alpha=0.90)
plot_sensitivity_frontiers(sens_one, "One-price")
plot_sensitivity_frontiers(sens_two, "Two-price")


# ── STEP 2 ───────────────────────────────

# =========================================================================
# ── STEP 2: PARTICIPATION IN ANCILLARY SERVICE MARKETS ───────────────────
# =========================================================================

# ── Task 2.1: In-sample Decision Making ──────────────────────────────────
print("\n" + "="*50)
print("RUNNING STEP 2: FCR-D UP CAPACITY PROVISION")
print("="*50)

# 1. Load and process 100 in-sample profiles
in_profiles = load_in_sample_profiles()
in_flex = compute_flexibility(in_profiles)

# 2. Solve for optimal bids using ALSO-X and CVaR
res_alsox = solve_alsox(in_flex, epsilon=0.10)
res_cvar = solve_cvar(in_flex, epsilon=0.10)

print(f"\n--- Task 2.1: Optimal Reserve Bids (P90) ---")
print(f"ALSO-X Bid: {res_alsox:.2f} kW")
print(f"CVaR Bid:   {res_cvar:.2f} kW")


# ── Task 2.2: Out-of-Sample Verification ────────────────────────────────
print(f"\n--- Task 2.2: Out-of-Sample Verification ---")

# 1. Load the 200 out-of-sample profiles
out_profiles = load_profiles("data/out_sample_profiles.csv")
out_flex = compute_flexibility(out_profiles)

# 2. Verify both bids and generate the comparison histogram
# Note: These functions should save the plot to 'results/task2_2_comparison.png'
sf_alsox = verify_p90_out_of_sample(res_alsox, out_profiles, "ALSO-X")
sf_cvar = verify_p90_out_of_sample(res_cvar, out_profiles, "CVaR")


# ── Task 2.3: Energinet Perspective (Sensitivity Analysis) ──────────────
print(f"\n--- Task 2.3: Reliability vs. Capacity Trade-off ---")

# 1. Define thresholds and run the sensitivity loop
# Note: This function generates the dual-axis plot in 'results/task2_3_tradeoff.png'
P_LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]
run_sensitivity_analysis(in_flex, out_flex, P_LEVELS)

print("\nStep 2 execution complete. All figures are located in the /results folder.")
