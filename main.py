"""Entry point: runs all Step 1 tasks (1.1, 1.2, 1.3, 1.4)."""

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
    run_cross_validation_vary_is,
    plot_cv_per_fold,
    plot_cv_avg_comparison,
    plot_vary_is_line,
    plot_vary_is_boxplot,
)

from step1_task_4 import (
    sweep_beta,
    assess_in_sample_sensitivity,
    plot_efficient_frontier,
    plot_offers_vs_beta,
    plot_profit_distributions,
    plot_sensitivity_frontiers,
)


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

vary_results = run_cross_validation_vary_is(scenarios)
plot_vary_is_line(vary_results)
plot_vary_is_boxplot(vary_results)

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