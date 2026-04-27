from step1_scenario_generation import dataframe_building, generate_combined_scenarios
from step1_task_1_and_2 import (
    solve_one_price,
    solve_two_price,
    plot_task_results,
    plot_offer_comparison,
    plot_profit_comparison,
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