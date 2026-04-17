
import pandas as pd
import numpy as np

#import the functions from step1_init
from step1_scenario_generation import dataframe_building, generate_imbalance_scenarios, generate_balancing_prices

df_final = dataframe_building(
        price_file="data/energy_prices.csv",
        wind_file="data/wind_power_data.csv",
        price_date_col="MTU (CET/CEST)",
        wind_date_col="timestamp",
        price_col="Day-ahead Price (EUR/MWh)",
        wind_col="wind_power_mw"
    )


df_final = generate_imbalance_scenarios(df_final)
df_final = generate_balancing_prices(df_final)

print(df_final)