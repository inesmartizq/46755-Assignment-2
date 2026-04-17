import os
import pandas as pd
import numpy as np

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def dataframe_building(price_file, wind_file, price_date_col, wind_date_col, price_col, wind_col):
    df_price = read_csv(price_file)
    df_wind = read_csv(wind_file)

    df_price["datetime"] = pd.to_datetime(
        df_price[price_date_col].str.split(" - ").str[0].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip(),
        dayfirst=True
    )
    df_wind["datetime"] = (
        pd.to_datetime(df_wind[wind_date_col], utc=True)
        .dt.tz_convert("Europe/Copenhagen")
        .dt.tz_localize(None)
    )

    df = pd.merge(df_price[["datetime", price_col]],
                  df_wind[["datetime", wind_col]],
                  on="datetime", how="inner")

    df = df.rename(columns={price_col: "price EUR/MWh", wind_col: "wind_mw"})

    installed_capacity = 8358  # MW

    df["wind_mw"] = df["wind_mw"] / installed_capacity
    df = df.rename(columns={"wind_mw": "cf"})

    case_study_capacity = 500  # MW

    df["cf"] = df["cf"] * case_study_capacity
    df = df.rename(columns={"cf": "wind_power_mw"})

    df = df.set_index("datetime")
    df_clipped = df["2024-01-01":"2024-01-20"]

    return df_clipped


def generate_imbalance_scenarios(df, p=0.5):
    df["imbalance"] = np.random.binomial(1, p, size=len(df))
    return df


def generate_balancing_prices(df):
    df["balancing_price EUR/MWh"] = np.where(df["imbalance"] == 1,
                                              1.25 * df["price EUR/MWh"],
                                              0.85 * df["price EUR/MWh"])
    return df