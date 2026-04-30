import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product as cartesian_product

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


# def generate_imbalance_scenarios(df, p=0.5):
#     df["imbalance"] = np.random.binomial(1, p, size=len(df))
#     return df


# def generate_balancing_prices(df):
#     df["balancing_price EUR/MWh"] = np.where(df["imbalance"] == 1,
#                                               1.25 * df["price EUR/MWh"],
#                                               0.85 * df["price EUR/MWh"])
#     return df


def generate_combined_scenarios(df, n_imbalance=4, seed=42):
    np.random.seed(seed)
    days = df.index.normalize().unique()

    wind_scens  = [df[df.index.normalize() == d]['wind_power_mw'].values for d in days if len(df[df.index.normalize() == d]) == 24]
    price_scens = [df[df.index.normalize() == d]['price EUR/MWh'].values for d in days if len(df[df.index.normalize() == d]) == 24]
    imb_scens   = [np.random.binomial(1, 0.5, 24) for _ in range(n_imbalance)]

    prob = 1 / (len(wind_scens) * len(price_scens) * len(imb_scens))
    scenarios = []
    for w, p, i in cartesian_product(wind_scens, price_scens, imb_scens):
        bp = np.where(i == 1, 1.25 * p, 0.85 * p)
        scenarios.append({'wind': w, 'price': p, 'imbalance': i, 'bp': bp, 'prob': prob})

    print(f"Total scenarios: {len(scenarios)}")
    return scenarios


def plot_scenarios(scenarios, n_sample=1600, seed=0, save_path="results/scenario_overview.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scenarios), size=min(n_sample, len(scenarios)), replace=False)
    sample = [scenarios[i] for i in idx]

    wind  = np.array([s["wind"]      for s in scenarios])
    price = np.array([s["price"]     for s in scenarios])
    bp    = np.array([s["bp"]        for s in scenarios])
    imb   = np.array([s["imbalance"] for s in scenarios])
    hours = np.arange(24)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    for s in sample:
        ax.plot(hours, s["wind"], color="steelblue", alpha=0.15, lw=0.8)
    ax.plot(hours, wind.mean(axis=0), color="navy", lw=2.2, label="mean")
    ax.set_title(f"Wind power profiles (sample of {len(sample)} / {len(scenarios)})")
    ax.set_xlabel("Hour"); ax.set_ylabel("Wind power [MW]")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for s in sample:
        ax.plot(hours, s["price"], color="darkorange", alpha=0.15, lw=0.8)
    ax.plot(hours, price.mean(axis=0), color="saddlebrown", lw=2.2, label="mean")
    ax.set_title("Day-ahead price profiles")
    ax.set_xlabel("Hour"); ax.set_ylabel("Price [EUR/MWh]")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for s in sample:
        ax.plot(hours, s["bp"], color="seagreen", alpha=0.15, lw=0.8)
    ax.plot(hours, bp.mean(axis=0),    color="darkgreen",  lw=2.2, label="balancing price (mean)")
    ax.plot(hours, price.mean(axis=0), color="saddlebrown", lw=2.0, ls="--", label="day-ahead price (mean)")
    ax.set_title("Balancing price profiles")
    ax.set_xlabel("Hour"); ax.set_ylabel("Price [EUR/MWh]")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.bar(hours, imb.mean(axis=0), color="slategray", edgecolor="black")
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="expected (0.5)")
    ax.set_ylim(0, 1)
    ax.set_title("Deficit frequency per hour (imbalance = 1)")
    ax.set_xlabel("Hour"); ax.set_ylabel("Fraction of scenarios")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Scenario overview ({len(scenarios)} total scenarios)", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved scenario overview to {save_path}")


def analyze_scenarios(scenarios):
    wind  = np.array([s["wind"]      for s in scenarios])
    price = np.array([s["price"]     for s in scenarios])
    bp    = np.array([s["bp"]        for s in scenarios])
    imb   = np.array([s["imbalance"] for s in scenarios])
    probs = np.array([s["prob"]      for s in scenarios])

    deficit_mask = imb == 1
    surplus_mask = imb == 0
    ratio_deficit = (bp[deficit_mask] / price[deficit_mask]).mean() if deficit_mask.any() else np.nan
    ratio_surplus = (bp[surplus_mask] / price[surplus_mask]).mean() if surplus_mask.any() else np.nan

    flat_wind  = wind.flatten()
    flat_price = price.flatten()
    corr_wind_price = np.corrcoef(flat_wind, flat_price)[0, 1]

    summary = {
        "n_scenarios":          len(scenarios),
        "sum_probabilities":    probs.sum(),
        "wind_mean":            wind.mean(),
        "wind_std":             wind.std(),
        "wind_min":             wind.min(),
        "wind_max":             wind.max(),
        "price_mean":           price.mean(),
        "price_std":            price.std(),
        "price_min":            price.min(),
        "price_max":            price.max(),
        "bp_mean":              bp.mean(),
        "bp_std":               bp.std(),
        "bp_min":               bp.min(),
        "bp_max":               bp.max(),
        "imbalance_frequency":  imb.mean(),
        "corr_wind_price":      corr_wind_price,
        "bp_over_price_deficit": ratio_deficit,
        "bp_over_price_surplus": ratio_surplus,
    }

    print("\n── Scenario analysis ───────────────────────────────────────────")
    print(f"  Total scenarios       : {summary['n_scenarios']}")
    print(f"  Sum of probabilities  : {summary['sum_probabilities']:.6f}  (expect 1.0)")
    print(f"  Wind  [MW]            : mean={summary['wind_mean']:.2f}  std={summary['wind_std']:.2f}  "
          f"min={summary['wind_min']:.2f}  max={summary['wind_max']:.2f}")
    print(f"  Price [EUR/MWh]       : mean={summary['price_mean']:.2f}  std={summary['price_std']:.2f}  "
          f"min={summary['price_min']:.2f}  max={summary['price_max']:.2f}")
    print(f"  BP    [EUR/MWh]       : mean={summary['bp_mean']:.2f}  std={summary['bp_std']:.2f}  "
          f"min={summary['bp_min']:.2f}  max={summary['bp_max']:.2f}")
    print(f"  Imbalance frequency   : {summary['imbalance_frequency']:.4f}  (expect ≈ 0.5)")
    print(f"  Corr(wind, price)     : {summary['corr_wind_price']:+.4f}")
    print(f"  bp/price | deficit    : {summary['bp_over_price_deficit']:.4f}  (expect 1.25)")
    print(f"  bp/price | surplus    : {summary['bp_over_price_surplus']:.4f}  (expect 0.85)")
    print("────────────────────────────────────────────────────────────────\n")

    return summary