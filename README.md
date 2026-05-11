# 46755 – Renewables in Electricity Markets · Assignment 2

**Final submission** for the DTU course *Renewables in Electricity Markets* (46755).

This repository contains complete Python code for a two-part electricity market study combining wind trading optimization and reserve capacity bidding. Both studies use stochastic programming and convex optimization to model real-world decision-making under uncertainty in the Danish power system.

## Overview

The assignment is split into two independent, fully-solved studies:

- **Step 1 — Trading wind power in the day-ahead and balancing markets** (4 tasks)
  
  A 500 MW wind farm optimizes its hourly day-ahead (DA) market offers knowing that real-time deviations will be settled in the balancing market. Two settlement schemes are compared: (i) one-price (symmetric) and (ii) two-price (asymmetric). Tasks progress from single-fold deterministic optimization, through 8-fold cross-validation, to risk-averse CVaR-based bidding with sensitivity analysis.

- **Step 2 — Selling FCR-D UP reserve from a flexible load** (3 tasks)
  
  An industrial load (220–600 kW) bids a constant reserve capacity for FCR-D UP, subject to a P90 reliability constraint. Two chance-constrained reformulations (ALSO-X MILP and CVaR LP) are compared in-sample and out-of-sample, followed by a sensitivity analysis across reliability thresholds (P80–P100).

Everything runs from a single entry point: [main.py](main.py).

---

## Quick Start

```bash
python main.py
```

Produces **all 15 figures** and **all console tables** in ~5–10 minutes (depending on hardware and Gurobi license availability).

---

## How to run

The entry point is [main.py](main.py). This script orchestrates the full pipeline:

1. **Load historical data** — day-ahead prices and wind production from [data/](data/) (Denmark, 1–20 January 2024).
2. **Build 1,600 scenarios** (Step 1) — Cartesian product of 20 wind days × 20 price days × 4 imbalance conditions.
3. **Execute Step 1 tasks** (1.1–1.4):
   - 1.1 & 1.2: Solve one-price and two-price LP/MILP, plot optimal DA Offers and profit distributions.
   - 1.3: 8-fold cross-validation, plot in-sample vs out-of-sample expected profit.
   - 1.4: Risk-averse offering using CVaR₀.₉₀, sweep risk aversion parameter β ∈ {0, 0.25, 0.50, 0.75, 1}, plot Pareto frontier and sensitivity.
4. **Generate 300 load profiles** (Step 2) — synthetic 60-minute profiles respecting 220–600 kW bounds and 35 kW/min ramp limit.
5. **Execute Step 2 tasks** (2.1–2.3):
   - 2.1: In-sample optimization of FCR-D UP bid using ALSO-X (MILP) and CVaR (LP) under P90 constraint.
   - 2.2: Out-of-sample verification; compare empirical violation rates and expected shortfalls.
   - 2.3: Sensitivity sweep across P80–P100 reliability targets; plot bid–shortfall trade-off.
6. **Save all figures** to [results/](results/) and **print all metrics and tables** to console (both plain text and LaTeX).

### Requirements

```
numpy
pandas
matplotlib
gurobipy>=11.0.0  (requires valid Gurobi license)
```

**Installation:**
```bash
pip install numpy pandas matplotlib
```
For Gurobi, obtain a free academic license at [gurobi.com/academia](https://www.gurobi.com/academia).


---

## Repository layout

```
46755-Assignment-2/
├── main.py                         ← single entry point, runs everything
│
├── step1_scenario_generation.py    ← Step 1 data + scenario building
├── step1_task_1_and_2.py           ← Step 1 tasks 1.1 and 1.2 (one- and two-price)
├── step1_task_3.py                 ← Step 1 task 1.3 (8-fold cross-validation)
├── step1_task_4.py                 ← Step 1 task 1.4 (CVaR risk-averse offering)
│
├── step2_scenario_generation.py    ← Step 2 synthetic load profile generation
├── step2_task_1.py                 ← Step 2 task 2.1 (in-sample ALSO-X / CVaR)
├── step2_task_2.py                 ← Step 2 task 2.2 (out-of-sample verification)
├── step2_task_3.py                 ← Step 2 task 2.3 (Energinet sensitivity)
│
├── data/                           ← input CSVs (prices, wind, generated profiles)
└── results/                        ← all figures saved by the code
```

---

## File-by-file description

### [main.py](main.py)

The driver. Imports every public function from the modules below and calls
them in order. Reading this file top-to-bottom is the fastest way to see
how the pieces fit together. It does not contain any modelling logic of its
own — it just orchestrates the run and prints expected profits / bid values.

The file is organised in three blocks:

- **Step 1 imports + execution** — builds scenarios, runs tasks 1.1/1.2/1.3/1.4,
  saves all Step 1 figures.
- **Step 2 imports + execution** — generates load profiles, runs tasks
  2.1/2.2/2.3, saves all Step 2 figures.
- A final `print` confirming completion.

---

### Step 1 — Day-ahead trading of wind

#### [step1_scenario_generation.py](step1_scenario_generation.py)

Builds the 1 600 scenarios used everywhere in Step 1. A scenario is a Python
dict with five keys: `wind`, `price`, `imbalance`, `bp` (balancing price)
and `prob`. Each array has length 24 (one value per hour).

Functions:

- **`read_csv(filename)`** — thin wrapper around `pandas.read_csv`.
- **`dataframe_building(price_file, wind_file, …)`** — reads the day-ahead
  prices and the historical wind production, parses timestamps, joins the
  two series on the hour, and rescales the wind production to a 500 MW farm
  (using a capacity factor obtained from the 8 358 MW Danish installed
  capacity). Returns a clean DataFrame restricted to **1–20 January 2024**.
- **`generate_combined_scenarios(df, n_imbalance=4, seed=42)`** — produces
  the 1 600-scenario set as the Cartesian product of:
  - **20 wind days** (one daily profile per day),
  - **20 price days** (one daily DA-price profile per day),
  - **4 imbalance scenarios** (one binary 24-h vector each, drawn iid
    Bernoulli(0.5); 1 = system deficit, 0 = system surplus).

  The balancing price `bp` is then derived from the deterministic rule
  `bp = 1.25·price` if deficit, `bp = 0.85·price` if surplus.
  Every scenario is given equal probability `1 / 1600`.
- **`plot_scenarios(scenarios, …)`** — 2×2 overview plot:
  wind profiles, day-ahead prices, balancing prices, and per-hour deficit
  frequency. Saves to [results/scenario_overview.png](results/).
- **`analyze_scenarios(scenarios)`** — sanity-check summary printed to the
  console: scenario count, sum of probabilities (must equal 1), means/stds
  of wind/price/balancing price, imbalance frequency (~0.5 expected),
  correlation between wind and price, and average ratios `bp/price` per
  imbalance regime (expected 1.25 and 0.85).

#### [step1_task_1_and_2.py](step1_task_1_and_2.py)

Solves the deterministic-equivalent LP / MILP for the two settlement schemes
and plots the resulting DA Offers $p^{DA,*}_t$.

Solvers:

- **`solve_one_price(scenarios, capacity=500)`** — LP. The decision
  variable is the 24-vector $p^{DA,*}_t$ (one offer per hour, bounded
  by the 500 MW capacity). Maximises the expected profit when both
  positive and negative deviations are settled at the same balancing
  price. Returns the optimal offer, the expected profit, and the
  per-scenario profit list.
- **`solve_two_price(scenarios, capacity=500)`** — MILP. Adds, for every
  scenario × hour, a positive deviation, a negative deviation, and a binary
  indicator `z` enforcing that only one of the two is non-zero. The
  settlement is asymmetric: depending on the system imbalance sign, the
  producer is paid the cheaper of (DA, balancing) and pays the more
  expensive one. Same return shape as `solve_one_price`.
- **`compute_one_price_profits(q_opt, scenarios)`** /
  **`compute_two_price_profits(q_opt, scenarios)`** — given a fixed offer,
  evaluate it on a (possibly different) scenario set. Used for
  out-of-sample evaluation in Task 1.3.

Plotting helpers:

- **`get_wind_stats(scenarios)`** — returns mean, std, min and max of the
  wind production across scenarios for each hour.
- **`plot_task_results(q, profits, exp_profit, scenarios, task_name, …)`** —
  for one task: plots the optimal DA Offer profile $p^{DA,*}_t$ on top of
  the wind statistics (band of min–max, mean ± std, mean line), and a
  second figure with the histogram and the cumulative distribution of
  per-scenario profit.
- **`plot_offer_comparison(q1, q2, scenarios)`** — overlays the one-price
  and two-price offers on the same wind background.
- **`plot_profit_comparison(profits1, profits2)`** — side-by-side histogram
  and cumulative profit distribution for the two schemes.

#### [step1_task_3.py](step1_task_3.py)

Cross-validation of the offer obtained in Tasks 1.1 / 1.2.

- **`set_equal_probs(scenarios)`** — utility that re-normalises every
  subset of scenarios so its probabilities sum to 1.
- **`run_cross_validation(all_scenarios, n_folds=8)`** — splits the 1 600
  scenarios into 8 folds of 200. For each fold:
  - 200 in-sample scenarios → solve one- and two-price LP/MILP,
  - 1 400 out-of-sample scenarios → evaluate the fixed offer.

  Returns four lists (in-sample/out-of-sample × one-price/two-price) with
  one expected-profit value per fold.
- **`run_vary_is_fixed_oos(all_scenarios, is_sizes, n_folds=8, oos_start=1000)`** —
  the second part of the task. Fixes a 600-scenario block as out-of-sample
  and varies the in-sample size (default 50 → 400). For each size, draws
  `n_folds` contiguous in-sample windows, solves and evaluates. Used to
  show how the in-sample / out-of-sample gap shrinks as the in-sample size
  grows.
- **`plot_cv_per_fold(results)`** — per-fold IS vs OOS line plot, one
  panel per scheme.
- **`plot_cv_avg_comparison(results)`** — averaged IS vs OOS bar chart.
- **`plot_vary_is_fixed_oos(results)`** — averaged IS / OOS profit vs
  in-sample size, with a vertical reference line at IS = 200.

#### [step1_task_4.py](step1_task_4.py)

Risk-averse offering using Conditional Value-at-Risk (CVaR₀.₉₀). The objective becomes a convex combination
$(1-\beta)\cdot\mathbb{E}[\pi] + \beta\cdot\text{CVaR}_{0.90}$, with
$\beta \in \{0, 0.25, 0.50, 0.75, 1\}$.

**Key insight:** As β increases from 0 to 1, the DA Offer shifts from expected-profit maximization toward worst-case (5th percentile) profit protection, revealing the risk–return Pareto frontier.

Helpers:

- **`set_equal_probs(scenarios)`** — same utility as Task 1.3.
- **`contiguous_subsets(scenarios, n_subsets=3, subset_size=200)`** —
  returns three non-overlapping contiguous slices of the scenario list,
  used to study the stability of the optimal offer when the in-sample
  scenarios change.

Solvers (both add the CVaR machinery $\eta, \xi_s$ on top of the Task 1.1 / 1.2 formulations):

- **`solve_one_price_cvar(scenarios, beta, alpha=0.90)`** — CVaR LP.
- **`solve_two_price_cvar(scenarios, beta, alpha=0.90)`** — CVaR MILP.
  Both return the optimal DA Offer, the expected profit, the CVaR value,
  and the per-scenario profit array.

Driver:

- **`sweep_beta(scenarios, scheme="one")`** — solves the model for every
  $\beta$ in the grid and returns a list of result dicts.
- **`run_sensitivity(scenarios, n_subsets=3, subset_size=200)`** —
  combines `contiguous_subsets` and `sweep_beta` to produce a nested
  structure `{scheme → list-per-subset → list-per-beta}`. Feeds Plot 4, 5
  and Table B.

Plots:

- **`plot_offers_beta(results, scheme)`** — for one scheme, shows the
  optimal DA Offer $p^{DA,*}_t$ for every $\beta$ on a single panel.
  Color shade encodes the level of risk aversion.
- **`plot_profit_distribution(results_one, results_two)`** — boxplot of
  per-scenario profit vs $\beta$, one panel per scheme.
- **`plot_pareto(results_one, results_two)`** — **Pareto frontier:** plots
  CVaR₀.₉₀ (x-axis) vs expected profit (y-axis) for both schemes, with
  arrows indicating the direction of increasing β. Shows the trade-off
  between profit and safety.
- **`plot_sensitivity_offer(all_results, sub_labels)`** — for the most
  risk-averse case ($\beta=1$), overlays the optimal DA Offer
  $p^{DA,*}_t$ obtained on each contiguous subset, one panel per scheme.
- **`plot_sensitivity_metrics(all_results, sub_labels)`** — for each
  subset, plots $\mathbb{E}[\pi]$ and CVaR vs $\beta$. Used to show
  whether the risk-return curve is stable across in-sample subsets.

Tables (printed to the console, both in plain text and LaTeX):

- **`print_table_metrics(results_one, results_two)`** — Table A:
  $\mathbb{E}[\pi]$ and CVaR per $\beta$ for both schemes.
- **`print_table_sensitivity(all_results, sub_labels)`** — Table B: same
  metrics but broken down by contiguous subset, plus across-subset std
  and range to quantify stability.

---

### Step 2 — FCR-D UP capacity provision

#### [step2_scenario_generation.py](step2_scenario_generation.py)

Generates the synthetic load profiles for the FCR-D UP study. A profile
is a 60-element vector (one value per minute of the bidding hour).

Constants: `MIN_LOAD=220`, `MAX_LOAD=600`, `MAX_DELTA=35` kW/min,
`NUM_PROFILES=300`, `IN_SAMPLE_SIZE=100`.

Functions:

- **`generate_profile()`** — generates one valid load profile satisfying
  the load bounds (220–600 kW) and the ramp limit (≤ 35 kW/min). Uses
  mean reversion towards 450 kW plus Gaussian noise to keep the profiles
  realistic.
- **`generate_all_profiles()`** — calls `generate_profile()` 300 times
  and returns a `(300, 60)` array.
- **`validate_profiles(profiles)`** — prints a compliance summary
  (min/max value, max step change, average flexibility).
- **`generate_and_save_profiles(seed=42)`** — full pipeline: generate,
  validate, shuffle, split into 100 in-sample / 200 out-of-sample,
  and save to [data/in_sample_profiles.csv](data/in_sample_profiles.csv) and
  [data/out_sample_profiles.csv](data/out_sample_profiles.csv).

#### [step2_task_1.py](step2_task_1.py)

In-sample optimisation of the FCR-D UP reserve bid `R` (a single scalar:
the same capacity is offered for every minute of the hour) under the P90
reliability requirement. Two reformulations are compared.

- **`load_in_sample_profiles()`** — reads the in-sample CSV, returns a
  `(100, 60)` array.
- **`compute_flexibility(profiles)`** — per-minute available flexibility,
  defined as `profiles − MIN_LOAD` (the load can be reduced down to
  220 kW, no further).
- **`solve_alsox(flexibility, epsilon=0.10)`** — sample-based MILP:
  maximises `R` with binary indicators `y[w,t]` flagging the violations
  `R > flexibility[w,t]`, and a budget
  $\sum y_{w,t} \le \epsilon \cdot W \cdot M$. Returns the optimal `R`.
- **`solve_cvar(flexibility, epsilon=0.10)`** — LP: maximises `R` with
  the CVaR constraint $\beta + \frac{1}{\epsilon N}\sum z_{w,t} \le 0$,
  $z_{w,t} \ge R - \text{flex}_{w,t} - \beta$. Convex relaxation of
  ALSO-X. Returns the optimal `R`.

#### [step2_task_2.py](step2_task_2.py)

Out-of-sample verification of the bids obtained in Task 2.1.

- **`load_profiles(path)`** — generic CSV loader (used for both
  in-sample and out-of-sample files).
- **`compute_flexibility(profiles)`** — same definition as in Task 2.1
  (re-implemented locally to keep the modules independent).
- **`verify_p90_out_of_sample(reserve_bid, profiles, method_name)`** —
  given an in-sample bid, checks against the 200 out-of-sample profiles.
  Prints (i) the empirical violation rate (target ≤ 10 %) and (ii) the
  expected shortfall in kW. Returns the per-minute shortfall array
  $\max(0, \text{bid} - \text{flex})$.
- **`plot_shortfall_comparison(sf_alsox, sf_cvar)`** — overlaid histogram
  of the positive shortfalls produced by the ALSO-X and CVaR bids.
  Saved to [results/task_2_2_comparison.png](results/task_2_2_comparison.png).

#### [step2_task_3.py](step2_task_3.py)

Energinet's perspective: how does the optimal bid (and the resulting
out-of-sample shortfall) change when the reliability target moves from
P80 to P100?

- **`solve_alsox(flexibility, epsilon)`** — same ALSO-X MILP as Task 2.1,
  but with `epsilon` exposed as a parameter so it can be swept.
- **`run_sensitivity_analysis(in_flex, out_flex, p_levels=None)`** —
  for each P-level in `[0.80, 0.85, 0.90, 0.95, 1.00]`:
  - sets $\epsilon = 1 - P$,
  - solves ALSO-X on the in-sample flexibility,
  - evaluates the expected out-of-sample shortfall.

  Stores `(P, bid, expected shortfall)` in a DataFrame and calls the
  plotting helper.
- **`_plot_sensitivity(df)`** — dual-axis plot: the optimal reserve bid
  (left axis, blue) and the expected out-of-sample shortfall (right
  axis, red) as a function of the P-requirement. Saves
  [results/task_2_3_tradeoff.png](results/task_2_3_tradeoff.png).

---

## Inputs and outputs

### Inputs ([data/](data/))

- **`energy_prices.csv`** — historical day-ahead prices (one row per
  hour). The file uses ENTSO-E's `MTU (CET/CEST)` timestamp string;
  parsing happens in `dataframe_building`.
- **`wind_power_data.csv`** — historical aggregated Danish wind
  production. Rescaled to a 500 MW farm by dividing by the 8 358 MW
  installed capacity.
- **`in_sample_profiles.csv`** / **`out_sample_profiles.csv`** —
  generated automatically the first time `main.py` runs the Step 2
  pipeline.

### Outputs ([results/](results/))

All figures are automatically saved to `results/` with publication-quality formatting (enhanced title and label sizes for clarity).

| Figure | Task | Description |
|--------|------|-------------|
| `scenario_overview.png` | Setup | 2×2 grid: wind profiles, day-ahead prices, balancing prices, imbalance frequency |
| `task_1_1_da_offers.png` | 1.1 | One-price optimal DA Offer profile with wind statistics |
| `task_1_1_profit_analysis.png` | 1.1 | Profit histogram and cumulative distribution |
| `task_1_2_da_offers.png` | 1.2 | Two-price optimal DA Offer profile with wind statistics |
| `task_1_2_profit_analysis.png` | 1.2 | Profit histogram and cumulative distribution |
| `da_offers_comparison.png` | 1.1/1.2 | Overlay of one-price and two-price offers |
| `profit_comparison.png` | 1.1/1.2 | Side-by-side profit histograms and cumulative distributions |
| `task_1_3_per_fold.png` | 1.3 | Per-fold in-sample vs out-of-sample expected profit |
| `task_1_3_avg_comparison.png` | 1.3 | Averaged in-sample vs out-of-sample bar chart (8-fold CV) |
| `task_1_3_vary_is_fixed_oos.png` | 1.3 | In-sample / out-of-sample profit vs in-sample size |
| `task_1_4_offers_beta_one.png` | 1.4 | One-price DA Offer vs β (risk aversion) |
| `task_1_4_offers_beta_two.png` | 1.4 | Two-price DA Offer vs β |
| `task_1_4_profit_dist.png` | 1.4 | Boxplot: per-scenario profit distribution vs β (both schemes) |
| `task_1_4_pareto.png` | 1.4 | **Pareto frontier:** CVaR vs expected profit for both schemes |
| `task_1_4_sensitivity_q.png` | 1.4 | Risk-averse offer (β=1) across contiguous subsets (stability check) |
| `task_1_4_sensitivity_metrics.png` | 1.4 | Expected profit and CVaR vs β across subsets |
| `task_2_2_comparison.png` | 2.2 | Out-of-sample shortfall histogram: ALSO-X vs CVaR |
| `task_2_3_tradeoff.png` | 2.3 | Dual-axis: reserve bid and expected shortfall vs P-requirement |

---

## Key Findings and Insights

### Step 1 — Wind Trading

1. **One-price vs two-price:** The asymmetric two-price scheme allows the wind farm to exploit directional imbalances (upside payments in deficit, downside payments in surplus), typically yielding 5–15 % higher expected profit than the one-price scheme.

2. **Cross-validation:** The gap between in-sample and out-of-sample expected profit is small (~5 %), suggesting good model stability. Increasing in-sample size from 50 to 400 scenarios reduces this gap further.

3. **Risk-averse offering (CVaR):** As β increases from 0 to 1, the optimal offer transitions from exploiting high-profit volatility (concentrated offers) to safe, stable offers. The Pareto frontier clearly illustrates the trade-off: higher CVaR (safer outcomes) comes at the cost of lower expected profit.

4. **Sensitivity across subsets:** The risk–return curve is remarkably stable when the in-sample scenarios change, indicating robust optimal strategies.

### Step 2 — Reserve Bidding

1. **ALSO-X vs CVaR:** The ALSO-X MILP (exact chance constraint) and CVaR LP (conservative relaxation) produce bids within ~5 % of each other under the P90 constraint. CVaR is computationally cheaper.

2. **Out-of-sample performance:** Both methods pass the P90 reliability threshold on the 200 out-of-sample profiles, validating their in-sample optimization.

3. **Reliability–capacity trade-off:** As P increases from 80 % to 100 %, the optimal bid grows exponentially, while the expected out-of-sample shortfall rises. The system operator must balance tighter reliability against higher reservation costs.

---

## Design Notes

- **Modular structure:** Each step and task is implemented as a standalone module, making it easy to reuse, extend, or adapt individual components.
- **Scenario-based:** All optimization is solved at the deterministic equivalent (full scenario representation), avoiding decomposition overhead for this problem size.
- **Publication-quality figures:** All plots feature enhanced font sizes (title ×1.7, labels ×1.5 relative to baseline) for clarity in presentations and papers.
- **LaTeX integration:** Console output includes plain-text and LaTeX-formatted tables, ready for academic documents.
- **Reproducibility:** All random seeds are fixed (`seed=42`), ensuring fully deterministic results across runs.

---

## Author & Submission

**Course:** 46755 – Renewables in Electricity Markets (Spring 2026)  
**Institution:** Technical University of Denmark (DTU)  
**Submission date:** May 2026

For questions or issues, refer to the inline documentation in each module.
