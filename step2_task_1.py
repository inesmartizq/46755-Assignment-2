"""
Step 2 – Task 2.1: In-sample Decision Making (FCR-D UP)
========================================================
Determines the optimal FCR-D UP reserve bid satisfying a P90 reliability
requirement using two approaches:
  - ALSO-X : sample-based MILP
  - CVaR   : linear programming approximation
"""

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

MIN_LOAD    = 220    # kW — minimum technical load
EPSILON     = 0.10   # violation budget (P90 requirement)
BIG_M       = 10000  # big-M constant for ALSO-X
NUM_MINUTES = 60     # minute-level resolution per profile


# ---------------------------------------------------------------------------
# Data loading and pre-processing
# ---------------------------------------------------------------------------

def load_in_sample_profiles():
    """
    Load the 100 in-sample load profiles from CSV.

    Returns
    -------
    np.array of shape (100, 60)
    """
    return pd.read_csv("data/in_sample_profiles.csv").values


def compute_flexibility(profiles):
    """
    Compute available FCR-D UP flexibility per minute per scenario.
    Reserve is provided by reducing consumption below current load.

    Parameters
    ----------
    profiles : np.array (W, M)  — load profiles [kW]

    Returns
    -------
    np.array (W, M)  — available flexibility [kW]
    """
    return profiles - MIN_LOAD


# ---------------------------------------------------------------------------
# Optimisation models
# ---------------------------------------------------------------------------

def solve_alsox(flexibility, epsilon=EPSILON):
    """
    ALSO-X sample-based MILP.
    Maximises the FCR-D UP reserve bid subject to a P90 reliability constraint.

    Parameters
    ----------
    flexibility : np.array (W, M)  — available flexibility per scenario/minute
    epsilon     : float            — allowed violation fraction (default 0.10)

    Returns
    -------
    float — optimal reserve bid [kW]
    """
    W, M = flexibility.shape

    m = gp.Model("alsox")
    m.Params.OutputFlag = 0

    R = m.addVar(lb=0, name="R")
    y = m.addVars(W, M, vtype=GRB.BINARY, name="y")   # 1 = violation

    m.setObjective(R, GRB.MAXIMIZE)

    for w in range(W):
        for t in range(M):
            m.addConstr(R <= flexibility[w, t] + BIG_M * y[w, t])

    m.addConstr(
        gp.quicksum(y[w, t] for w in range(W) for t in range(M))
        <= epsilon * W * M
    )

    m.optimize()
    return R.X


def solve_cvar(flexibility, epsilon=EPSILON):
    """
    CVaR-based LP approximation of the P90 reserve bid problem.

    Parameters
    ----------
    flexibility : np.array (W, M)
    epsilon     : float  — tail probability (default 0.10)

    Returns
    -------
    float — optimal reserve bid [kW]
    """
    W, M = flexibility.shape
    N = W * M

    m = gp.Model("cvar")
    m.Params.OutputFlag = 0

    R    = m.addVar(lb=0,              name="R")
    beta = m.addVar(lb=-GRB.INFINITY,  name="beta")
    z    = m.addVars(W, M, lb=0,       name="z")

    m.setObjective(R, GRB.MAXIMIZE)

    for w in range(W):
        for t in range(M):
            m.addConstr(z[w, t] >= R - flexibility[w, t] - beta)

    m.addConstr(
        beta + (1.0 / (epsilon * N)) *
        gp.quicksum(z[w, t] for w in range(W) for t in range(M))
        <= 0
    )

    m.optimize()
    return R.X