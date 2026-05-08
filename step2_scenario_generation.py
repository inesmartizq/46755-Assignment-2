"""
Step 2 – Scenario Generation: Load Profile Generation
======================================================
Generates 300 synthetic load profiles satisfying operational constraints,
splits them into 100 in-sample and 200 out-of-sample, and saves to CSV.
"""

import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

NUM_PROFILES   = 300   # total profiles to generate
MIN_LOAD       = 220   # kW
MAX_LOAD       = 600   # kW
MAX_DELTA      = 35    # kW per minute
NUM_MINUTES    = 60    # single bidding hour
IN_SAMPLE_SIZE = 100   # profiles for optimisation


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def generate_profile():
    """
    Generate one valid load profile satisfying load and ramp constraints.
    Uses mean reversion to keep the profile in a realistic operating range.

    Returns
    -------
    np.array of shape (NUM_MINUTES,)
    """
    profile = np.zeros(NUM_MINUTES)
    current = np.random.uniform(400, 550)
    profile[0] = current
    target_mean = 450

    for t in range(1, NUM_MINUTES):
        reversion = 0.1 * (target_mean - current)
        noise     = np.random.normal(0, 12)
        change    = np.clip(reversion + noise, -MAX_DELTA, MAX_DELTA)
        current   = np.clip(current + change, MIN_LOAD, MAX_LOAD)
        profile[t] = current

    return profile


def generate_all_profiles():
    """
    Generate the full set of NUM_PROFILES load profiles.

    Returns
    -------
    np.array of shape (NUM_PROFILES, NUM_MINUTES)
    """
    return np.array([generate_profile() for _ in range(NUM_PROFILES)])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_profiles(profiles):
    """Print a constraint validation summary for the generated profiles."""
    print("\n--- Profile Validation ---")
    print(f"  Min value      : {profiles.min():.2f} kW  (constraint: >= {MIN_LOAD})")
    print(f"  Max value      : {profiles.max():.2f} kW  (constraint: <= {MAX_LOAD})")
    print(f"  Max step change: {np.max(np.abs(np.diff(profiles, axis=1))):.2f} kW  "
          f"(constraint: <= {MAX_DELTA})")
    print(f"  Avg flexibility: {np.mean(profiles - MIN_LOAD):.2f} kW")


# ---------------------------------------------------------------------------
# Generate, split, and save
# ---------------------------------------------------------------------------

def generate_and_save_profiles(seed=42):
    """
    Generate all profiles, validate, shuffle, split, and save to CSV.

    Saves
    -----
    data/in_sample_profiles.csv   (100 profiles)
    data/out_sample_profiles.csv  (200 profiles)
    """
    np.random.seed(seed)

    all_profiles = generate_all_profiles()
    validate_profiles(all_profiles)

    np.random.shuffle(all_profiles)

    in_sample  = all_profiles[:IN_SAMPLE_SIZE]
    out_sample = all_profiles[IN_SAMPLE_SIZE:]

    print(f"\n  In-sample shape :  {in_sample.shape}")
    print(f"  Out-of-sample shape: {out_sample.shape}")

    pd.DataFrame(in_sample).to_csv("data/in_sample_profiles.csv",  index=False)
    pd.DataFrame(out_sample).to_csv("data/out_sample_profiles.csv", index=False)

    print("  Profiles saved to data/")

    return in_sample, out_sample