import numpy as np
import pandas as pd


# PARAMETERS


NUM_PROFILES = 300
MIN_LOAD = 220      # kW
MAX_LOAD = 600      # kW
MAX_DELTA = 35      # kW per minute
NUM_MINUTES = 60

IN_SAMPLE_SIZE = 100



# PROFILE GENERATION


def generate_profile():
    """
    Generate one valid load profile satisfying load and delta constraint
    """
    profile = np.zeros(NUM_MINUTES)

    # Initial value
    current = np.random.uniform(MIN_LOAD, MAX_LOAD)
    profile[0] = current

    for t in range(1, NUM_MINUTES):
        change = np.random.uniform(-MAX_DELTA, MAX_DELTA)
        new_value = current + change

        # Enforce bounds
        new_value = max(MIN_LOAD, min(MAX_LOAD, new_value))

        profile[t] = new_value
        current = new_value

    return profile


# GENERATE ALL SCENARIOS


def generate_all_profiles():
    profiles = np.array([generate_profile() for _ in range(NUM_PROFILES)])
    return profiles



# VALIDATION 


def validate_profiles(profiles):
    print("Min value:", profiles.min())
    print("Max value:", profiles.max())

    max_diff = np.max(np.abs(np.diff(profiles, axis=1)))
    print("Max step change:", max_diff)


# MAIN EXECUTION


if __name__ == "__main__":

    # Generate data
    profiles = generate_all_profiles()

    # Validate
    validate_profiles(profiles)

    # Shuffle before split
    np.random.shuffle(profiles)

    # Split
    in_sample = profiles[:IN_SAMPLE_SIZE]
    out_sample = profiles[IN_SAMPLE_SIZE:]

    print("In-sample shape:", in_sample.shape)
    print("Out-of-sample shape:", out_sample.shape)

    # Save
    pd.DataFrame(in_sample).to_csv("data/in_sample_profiles.csv", index=False)
    pd.DataFrame(out_sample).to_csv("data/out_sample_profiles.csv", index=False)

    print("Data saved successfully.")