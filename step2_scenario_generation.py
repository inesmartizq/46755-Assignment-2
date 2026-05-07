import numpy as np
import pandas as pd
import os


# PARAMETERS

NUM_PROFILES = 300           # Total profiles to generate
MIN_LOAD = 220               # kW 
MAX_LOAD = 600               # kW 
MAX_DELTA = 35               # kW per minute  
NUM_MINUTES = 60             # Single bidding hour

IN_SAMPLE_SIZE = 100         # Profiles for training [cite: 78]


# PROFILE GENERATION

def generate_profile():
    """
    Generate one valid load profile satisfying load and delta constraints.
    Uses a normal distribution and mean reversion to keep the load 
    in a realistic operating range (above the 220kW floor).
    """
    profile = np.zeros(NUM_MINUTES)

    # 1. Start the load (e.g., between 400-550 kW)

    current = np.random.uniform(400, 550)
    profile[0] = current
    
    # Target mean to prevent the random walk to stay at 220kW 
    target_mean = 450 

    for t in range(1, NUM_MINUTES):
        #  Generate a change with a slight pull towards the target mean

        reversion = 0.1 * (target_mean - current)
        noise = np.random.normal(0, 12)
        change = reversion + noise

        # Enforce the MAX_DELTA constraint (35 kW/min) 
        change = np.clip(change, -MAX_DELTA, MAX_DELTA)
        
        new_value = current + change

        # Enforce the Load Bounds (220-600 kW) 
        new_value = np.clip(new_value, MIN_LOAD, MAX_LOAD)

        profile[t] = new_value
        current = new_value

    return profile

def generate_all_profiles():
    """
    Generates the full set of 300 profiles [cite: 74]
    """
    profiles = np.array([generate_profile() for _ in range(NUM_PROFILES)])
    return profiles


# VALIDATION 

def validate_profiles(profiles):
    print("--- Profile Validation ---")
    print(f"Absolute Min value: {profiles.min():.2f} kW (Constraint: >= 220)")
    print(f"Absolute Max value: {profiles.max():.2f} kW (Constraint: <= 600)")

    max_diff = np.max(np.abs(np.diff(profiles, axis=1)))
    print(f"Max step change:    {max_diff:.2f} kW (Constraint: <= 35)")
    
    avg_flex = np.mean(profiles - MIN_LOAD)
    print(f"Average available flexibility: {avg_flex:.2f} kW")


# MAIN EXECUTION

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Generate data
    all_profiles = generate_all_profiles()

    # Validate constraints
    validate_profiles(all_profiles)

    # Shuffle 
    np.random.shuffle(all_profiles)

    # Split into In-Sample (100) and Out-of-Sample (200)
    in_sample = all_profiles[:IN_SAMPLE_SIZE]
    out_sample = all_profiles[IN_SAMPLE_SIZE:]

    print(f"\nIn-sample shape:     {in_sample.shape}")
    print(f"Out-of-sample shape: {out_sample.shape}")

    # Save to CSV
    pd.DataFrame(in_sample).to_csv("data/in_sample_profiles.csv", index=False)
    pd.DataFrame(out_sample).to_csv("data/out_sample_profiles.csv", index=False)

    print("\nData saved successfully to 'data/' folder.")