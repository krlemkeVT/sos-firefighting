import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "examples/wildfire/data/doe/output/updated_doe_palisades_out_simulation.csv"

# Weights
w1, w2, w3, w4, w5 = 0.2, 0.2, 0.2, 0.3, 0.1

# Pyrenees scenario maxima (* values)
BURNT_AREA_STAR_HA   = 9087        # ha
COST_AREA_STAR_MEUR  = 191106       # million €
EMISSION_AREA_STAR_T = 131224     # ton
FLEET_OPS_STAR_KEUR  = 250         # k€

# ---- FLEET COSTS PER DATA POINT (YOU MUST FILL THESE) ----
# Data points are 1..8 (each has 30 runs in the file, in order)
# Put your actual fleet acquisition cost in million € and ops cost in k€
fleet_acq_cost_MEUR_per_point = {
    1: 96,
    2: 96,
    3: 101.75,
    4: 96,
    5: 107.5,
    6: 84.5,
    7: 90.25,
    8: 96,
    
}



# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)

# Create data-point index: 30 runs per point, 8 points total
n_runs_per_point = 30
df["data_point"] = (df.index // n_runs_per_point) + 1

# -----------------------------
# UNIT CONVERSIONS
# -----------------------------
# burnt_area is in m^2 -> convert to ha
df["burnt_area_ha"] = df["burnt_area"] / 10_000.0

# fire cost is in € -> convert to M€
df["total_fire_cost_MEUR"] = df["total_fire_cost"] / 1_000_000.0

# emissions already in tons (assumed)
df["total_fire_emissions_ton"] = df["total_fire_emissions"]

# Map fleet costs per data point onto every run
df["fleet_acq_cost_MEUR"] = df["data_point"].map(fleet_acq_cost_MEUR_per_point)
df["fleet_ops_cost_KEUR"] = df["data_point"].map(fleet_ops_cost_KEUR_per_point)

# -----------------------------
# MOE PER RUN
# -----------------------------
term1 = w1 * (df["burnt_area_ha"]          / BURNT_AREA_STAR_HA)
term2 = w2 * (df["total_fire_cost_MEUR"]   / COST_AREA_STAR_MEUR)
term3 = w3 * (df["total_fire_emissions_ton"] / EMISSION_AREA_STAR_T)
term4 = w4 * (df["fleet_acq_cost_MEUR"]    / 100.0)                 # denominator is 100 M€
term5 = w5 * (df["fleet_ops_cost_KEUR"]    / FLEET_OPS_STAR_KEUR)

df["MoE"] = 1.0 - (term1 + term2 + term3 + term4 + term5)

# -----------------------------
# MOE STATS PER DATA POINT
# -----------------------------
moe_stats = df.groupby("data_point")["MoE"].agg(
    mean_MoE="mean",
    var_MoE="var"   # sample variance (ddof=1)
)

print("=== MoE statistics per data point (30 runs each) ===")
print(moe_stats)
print()

# -----------------------------
# CASUALTY & FLIGHT-TIME TRENDS
# -----------------------------
casualty_stats = df.groupby("data_point")["total_casualties"].agg(
    mean_casualties="mean",
    var_casualties="var"
)

# Using fleet_cumulative_flight_time as the "flight time" metric
flight_time_stats = df.groupby("data_point")["fleet_cumulative_flight_time"].agg(
    mean_flight_time="mean",
    var_flight_time="var"
)

print("=== Casualties per data point ===")
print(casualty_stats)
print()

print("=== Fleet cumulative flight time per data point ===")
print(flight_time_stats)
print()

# Optional: combine for easy inspection / plotting
trend_df = moe_stats.join(casualty_stats).join(flight_time_stats)
print("=== Combined summary (MoE, casualties, flight time) ===")
print(trend_df)

# -----------------------------
# SAVE RESULTS (optional)
# -----------------------------
df.to_csv("updated_doe_palisades_out_with_moe.csv", index=False)
trend_df.to_csv("palisades_datapoint_trends.csv")
