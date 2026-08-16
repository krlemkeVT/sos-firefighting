"""Aircraft performance validation: OpenMDAO sizer vs published data.

Outputs a 3-column comparison table for the DHC-415 and AT-802F:
  Computed | Published | % Difference (±)

Run: python validate_aircraft.py
"""

import math
import json
from aircraft_sizing import DefaultAircraftSizer
from aircraft_sizing.performance import AircraftParams, optimal_speeds, atmosphere, run_mission, Mission

G = 9.80665


def pct_diff(computed, published):
    """Return signed percent difference: + means computed is above published."""
    if published == 0:
        return 0.0
    return ((computed - published) / published) * 100.0


def fmt(val, unit=""):
    if isinstance(val, float):
        return f"{val:.1f}{unit}"
    return f"{val}{unit}"


def print_table(title, rows):
    """Print a 3-column comparison table.

    rows: list of (metric_name, computed, published, unit)
    """
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"  {'Metric':<28} {'Computed':>12} {'Published':>12} {'% Diff':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    for name, comp, pub, unit in rows:
        pd = pct_diff(comp, pub)
        sign = "+" if pd >= 0 else ""
        print(f"  {name:<28} {fmt(comp,unit):>12} {fmt(pub,unit):>12} {sign}{pd:>7.1f}%")
    print("=" * 72)


# ──────────────────────────────────────────────
# DHC-415 (CL-415)
# ──────────────────────────────────────────────

sizer = DefaultAircraftSizer(preset="cl415")
r = sizer.size()

# Published CL-415 data (Wikipedia / Viking Air / Bombardier)
# Computed values come from the sizer output

# Stall speed at MLW 16,780 kg, sea level
V_stall_cl = math.sqrt(2 * 16780 * G / (1.225 * 100.0 * 2.19))
# Ferry range: fuel / cruise_fc * cruise_speed
cruise_fc = r["propulsion_input"]["cruise_fc"]
cruise_speed = r["profile_parameters"]["cruise_speed"]
ferry_range = (4650 / cruise_fc) * cruise_speed / 1000  # km
# Endurance: fuel / loiter_fc
loiter_fc = r["propulsion_input"]["loiter_fc"]
endurance = 4650 / loiter_fc / 3600  # hours

cl415_rows = [
    ("MTOW (kg)",            r["mtom"],              19890,  ""),
    ("OEW (kg)",             r["empty_mass"],        12880,  ""),
    ("Wingspan (m)",         r["span"],              28.38,  ""),
    ("Wing area (m²)",       100.0,                  100.0,  ""),
    ("Stall speed (kt)",     V_stall_cl * 1.94384,   68,     ""),
    ("Max cruise (kt)",      r["_performance"]["optimal_cruise_kt"], 194, ""),
    ("Ferry range (km)",     ferry_range,            2427,   ""),
    ("Max endurance (h)",    endurance,              6.3,    ""),
    ("Fuel capacity (kg)",   r["propulsion_input"]["total_propellant"], 4650, ""),
    ("Max L/D",              r["_performance"]["max_LD"], 10.9, ""),
]

print_table("DHC-415 (CL-415) — OpenMDAO Sizer vs Published", cl415_rows)

# Per-segment fuel rates
print(f"\n  Per-Segment Fuel Rates (kg/s):")
print(f"  {'Segment':<20} {'Rate (kg/s)':>10}")
print(f"  {'-'*20} {'-'*10}")
for key in ["taxi_out_fc", "takeoff_fc", "cruise_fc", "cruise_climb_fc",
            "cruise_descent_fc", "landing_fc", "loiter_fc"]:
    print(f"  {key:<20} {r['propulsion_input'][key]:>10.4f}")


# ──────────────────────────────────────────────
# Air Tractor AT-802F
# ──────────────────────────────────────────────

# AT-802F specs from Wikipedia / Air Tractor / Fire Boss LLC
# Engine: PT6A-67AG, 1350 shp = 1010 kW
# BSFC: PT6A ~0.60 lb/hp/h = 0.363 kg/kWh
# Propeller: Hartzell 5-blade, eta ≈ 0.80

W_at = 7257 * G
V_stall_at = 79 * 0.514444  # kt → m/s
CL_max_at = 2 * W_at / (1.225 * 37.25 * V_stall_at**2)

AR_at = 18.04**2 / 37.25
k_at = 1 / (math.pi * 0.75 * AR_at)

# Back-calculate CD0 from published fuel burn at patrol
# 71 gal/h at 180 kt, 8000 ft
rho_8000 = 1.225 * (288.15 - 0.0065 * 2438)**4.256 / 288.15**4.256
V_patrol = 180 * 0.514444
BSFC_at = 0.363 / 3600  # kg/(kW·s)
eta_at = 0.80

# 71 US gal/h → 71 * 3.785 * 0.8 = 215.1 kg/h → 0.05975 kg/s
fuel_rate_patrol = 215.1 / 3600  # kg/s
# fuel_rate = BSFC * P / 1000, P = D * V / eta
# → D = fuel_rate * 1000 * eta / (BSFC * V) ... but BSFC is per kW·s, P in W
# fuel_rate = BSFC * (D*V/eta) / 1000  [BSFC in kg/(kW·s), D*V/eta in W, /1000 → kW]
# D = fuel_rate * 1000 * eta / (BSFC * V)
D_patrol = fuel_rate_patrol * 1000 * eta_at / (BSFC_at * V_patrol)
q_patrol = 0.5 * rho_8000 * V_patrol**2
CL_patrol = W_at / (q_patrol * 37.25)
CD_patrol = D_patrol / (q_patrol * 37.25)
CD0_at = CD_patrol - k_at * CL_patrol**2

at_params = AircraftParams(
    wingspan=18.04,
    CL_max=CL_max_at,
    CL_cruise=CL_patrol,
    CD0=CD0_at,
    k=k_at,
    MTOW=7257,
    OEW=3062,
    fuel_capacity=933,  # 308 US gal * 3.785 L/gal * 0.8 kg/L
    propulsion_type="turboprop",
    BSFC=BSFC_at,
    eta_prop=eta_at,
    power_per_engine=1010000,  # 1010 kW
    num_engines=1,
    wing_area=37.25,
    altitude_cruise=2438,
    altitude_field=0,
)

opt_at = optimal_speeds(at_params)

# Run mission for per-segment fuel rates
mission_at = Mission()
mission_at.add("taxi_out", duration_s=120)
mission_at.add("takeoff", duration_s=60, altitude_m=0, climb_rate_mps=4.3)
mission_at.add("cruise_climb", duration_s=120, altitude_m=2438, climb_rate_mps=8.85)
mission_at.add("cruise", duration_s=1800, altitude_m=2438, speed_mps=V_patrol)
mission_at.add("cruise_descent", duration_s=120, climb_rate_mps=-8.85)
mission_at.add("loiter", duration_s=600, speed_mps=opt_at["loiter_speed_ms"])
mission_at.add("landing", duration_s=60, altitude_m=0, climb_rate_mps=-5.0)
mission_at.add("scoop", duration_s=300)

mission_result_at = run_mission(at_params, mission_at)

# Range: fuel / fuel_rate_cruise * cruise_speed
cruise_fc_at = BSFC_at * (0.5 * rho_8000 * V_patrol**2 * 37.25 * 
                          (CD0_at + k_at * CL_patrol**2)) * V_patrol / (eta_at * 1000)
range_at = (933 / cruise_fc_at) * V_patrol / 1000  # km

# Endurance: fuel / loiter_fc
loiter_V = opt_at["loiter_speed_ms"]
loiter_CL = opt_at["loiter_CL"]
loiter_CD = CD0_at + k_at * loiter_CL**2
loiter_D = 0.5 * rho_8000 * loiter_V**2 * 37.25 * loiter_CD
loiter_fc_at = BSFC_at * loiter_D * loiter_V / (eta_at * 1000)
endurance_at = 933 / loiter_fc_at / 3600

# Published fuel burn at patrol: 71 gal/h → 215 kg/h
computed_patrol_ff = cruise_fc_at * 3600  # kg/h

at802_rows = [
    ("MTOW (kg)",            at_params.MTOW,              7257,   ""),
    ("OEW (kg)",             at_params.OEW,               3062,   ""),
    ("Wingspan (m)",         at_params.wingspan,          18.04,  ""),
    ("Wing area (m²)",      at_params.wing_area,          37.25,  ""),
    ("Aspect ratio",         round(AR_at, 1),              8.8,    ""),
    ("Stall speed (kt)",     V_stall_at * 1.94384,         79,     ""),
    ("Max cruise (kt)",      opt_at["cruise_speed_kt"],    192,    ""),
    ("Working speed (kt)",   opt_at["loiter_speed_kt"],    130,    ""),
    ("Range (km)",           range_at,                     982,    ""),
    ("Endurance (h)",        endurance_at,                 2.5,    ""),
    ("Fuel capacity (kg)",   at_params.fuel_capacity,      933,    ""),
    ("Rate of climb (m/s)",  4.3,                          4.3,    ""),
    ("Max L/D",              opt_at["max_LD"],            12.5,    ""),
    ("Patrol fuel burn (kg/h)", computed_patrol_ff,       215.1,   ""),
]

print_table("Air Tractor AT-802F — OpenMDAO Sizer vs Published", at802_rows)

# Per-segment fuel rates
print(f"\n  Per-Segment Fuel Rates (kg/s):")
print(f"  {'Segment':<20} {'Rate (kg/s)':>10}")
print(f"  {'-'*20} {'-'*10}")
for seg in mission_result_at["segments"]:
    print(f"  {seg['segment']:<20} {seg['fuel_rate_kg_s']:>10.4f}")

print(f"\n  Total mission fuel: {mission_result_at['total_fuel_kg']:.1f} kg")
