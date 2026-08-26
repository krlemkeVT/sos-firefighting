"""Example: compute aircraft performance from config specs and compare to published.

Demonstrates the SpecAircraftModel — takes configuration/dimension specs as
inputs, computes performance via OpenConcept, and compares against published
reference data.

Branch: feature/performance-spec-model
"""

import json
from aircraft_sizing import SpecAircraftModel, ConfigSpec


def main():
    # ── CL-415: compute and compare ──
    model = SpecAircraftModel(preset="cl415")
    perf = model.compute()

    print("=" * 70)
    print("  CL-415 — Computed Performance from Config Specs")
    print("=" * 70)
    print(json.dumps(perf.__dict__, indent=2))

    print("\n" + "=" * 70)
    print("  CL-415 — Computed vs Published")
    print("=" * 70)
    print(f"  {'Metric':<28} {'Computed':>12} {'Published':>12} {'% Diff':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")

    result = model.compare()
    for row in result["comparison"]:
        comp = f"{row['computed']:.4g}"
        pub = f"{row['published']:.4g}"
        pd = f"{row['pct_diff']:+.1f}%" if row['pct_diff'] is not None else "N/A"
        print(f"  {row['metric']:<28} {comp:>12} {pub:>12} {pd:>10}")
    print("=" * 70)

    # ── AT-802F ──
    print("\n" + "=" * 70)
    print("  AT-802F — Computed vs Published")
    print("=" * 70)
    model_at = SpecAircraftModel(preset="at802f")
    result_at = model_at.compare()
    print(f"  {'Metric':<28} {'Computed':>12} {'Published':>12} {'% Diff':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    for row in result_at["comparison"]:
        comp = f"{row['computed']:.4g}"
        pub = f"{row['published']:.4g}"
        pd = f"{row['pct_diff']:+.1f}%" if row['pct_diff'] is not None else "N/A"
        print(f"  {row['metric']:<28} {comp:>12} {pub:>12} {pd:>10}")
    print("=" * 70)

    # ── Custom config ──
    print("\n--- Custom Config Spec ---")
    custom = ConfigSpec(
        wingspan=20.0, wing_area=50.0, cl_max=2.0, cl_cruise=0.5,
        cd0=0.03, e=0.75, mtow=10000, oew=5000, fuel_capacity=2000,
        num_engines=1, power_per_engine=500000,
    )
    custom_perf = SpecAircraftModel().compute(custom)
    print(f"  Stall speed:    {custom_perf.stall_speed_kt:.1f} kt")
    print(f"  Max L/D:        {custom_perf.max_LD:.1f}")
    print(f"  Cruise speed:   {custom_perf.optimal_cruise_kt:.0f} kt")
    print(f"  Loiter speed:   {custom_perf.optimal_loiter_kt:.0f} kt")
    print(f"  Ferry range:    {custom_perf.ferry_range_km:.0f} km")
    print(f"  Max endurance:  {custom_perf.max_endurance_h:.1f} h")


if __name__ == "__main__":
    main()
