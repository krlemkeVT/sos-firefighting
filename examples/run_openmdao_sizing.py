"""Example: generate aircraft simulation input using OpenMDAO performance.

This demonstrates the aircraft_sizing package producing output in the
wildfire simulation's aircraft JSON format.
"""

from aircraft_sizing import DefaultAircraftSizer
import json


def main():
    # Use the CL-415 preset
    sizer = DefaultAircraftSizer(preset="cl415")

    # Optionally override design variables
    design = {
        "wing_area_m2": 100.0,
        "payload_kg": 6200.0,
        "cruise_speed_mps": 95.0,
        "fuel_mass_kg": 2130.0,
    }

    result = sizer.size(design)

    print(json.dumps(result, indent=2))

    print("\n--- Performance Summary ---")
    perf = result["_performance"]
    print(f"Total fuel burn:      {perf['total_fuel_kg']} kg")
    print(f"Optimal cruise:       {perf['optimal_cruise_ms']} m/s "
          f"({perf['optimal_cruise_kt']:.0f} kt, M{perf['cruise_mach']})")
    print(f"Optimal loiter:       {perf['optimal_loiter_ms']} m/s "
          f"({perf['optimal_loiter_kt']:.0f} kt)")
    print(f"Max L/D:              {perf['max_LD']}")


if __name__ == "__main__":
    main()
