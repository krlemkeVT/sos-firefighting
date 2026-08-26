"""Example: generate aircraft simulation input using spec-based performance.

This demonstrates the SpecAircraftModel — a direct pass-through model that
takes performance specs (fuel rates, speeds, weights) as inputs and outputs
the wildfire simulation's aircraft JSON format. No sizing, no mission analysis.

Branch: feature/performance-spec-model
"""

from aircraft_sizing import SpecAircraftModel, PerformanceSpec
import json


def main():
    # Use the CL-415 preset
    model = SpecAircraftModel(preset="cl415")
    result = model.model()
    print(json.dumps(result, indent=2))

    # Or provide your own spec directly
    custom = PerformanceSpec(
        mtom=15000, empty_mass=8000, payload=4000, span=20.0,
        total_propellant=2000, reserve_propellant=300,
        flow_rate=1.5, can_scoop=True, scooping_distance=300,
        cruise_fc=0.15, takeoff_fc=0.3, loiter_fc=0.08,
        cruise_speed=75.0, loiter_speed=55.0,
    )
    custom_result = SpecAircraftModel().model(custom)
    print("\n--- Custom Spec ---")
    print(json.dumps(custom_result, indent=2))


if __name__ == "__main__":
    main()
