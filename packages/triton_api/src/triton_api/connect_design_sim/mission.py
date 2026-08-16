"""Firefighting mission profile builder.

Builds the mission segments that the aircraft performance model evaluates
to compute per-segment fuel burn rates for the wildfire simulation.
"""

from triton_api.connect_design_sim.setup import MissionSegment


def build_firefighting_mission(
    cruise_altitude: float = 3000.0,
    cruise_speed: float | None = None,
    loiter_speed: float | None = None,
) -> list[MissionSegment]:
    """Build a firefighting mission profile.

    The mission consists of:
    - taxi_out (120s)
    - takeoff (60s, climb to 457m)
    - cruise_climb (120s, climb to cruise altitude)
    - cruise (1800s at cruise altitude)
    - cruise_descent (120s)
    - loiter (600s at loiter speed)
    - landing (60s)
    - scoop (300s, water scooping)

    Returns a list of MissionSegment objects compatible with the sim.
    """
    return [
        MissionSegment(
            name="taxi_out",
            duration_s=120.0,
        ),
        MissionSegment(
            name="takeoff",
            duration_s=60.0,
            altitude_m=457.2,
            climb_rate_mps=3.78,
        ),
        MissionSegment(
            name="cruise_climb",
            duration_s=120.0,
            altitude_m=cruise_altitude,
            climb_rate_mps=8.85,
        ),
        MissionSegment(
            name="cruise",
            duration_s=1800.0,
            altitude_m=cruise_altitude,
            speed_mps=cruise_speed,
        ),
        MissionSegment(
            name="cruise_descent",
            duration_s=120.0,
            climb_rate_mps=-8.85,
        ),
        MissionSegment(
            name="loiter",
            duration_s=600.0,
            speed_mps=loiter_speed,
        ),
        MissionSegment(
            name="landing",
            duration_s=60.0,
            altitude_m=457.2,
            climb_rate_mps=-5.0,
        ),
        MissionSegment(
            name="scoop",
            duration_s=300.0,
        ),
    ]
