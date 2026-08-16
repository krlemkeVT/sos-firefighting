"""Tests for the aircraft_sizing package."""

import pytest
from aircraft_sizing import DefaultAircraftSizer, AircraftParams, optimal_speeds
from aircraft_sizing.performance import run_mission, atmosphere


def test_cl415_preset():
    """CL-415 preset should produce reasonable numbers."""
    sizer = DefaultAircraftSizer(preset="cl415")
    result = sizer.size()

    assert result["mtom"] > 15000
    assert result["mtom"] < 25000
    assert result["span"] == pytest.approx(28.38, rel=0.01)
    assert result["propulsion_input"]["total_propellant"] > 0
    assert result["propulsion_input"]["cruise_fc"] > 0
    assert result["propulsion_input"]["takeoff_fc"] > 0
    assert result["profile_parameters"]["cruise_speed"] > 50
    assert "_performance" in result


def test_optimal_speeds_cl415():
    """CL-415 optimal speeds should match published data."""
    params = AircraftParams(
        wingspan=28.38, wing_area=100.0, CL_max=2.19, CL_cruise=0.43,
        CD0=0.0414, k=0.0507, MTOW=19890, OEW=12880, fuel_capacity=4650,
        propulsion_type="turboprop", BSFC=0.286/3600,
        eta_prop=0.82, power_per_engine=1775000, num_engines=2,
        altitude_cruise=1500,
    )
    opt = optimal_speeds(params)
    # Max L/D should be ~10.9 for the CL-415
    assert opt["max_LD"] == pytest.approx(10.9, rel=0.1)
    # Cruise speed (turboprop = max L/D speed) should be reasonable
    assert opt["cruise_speed_ms"] > 50
    assert opt["cruise_speed_ms"] < 100
    # Loiter (min power) should be slower than cruise
    assert opt["loiter_speed_ms"] < opt["cruise_speed_ms"]


def test_design_override():
    """Design variable overrides should change the output."""
    sizer = DefaultAircraftSizer(preset="cl415")
    base = sizer.size()
    modified = sizer.size({"payload_kg": 3000.0})
    assert modified["payload"] != base["payload"]
    assert modified["payload"] == 3000.0


def test_atmosphere():
    """ISA atmosphere should give reasonable values."""
    sl = atmosphere(0)
    assert sl["rho"] == pytest.approx(1.225, rel=0.01)
    assert sl["T"] == pytest.approx(288.15, rel=0.01)

    alt = atmosphere(5000)
    assert alt["rho"] < sl["rho"]
    assert alt["T"] < sl["T"]


def test_mission_fuel_burn():
    """Mission fuel burn should be positive and less than fuel capacity."""
    params = AircraftParams(
        wingspan=28.38, wing_area=100.0, CL_max=2.19, CL_cruise=0.43,
        CD0=0.0414, k=0.0507, MTOW=19890, OEW=12880, fuel_capacity=4650,
        propulsion_type="turboprop", BSFC=0.286/3600,
        eta_prop=0.82, power_per_engine=1775000, num_engines=2,
        altitude_cruise=3000,
    )
    from aircraft_sizing.performance import Mission
    mission = Mission()
    mission.add("taxi_out", duration_s=120)
    mission.add("takeoff", duration_s=60)
    mission.add("cruise", duration_s=1800, altitude_m=3000, speed_mps=95)
    mission.add("landing", duration_s=60)

    results = run_mission(params, mission)
    assert results["total_fuel_kg"] > 0
    assert results["total_fuel_kg"] < params.fuel_capacity
