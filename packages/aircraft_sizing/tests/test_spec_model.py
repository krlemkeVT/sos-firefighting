"""Tests for the spec-based aircraft performance model (SpecAircraftModel).

Tests that ConfigSpec inputs → computed PerformanceSpec outputs, and that
the computed values are in reasonable ranges for known aircraft.
"""

import pytest
from aircraft_sizing import SpecAircraftModel, ConfigSpec, PerformanceSpec


def test_cl415_compute():
    """CL-415 config should produce reasonable performance numbers."""
    model = SpecAircraftModel(preset="cl415")
    perf = model.compute()

    assert isinstance(perf, PerformanceSpec)
    # Stall speed: CL-415 published ~68 kt
    assert perf.stall_speed_kt > 50
    assert perf.stall_speed_kt < 90
    # Max L/D: published ~10.9
    assert perf.max_LD > 8
    assert perf.max_LD < 15
    # Optimal cruise: should be > 50 m/s
    assert perf.optimal_cruise_ms > 50
    # Loiter should be slower than cruise (turboprop)
    assert perf.optimal_loiter_ms < perf.optimal_cruise_ms
    # Fuel rates should be positive
    assert perf.cruise_fc > 0
    assert perf.loiter_fc > 0
    assert perf.takeoff_fc > 0
    # Ferry range should be positive
    assert perf.ferry_range_km > 100


def test_at802f_compute():
    """AT-802F config should produce reasonable performance numbers."""
    model = SpecAircraftModel(preset="at802f")
    perf = model.compute()

    # Stall speed: AT-802 published ~79 kt
    assert perf.stall_speed_kt > 60
    assert perf.stall_speed_kt < 100
    # Max L/D: estimated ~12.5
    assert perf.max_LD > 9
    assert perf.max_LD < 20
    # Fuel rates positive
    assert perf.cruise_fc > 0
    assert perf.takeoff_fc > 0


def test_config_from_dataclass():
    """Should work with a ConfigSpec dataclass directly."""
    cfg = ConfigSpec(
        wingspan=20.0, wing_area=50.0, cl_max=2.0, cl_cruise=0.5,
        cd0=0.03, e=0.75, mtow=10000, oew=5000, fuel_capacity=2000,
        num_engines=1, power_per_engine=500000,
    )
    model = SpecAircraftModel()
    perf = model.compute(cfg)

    assert perf.stall_speed_ms > 0
    assert perf.max_LD > 0
    assert perf.cruise_fc > 0


def test_config_dict_override():
    """Dict overrides should merge with preset config."""
    model = SpecAircraftModel(preset="cl415")
    base = model.compute()
    modified = model.compute({"mtow": 15000.0})

    # Lighter aircraft should have lower stall speed
    assert modified.stall_speed_ms < base.stall_speed_ms


def test_compare():
    """compare() should return computed values and comparison rows."""
    model = SpecAircraftModel(preset="cl415")
    result = model.compare()

    assert result["preset"] == "cl415"
    assert "performance" in result
    assert "comparison" in result
    assert len(result["comparison"]) > 0

    # Each comparison row should have the right fields
    for row in result["comparison"]:
        assert "metric" in row
        assert "computed" in row
        assert "published" in row
        assert "pct_diff" in row


def test_to_json():
    """to_json should produce valid JSON with performance values."""
    import json
    model = SpecAircraftModel(preset="cl415")
    j = model.to_json()
    parsed = json.loads(j)
    assert "stall_speed_ms" in parsed
    assert "max_LD" in parsed
    assert "cruise_fc" in parsed


def test_c172_compute():
    """Cessna 172 config should produce reasonable numbers."""
    model = SpecAircraftModel(preset="c172")
    perf = model.compute()

    # C172 stall ~40 kt
    assert perf.stall_speed_kt > 25
    assert perf.stall_speed_kt < 55
    # Light aircraft — fuel rates should be small
    assert perf.cruise_fc > 0
    assert perf.cruise_fc < 0.01  # kg/s, very small for a 172
