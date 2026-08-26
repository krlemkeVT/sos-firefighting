"""Tests for the spec-based aircraft model (SpecAircraftModel)."""

import pytest
from aircraft_sizing import SpecAircraftModel, PerformanceSpec


def test_cl415_spec_preset():
    """CL-415 spec preset should produce sim-compatible output with direct values."""
    model = SpecAircraftModel(preset="cl415")
    result = model.model()

    assert result["mtom"] == pytest.approx(19890, rel=0.01)
    assert result["empty_mass"] == pytest.approx(12880, rel=0.01)
    assert result["span"] == pytest.approx(28.38, rel=0.01)
    assert result["can_scoop"] is True
    assert result["propulsion_input"]["total_propellant"] > 0
    assert result["propulsion_input"]["cruise_fc"] > 0
    assert result["propulsion_input"]["takeoff_fc"] > 0
    assert result["profile_parameters"]["cruise_speed"] > 50
    # No _performance key — that's sizing-only
    assert "_performance" not in result


def test_at802f_spec_preset():
    """AT-802F spec preset should produce correct values."""
    model = SpecAircraftModel(preset="at802f")
    result = model.model()

    assert result["mtom"] == pytest.approx(7257, rel=0.01)
    assert result["can_scoop"] is False
    assert result["propulsion_input"]["total_propellant"] > 0


def test_spec_from_dataclass():
    """Should work with a PerformanceSpec dataclass directly."""
    spec = PerformanceSpec(
        mtom=10000, empty_mass=5000, payload=2000, span=15.0,
        total_propellant=1000, reserve_propellant=150,
        cruise_fc=0.1, takeoff_fc=0.2, loiter_fc=0.05,
        cruise_speed=70.0, loiter_speed=50.0,
    )
    model = SpecAircraftModel()
    result = model.model(spec)

    assert result["mtom"] == pytest.approx(10000, rel=0.01)
    assert result["empty_mass"] == pytest.approx(5000, rel=0.01)
    assert result["payload"] == pytest.approx(2000, rel=0.01)
    assert result["propulsion_input"]["cruise_fc"] == pytest.approx(0.1, abs=1e-6)
    assert result["propulsion_input"]["takeoff_fc"] == pytest.approx(0.2, abs=1e-6)
    assert result["profile_parameters"]["cruise_speed"] == 70.0


def test_spec_dict_override():
    """Dict overrides should merge with preset defaults."""
    model = SpecAircraftModel(preset="cl415")
    base = model.model()
    modified = model.model({"payload": 3000.0, "cruise_speed": 90.0})

    assert modified["payload"] == pytest.approx(3000, rel=0.01)
    assert modified["profile_parameters"]["cruise_speed"] == 90.0
    # Other values should remain from preset
    assert modified["mtom"] == base["mtom"]


def test_spec_to_json():
    """to_json should produce valid JSON string."""
    model = SpecAircraftModel(preset="cl415")
    import json
    j = model.to_json()
    parsed = json.loads(j)
    assert parsed["mtom"] == pytest.approx(19890, rel=0.01)
    assert "propulsion_input" in parsed
    assert "profile_parameters" in parsed


def test_spec_no_sizing_artifacts():
    """Spec model should not contain any sizing/optimization artifacts."""
    model = SpecAircraftModel(preset="cl415")
    result = model.model()

    # These keys exist in the sizer output but should NOT exist in spec output
    assert "_performance" not in result
    assert "optimal_cruise_ms" not in result
    assert "max_LD" not in result
