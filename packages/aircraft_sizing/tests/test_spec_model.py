"""Tests for the spec-based aircraft performance model (SpecAircraftModel).

Tests that ConfigSpec inputs → computed PerformanceSpec outputs, and that
the computed values are in reasonable ranges for known aircraft.

Running this file directly (python test_spec_model.py) prints full tables:
  - Input config specs
  - Computed performance specs
  - Computed vs published with % diff
"""

import json
import pytest
from aircraft_sizing import SpecAircraftModel, ConfigSpec, PerformanceSpec


# ──────────────────────────────────────────────
# Helpers for printing
# ──────────────────────────────────────────────

def _print_config(name, cfg):
    """Print the input ConfigSpec for a preset."""
    print(f"\n  Input Config Spec: {name}")
    print(f"  {'Parameter':<30} {'Value':>14}")
    print(f"  {'-'*30} {'-'*14}")
    for k, v in cfg.__dict__.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float):
            print(f"  {k:<30} {v:>14.4f}")
        else:
            print(f"  {k:<30} {str(v):>14}")
    print(f"  {'AR (derived)':<30} {cfg.AR:>14.4f}")
    print(f"  {'k (derived)':<30} {cfg.k:>14.6f}")


def _print_performance(perf):
    """Print the computed PerformanceSpec."""
    print(f"\n  Computed Performance Spec")
    print(f"  {'Metric':<30} {'Value':>14}")
    print(f"  {'-'*30} {'-'*14}")
    for k, v in perf.__dict__.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:>14.4f}")
        else:
            print(f"  {k:<30} {str(v):>14}")


def _print_comparison(result):
    """Print the comparison table: computed vs published."""
    print(f"\n  Computed vs Published ({result['preset']})")
    print(f"  {'Metric':<28} {'Computed':>12} {'Published':>12} {'% Diff':>10}   {'Source'}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*10}   {'-'*20}")
    for row in result["comparison"]:
        comp = f"{row['computed']:.4g}"
        pub = f"{row['published']:.4g}"
        if row["pct_diff"] is not None:
            pd = f"{row['pct_diff']:+.1f}%"
        else:
            pd = "N/A"
        src = row.get("source", "")
        print(f"  {row['metric']:<28} {comp:>12} {pub:>12} {pd:>10}   {src}")


# ──────────────────────────────────────────────
# Print-on-run helper
# ──────────────────────────────────────────────

def _run_and_print(preset):
    """Compute for a preset, print config + performance + comparison."""
    model = SpecAircraftModel(preset=preset)
    cfg = model._build_config(None)
    perf = model.compute()
    result = model.compare()

    print(f"\n{'='*80}")
    print(f"  {preset.upper()} — Config → Performance → Comparison")
    print(f"{'='*80}")
    _print_config(preset, cfg)
    _print_performance(perf)
    _print_comparison(result)
    print(f"{'='*80}")


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_cl415_compute():
    """CL-415 config should produce reasonable performance numbers."""
    model = SpecAircraftModel(preset="cl415")
    perf = model.compute()

    assert isinstance(perf, PerformanceSpec)
    assert perf.stall_speed_kt > 50
    assert perf.stall_speed_kt < 90
    assert perf.max_LD > 8
    assert perf.max_LD < 15
    assert perf.optimal_cruise_ms > 50
    assert perf.optimal_loiter_ms < perf.optimal_cruise_ms
    assert perf.cruise_fc > 0
    assert perf.loiter_fc > 0
    assert perf.takeoff_fc > 0
    assert perf.ferry_range_km > 100


def test_at802f_compute():
    """AT-802F config should produce reasonable performance numbers."""
    model = SpecAircraftModel(preset="at802f")
    perf = model.compute()

    assert perf.stall_speed_kt > 60
    assert perf.stall_speed_kt < 100
    assert perf.max_LD > 9
    assert perf.max_LD < 20
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

    assert modified.stall_speed_ms < base.stall_speed_ms


def test_compare():
    """compare() should return computed values and comparison rows."""
    model = SpecAircraftModel(preset="cl415")
    result = model.compare()

    assert result["preset"] == "cl415"
    assert "performance" in result
    assert "comparison" in result
    assert len(result["comparison"]) > 0

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

    assert perf.stall_speed_kt > 25
    assert perf.stall_speed_kt < 55
    assert perf.cruise_fc > 0
    assert perf.cruise_fc < 0.01


# ──────────────────────────────────────────────
# Print full report when run directly
# ──────────────────────────────────────────────

if __name__ == "__main__":
    for preset in ["cl415", "dhc515", "at802f", "c172"]:
        _run_and_print(preset)
