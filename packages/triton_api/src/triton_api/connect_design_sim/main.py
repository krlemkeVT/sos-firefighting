"""FastAPI endpoint for aircraft design-to-simulation translation.

Uses the aircraft_sizing package (OpenMDAO-based) to compute aircraft
performance numbers and convert them to the wildfire simulation format.
"""

import fastapi

from triton_api.connect_design_sim.setup import DesignVariables
from triton_api.connect_design_sim.readMDAO import convert_openmdao_to_sim


app = fastapi.FastAPI(
    title="Aircraft Design Translator (OpenMDAO)"
)


@app.post("/design-to-simulation")
def design_to_simulation(
    design: DesignVariables,
    preset: str = "cl415",
):
    """Convert aircraft design variables to simulation input.

    Uses OpenMDAO to compute:
    - Fuel burn per second for each mission segment
    - Optimal cruise and loiter speeds
    - Full mission profile with per-segment fuel rates

    Returns a dict matching the wildfire sim aircraft JSON format.
    """
    simulation_input = convert_openmdao_to_sim(design, preset=preset)
    return simulation_input


@app.get("/presets")
def list_presets():
    """List available aircraft presets."""
    from aircraft_sizing.sizer import PRESETS
    return {
        name: {
            "wingspan": p.get("wingspan"),
            "mtow": p.get("mtow"),
            "propulsion": p.get("propulsion"),
        }
        for name, p in PRESETS.items()
    }
