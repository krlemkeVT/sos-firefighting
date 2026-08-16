"""Design-to-simulation translator using OpenMDAO aircraft performance.

Replaces the OpenConcept-based translator with the aircraft_sizing package,
which uses OpenMDAO to compute fuel burn per mission segment and outputs
the SimulationInput format expected by the wildfire simulation.
"""

from aircraft_sizing import DefaultAircraftSizer
from triton_api.connect_design_sim.setup import DesignVariables, SimulationInput


def convert_openmdao_to_sim(
    design: DesignVariables | dict,
    preset: str = "cl415",
) -> dict:
    """Convert aircraft design variables to simulation input via OpenMDAO.

    Args:
        design: DesignVariables pydantic model or dict with wing_area_m2,
                aspect_ratio, payload_kg, cruise_speed_mps, fuel_mass_kg.
        preset: Aircraft preset to use ('cl415', 'dhc515', 'c172').

    Returns:
        Dict matching the wildfire sim aircraft JSON format.
    """
    sizer = DefaultAircraftSizer(preset=preset)

    if hasattr(design, "model_dump"):
        design_dict = design.model_dump()
    elif hasattr(design, "dict"):
        design_dict = design.dict()
    else:
        design_dict = dict(design)

    return sizer.size(design_dict)
