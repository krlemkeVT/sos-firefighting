import pydantic
from typing import Dict, List

# These are in alignment with the setup of the repository
# Mission will need to be further developed for complexity
# Note any inputs that need to be added


class DesignVariables(pydantic.BaseModel):

    wing_area_m2: float
    aspect_ratio: float
    payload_kg: float
    cruise_speed_mps: float
    fuel_mass_kg: float



class MissionSegment(pydantic.BaseModel):

    name: str
    duration_s: float

    altitude_m: float | None = None
    speed_mps: float | None = None
    climb_rate_mps: float | None = None



class SimulationInput(pydantic.BaseModel):

    icon: str

    takeoff_landing_type: str

    autonomous: bool

    mtom: float
    empty_mass: float
    payload: float

    flow_rate: float
    can_scoop: bool
    scooping_distance: float

    span: float


    propulsion_input: Dict

    mission_profile: List[MissionSegment]