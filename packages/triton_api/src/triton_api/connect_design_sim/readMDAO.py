from packages.triton_api.src.triton_api.connect_design_sim.setup import SimulationInput
from packages.triton_api.src.triton_api.connect_design_sim.mission import build_firefighting_mission

# Following structure from OpenConcept to get the various variables. 
# Assuming reserve propellant is 15% of original (this differs from the Sprnig 2026 version with a reserve mission)
# Note various firefighting specific variables constant (flow rate, scoop/scooping distance, etc.)
# Aircraft TO/Land type is defined and not variable. Need to change from vertipad to proper type in alignment with the OpenConcept vehicles

def convert_openconcept_to_sim(prob):


    mtom = prob.get_val(
        "ac|weights|MTOW",
        units="kg"
    ).item()


    empty_mass = prob.get_val(
        "OEW",
        units="kg"
    ).item()


    span = prob.get_val(
        "ac|geom|wing|span",
        units="m"
    ).item()



    propulsion = {

        "architecture":
            "conventional",


        "total_propellant":
            prob.get_val(
                "fuel_mass",
                units="kg"
            ).item(),


        "reserve_propellant":
            0.15 *
            prob.get_val(
                "fuel_mass",
                units="kg"
            ).item(),


        "propellant_unit":
            "kg",


        "takeoff_fc":
            prob.get_val(
                "takeoff.fuel_flow"
            ).mean(),


        "cruise_fc":
            prob.get_val(
                "cruise.fuel_flow"
            ).mean(),


        "hover_fc":
            prob.get_val(
                "hover.fuel_flow"
            ).mean()

    }



    return SimulationInput(

        icon="evtol.svg",

        takeoff_landing_type="vertipad",

        autonomous=False,


        mtom=mtom,

        empty_mass=empty_mass,

        payload=prob.get_val(
            "payload",
            units="kg"
        ).item(),


        flow_rate=720,

        can_scoop=True,

        scooping_distance=8,


        span=span,


        propulsion_input=propulsion,


        mission_profile=
            build_firefighting_mission()

    )