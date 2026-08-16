import fastapi

from packages.triton_api.src.triton_api.connect_design_sim.setup import DesignVariables

# This needs to be updated after the OpenConcept is imported as a package. 
# This calls the "prob" variable and other functions from the OpenConcept model in effort to not do work that is already there
from openconcept_model import create_aircraft_problem

from packages.triton_api.src.triton_api.connect_design_sim.readMDAO import convert_openconcept_to_sim


app = fastapi.FastAPI(
    title="Aircraft Design Translator"
)



@app.post("/design-to-simulation")
def design_to_simulation(
        design: DesignVariables
):

    # build OpenConcept model
    prob = create_aircraft_problem(
        design
    )
    # solve sizing
    prob.run_model()

    # convert only
    simulation_input = (
        convert_openconcept_to_sim(prob)
    )

    return simulation_input