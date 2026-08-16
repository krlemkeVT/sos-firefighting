# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Script used to run the Wildfire DoE.

Arguments can be passed through ``INPUT_CSV`` in order to replace
default values of the simulation.
"""

import argparse
from io import DEFAULT_BUFFER_SIZE
import pathlib

from examples.wildfire.paths import DATA_DIR
from examples.wildfire.simulation import WildfireParameters, WildfireSimulation
from sosid.doe.doe_runner import DesignOfExperiments
from sosid.output import OutputFormat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--doe-config", required=True)
    return parser.parse_args()

args = parse_args()
SCENARIO = args.scenario
DOE_CONFIG_PATH = args.doe_config


# Input/Output csv files
#INPUT_FILE = "ratio_doe_pyrenees.json"

INPUT_FILE = args.scenario

#DEFAULT_PARAMETERS_FILE = "baseline_pyrenees.json"

DEFAULT_PARAMETER_FILE = args.doe_config

input_file_path = DATA_DIR / "doe" / "gen_output" / INPUT_FILE
default_parameter_file_path = (
    DATA_DIR / "scenarios" / "inputs" / DEFAULT_PARAMETER_FILE
)
output_file_path = (
    input_file_path.parent.parent
    / "output"
    / (input_file_path.stem + "_out" + input_file_path.suffix)
)


# Defining special converter functions
def uniform_dist(values):
    """."""
    return lambda r: r.uniform(*values)


# Listing input parameters requiring special conversions
special_converters = {
    "response_time": uniform_dist,
    "wind_aspect": uniform_dist,
    "wind_speed": uniform_dist,
}


def doe_init():
    """."""
    # validate default parameters
    doe = DesignOfExperiments(
        simulation=WildfireSimulation,
        parameters=WildfireParameters,
        default_inputs=default_parameter_file_path,
        input_file=input_file_path,
        output_file=output_file_path,
        multiprocessing=True,
        n_repeats=15,
        output_sim_param=True,
        seed_start=0,
        output_format=OutputFormat.CSV,
        flatten=False,
    )

    doe.start()


if __name__ == "__main__":
    doe_init()
