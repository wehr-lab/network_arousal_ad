"""
This is a configuration file for network computation project.
It contains settings for the project, including paths to data directories, model configurations, and other parameters.
"""

import os
import sys
from pathlib import Path

## Project data directories
DATA_PATH = {}
## on hpc
# DATA_PATH["universal"] = "/".join(os.path.dirname(__file__).split("/")[:-2]) + "/data/"
# DATA_PATH["toneDecode"] = os.path.join(DATA_PATH["universal"], "tone_decode_data")

# on local machine
DATA_PATH["toneDecode"] = (
    "/Users/praveslamichhane/project_data/wehrlab/tone_decode_data"
)

DATA_PATH["2_photon"] = "/Users/praveslamichhane/project_data/wehrlab/2_photon_data/"

## Project code directories
CODE_PATH = {}

CODE_PATH["2-photon"] = "/".join(os.path.dirname(__file__).split("/")[:-1]) + "/src/"
# print(CODE_PATH)

CODE_PATH["EXT_LIB"] = "/gpfs/home/praves/wehrlab/ext_lib/Dynamic-Graph-Metrics"
