"""
Script to process a session. It aligns pupil data, neural spiking data, tone stimuli, and GC network data temporally.
"""

##
import os
import sys
from pathlib import Path
import scipy.io as sio
import numpy as np
import glob
import argparse

## import custom modules
repoDir = Path(__file__).resolve().parent.parent.parent  ## dir of the repo root
print("repoDir:", repoDir)
configDir = repoDir / "config"
sys.path.append(str(configDir))

## define path variables
from settings import DATA_PATH, CODE_PATH  ## DATA_PATH might not be used here

sys.path.append(str(CODE_PATH["2-photon"]))

from utils.util_funcs import extract_value, strcmp, hdf5ToDict
from ephys.behavior import Behavior
from ephys.cell import Cell
from ephys.cell_ensemble import CellEnsemble
from ephys.events import Events
from ephys.events_process import EventsProcess
from ephys.session import Session

##
## define the data path for the session to analyze
parser = argparse.ArgumentParser(
    description="Process neural and pupil data for a given session."
)
parser.add_argument(
    "--session_path",
    type=str,
    required=True,
    help="Path to the session data directory.",
)
args = parser.parse_args()
DATA_PATH_AROUSAL = Path(args.session_path)  ## needs the absolute path to the data
# print("DATA_PATH:", DATA_PATH)

## make sure that data path exists
assert DATA_PATH_AROUSAL.exists(), f"Data path {DATA_PATH_AROUSAL} does not exist!"
# print(os.listdir(DATA_PATH_AROUSAL))

##
## load the .mat files
behaviorFilePath = glob.glob(str(Path(DATA_PATH_AROUSAL, "Beh*.mat")))[0]
behaviorFile = Behavior(behaviorFilePath)

headFile = behaviorFile.get_head_data()
reyeFile = behaviorFile.get_reye_data()

## define the ephys dir
sessionName = reyeFile["dirName"]
sessionName = extract_value(sessionName, "dirName")
ephysDirPath = os.path.join(DATA_PATH_AROUSAL, sessionName)

## load the sortedUnits file
sortedUnitsFilePath = glob.glob(os.path.join(ephysDirPath, "Sort*.mat"))[0]
sortedUnitsFile = sio.loadmat(sortedUnitsFilePath, struct_as_record=True)
sortedUnitsFile = sortedUnitsFile["SortedUnits"]
sortedUnitsFile = sortedUnitsFile.reshape(-1)

##
## create a list of Cell objects to pass to CellEnsemble class
cells = []
for cell in sortedUnitsFile:
    cells.append(Cell(cell))

## create a CellEnsemble object
cellEnsemble = CellEnsemble(cells)

## create the Events object and make a Session object
eventsFilePath = glob.glob(os.path.join(ephysDirPath, "Events*.mat"))[0]
eventsFile = sio.loadmat(eventsFilePath, struct_as_record=True)["Events"]
eventsFile = eventsFile.reshape(-1)

events = []
for event in eventsFile:
    events.append(Events(event))

session = Session(events)
sessionDF = session.toDataFrame()

##
pupilDiameter = behaviorFile.get_pupilDiameter_data()
cellEnsembleDF = cellEnsemble.toDataFrame()

##
## Extract stimulus onset and offset times
ms = 1e3  ## convert time in seconds to milliseconds

##
## define the frame numbers when the first and last stimuli were presented
sessionFrameNums = reyeFile["TTs"][0, 0].reshape(
    -1
)  ## frame numbers when the stimuli were presented
firstEventFrame = sessionFrameNums[
    0
]  ## first frame number when the first stimulus was presented
lastEventFrame = sessionFrameNums[
    -1
]  ## last frame number when the last stimulus was presented

##
## extract the time for OpenEphys for the first and last stimulus presentation
OEFirstEventTime = (
    session.get_event(0).soundcard_trigger_timestamp_sec
)  ## OE time in seconds when the first stimulus was presented
OELastEventTime = (
    session.get_event(-1).soundcard_trigger_timestamp_sec
)  ## OE time in seconds when the last stimulus was presented

##
## Calculate frames per second (frame rate)
framesPerSec = (lastEventFrame - firstEventFrame) / (OELastEventTime - OEFirstEventTime)
print(f"Calculated frame rate: {framesPerSec:.2f} frames/sec")
frameRate = extract_value(reyeFile["vid"][0][0]["framerate"])
print(f"Actual frame rate: {frameRate}")

##
pupil_frames = np.arange(
    len(pupilDiameter)
)  ## frame numbers for each pupil measurement

## Convert frame numbers to time (in sec) -> aligns frame numbers to the start of OEStart time
intercept = firstEventFrame - (framesPerSec * OEFirstEventTime)
timePupilEphys = (pupil_frames - intercept) / framesPerSec

## Convert to milliseconds if needed
timePupilEphys_ms = timePupilEphys * ms

##
## Create aligned pupil data structure
pupil_aligned = {
    "diameter": pupilDiameter,
    "frames": pupil_frames,
    "time_sec": timePupilEphys,
    "time_ms": timePupilEphys_ms,
    "frame_rate": framesPerSec,
}

# print(f"Pupil data aligned: {len(pupilDiameter)} samples")
# print(f"Time range: {timePupilEphys[0]:.3f}s to {timePupilEphys[-1]:.3f} s")
# print(f"Duration: {timePupilEphys[-1] - timePupilEphys[0]:.3f} s")

##
## Extract spike data aligned to pupil timing
# Get spike times for all cells
all_spike_times = []
cell_ids = []
spike_counts_binned = []

for i, cell in enumerate(cells):
    spike_times = cell.spiketimes
    all_spike_times.extend([(t, i) for t in spike_times])  # (time, cell_id) pairs
    cell_ids.append(i)
    # print(f"Cell {i}: {len(spike_times)} spikes")

##
# Sort spikes by time
all_spike_times.sort(key=lambda x: x[0])
spike_times_only = [st[0] for st in all_spike_times]
spike_cell_ids = [st[1] for st in all_spike_times]

# Create binned spike data to match pupil sampling rate
pupil_sampling_interval = np.median(
    np.diff(timePupilEphys)
)  # Median interval between pupil samples

##
# Bin spikes into pupil time bins
pupil_time_bins = timePupilEphys  ## binning the spikes in these bins
spike_rate_per_cell = np.zeros((len(cells), len(pupil_time_bins)))

for cell_idx in range(len(cells)):
    cell_spikes = [
        st[0] for st in all_spike_times if st[1] == cell_idx
    ]  ## incliding the spikes for the current cell[cell_idx], this is actually inefficient
    if len(cell_spikes) > 0:
        # Count spikes in each pupil time bin
        spike_counts, _ = np.histogram(
            cell_spikes,
            bins=np.concatenate(
                [pupil_time_bins, [pupil_time_bins[-1] + pupil_sampling_interval]]
            ),
        )  ## adding the last bin limit
        spike_rate_per_cell[cell_idx, :] = (
            spike_counts / pupil_sampling_interval
        )  # Convert to firing rate (Hz) so divide by time unit of sec

##
# Calculate population firing rate per time bin
population_firing_rate = np.mean(spike_rate_per_cell, axis=0)  ## per bin
total_spike_count = np.sum(spike_rate_per_cell, axis=0)  ## per bin

##
## Create comprehensive aligned data structure
neural_pupil_network_data = {
    "pupil": {
        "diameter": pupilDiameter,
        "time_sec": timePupilEphys,
        "time_ms": timePupilEphys_ms,
        "frames": pupil_frames,
        "frame_rate": framesPerSec,
    },
    "spikes": {
        "all_spike_times": spike_times_only,
        "all_spike_cells": spike_cell_ids,
        "spike_rate_per_cell": spike_rate_per_cell,
        "population_firing_rate": population_firing_rate,
        "total_spike_count": total_spike_count,
        "time_bins": pupil_time_bins,
    },
    "stimuli": {
        "event_times": [
            session.get_event(i).soundcard_trigger_timestamp_sec
            for i in range(len(sessionFrameNums))
        ],
        "event_frames": sessionFrameNums,
    },
}


# Adding GC network and aligning it temporally.
temp_network_path = glob.glob(str(Path(DATA_PATH_AROUSAL, "AGC_out*.mat")))[0]
assert (Path(temp_network_path)).exists()

##
AGC_data = hdf5ToDict(temp_network_path, include=["AGC"])

##
temporal_net = AGC_data["AGC"]

# The temporal_net should be (n_cells, n_cells, n_time_bins_network)
n_cells_network = temporal_net.shape[0]

# Check if network cells match spike data cells
if n_cells_network != len(cells):
    print(
        f"WARNING: Network has {n_cells_network} cells but spike data has {len(cells)} cells"
    )
else:
    print(f"Network and spike data have matching cell counts: {len(cells)} cells")

##
# Calculate network time bin duration
temporal_network_time_start = (
    0  # in seconds -> this is derived from known experimental design
)
temporal_network_time_end = 1800  # in seconds
total_network_duration = (
    temporal_network_time_end - temporal_network_time_start
)  # in seconds
n_time_bins_network = temporal_net.shape[2]  # number of time bins in the network data

# Each network time bin duration
network_bin_duration = total_network_duration / n_time_bins_network  # seconds per bin
network_bin_duration_ms = network_bin_duration * 1000  # milliseconds per bin

# Compare with pupil sampling
pupil_sampling_interval_ms = pupil_sampling_interval * 1000

##
## Create temporal alignment for network data
total_recording_duration = timePupilEphys[-1] - timePupilEphys[0]  # in seconds

network_time_array = np.linspace(
    temporal_network_time_start, temporal_network_time_end, n_time_bins_network
)  ## giving each network a time value for alignment

# Calculate various network measures for each time bin
network_measures = {}

## convert NaN to zeros
temporal_net = np.nan_to_num(temporal_net, nan=0.0)

total_connections_network = np.zeros(n_time_bins_network)
for t in range(n_time_bins_network):
    adj_matrix = temporal_net[:, :, t]
    total_connections_network[t] = np.nansum(adj_matrix)

## Interpolate network measures to match pupil sampling rate
from scipy import interpolate

# Interpolate network measures to pupil time points
total_connections_network_interp = interpolate.interp1d(
    network_time_array,
    total_connections_network,
    kind="linear",
    bounds_error=False,
    fill_value=0,
)(timePupilEphys)

## Update comprehensive aligned data structure with network data
neural_pupil_network_data["network"] = {
    # Raw network data
    "adjacency_matrix": temporal_net,  # (n_cells, n_cells, n_time_bins_network)
    "time_bins_raw": network_time_array,
    "bin_duration": network_bin_duration,
    # Network measures (interpolated to pupil sampling rate)
    "total_connections_per_bin": total_connections_network_interp,
}

## save the aligned data structure as a .pkl file
import pickle

output_path = Path(DATA_PATH_AROUSAL, "neural_pupil_network_aligned_temporally.pkl")
# Save the dictionary
with open(output_path, "wb") as file:
    pickle.dump(neural_pupil_network_data, file)
