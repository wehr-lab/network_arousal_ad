## import standard libraries
import argparse
import os
import sys
from pathlib import Path
import glob
import numpy as np
import scipy.io as sio
import pandas as pd

# Current script directory
repoDir = Path(__file__).resolve().parent.parent.parent
print(f"Repo dir: {repoDir}")
configDir = repoDir / "config"
sys.path.append(str(configDir))
print(f"Sys path: {sys.path}")
from settings import DATA_PATH, CODE_PATH

sys.path.append(str(CODE_PATH["2-photon"]))

from utils.util_funcs import extract_value, strcmp
from ephys.behavior import Behavior
from ephys.cell import Cell
from ephys.cell_ensemble import CellEnsemble
from ephys.events import Events
from ephys.events_process import EventsProcess
from ephys.session import Session
from ephys.session_process import SessionProcess

## define variables
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

## load the notebook file
notebookFilePath = glob.glob(os.path.join(ephysDirPath, "note*.mat"))[0]
notebookFile = sio.loadmat(notebookFilePath, struct_as_record=True)
stimLog = notebookFile["stimlog"].reshape(-1)

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

pupilDiameter = behaviorFile.get_pupilDiameter_data()
cellEnsembleDF = cellEnsemble.toDataFrame()

## Extract stimulus onset and offset times
timeWindow = [
    -0.2,
    0.5,
]  ## time window in seconds to extract spikes around the stimulus onset

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

## extract the time for OpenEphys for the first and last stimulus presentation
OEFirstEventTime = (
    session.get_event(0).soundcard_trigger_timestamp_sec
)  ## OE time in seconds when the first stimulus was presented
OELastEventTime = session.get_event(
    -1
).soundcard_trigger_timestamp_sec  ## OE time in seconds when the last stimulus was presented -> TODO! this comes to 1919.9 sec which is not the expected time of 1800 sec

## to track the frame numbers with OE time, find the slope and intercept of FrameNumber vs OE time
framesPerSec = (lastEventFrame - firstEventFrame) / (OELastEventTime - OEFirstEventTime)
intercept = firstEventFrame - (
    framesPerSec * OEFirstEventTime
)  ## frame number when OE time is 0

## processing each event now
eventsProcess = []
for trial in range(len(session)):
    if strcmp(session.get_event(trial).type, ("tone", "whitenoise", "silentsound")):
        if hasattr(session.get_event(trial), "soundcard_trigger_timestamp_sec"):
            eventStartTime = session.get_event(trial).soundcard_trigger_timestamp_sec
        else:
            raise ValueError(
                "Event does not have soundcard_trigger_timestamp_sec attribute."
            )

        ## now store all the event specific information in the EventsProcess object
        currEvent = EventsProcess(
            session.get_event(trial)
        )  ## initialize with the Events object

        ## event (trial) specific start and stop times
        currEvent.OEStartTime = np.float32(
            eventStartTime + timeWindow[0]
        )  ## start time in milliseconds
        currEvent.OEEndTime = np.float32(eventStartTime + timeWindow[1])
        currEvent.freqStart = np.float32(eventStartTime)
        currEvent.universalStartTime = np.float32(
            session.get_event(0).soundcard_trigger_timestamp_sec + timeWindow[0]
        )  ## start time in milliseconds
        currEvent.universalEndTime = np.float32(
            session.get_event(-1).soundcard_trigger_timestamp_sec + timeWindow[1]
        )  ## end time in milliseconds

        currEvent.freqEnd = np.float32(
            currEvent.freqStart + currEvent.event.duration / 1000
        )  ## end time of stimulus in seconds; note that the duration is in milliseconds

        ## find the frame number for the start and end time of the stimulus
        currEvent.frameRate = extract_value(
            reyeFile["vid"][0][0]["framerate"]
        )  ## lots of nesting, but checked and its correct
        currEvent.firstFrame = np.int32(
            np.ceil(sessionFrameNums[trial] + timeWindow[0] * currEvent.frameRate)
        )  ## frame number for the start time of the stimulus
        currEvent.lastFrame = np.int32(
            np.ceil(sessionFrameNums[trial] + timeWindow[1] * currEvent.frameRate)
        )

        ## extracting the spiketimes now
        for cellNum in range(len(cellEnsemble)):
            currCell = cellEnsemble.get_cell(cellNum)
            currEvent.spikeTimes.append(
                currCell.spiketimes[
                    (currCell.spiketimes >= currEvent.OEStartTime)
                    & (currCell.spiketimes <= currEvent.OEEndTime)
                ]
            )  ## extract the spike times within the time window for each cell and append

        ## extract the pupil diameter for the event
        currEvent.pupilDiameter = pupilDiameter[
            currEvent.firstFrame : currEvent.lastFrame
        ]  ## extract the pupil diameter for the event -> note that this returns an array itself

        ## time of frames
        currEvent.frameNums = np.arange(
            currEvent.firstFrame, currEvent.lastFrame + 1
        )  ## frame numbers for the event
        currEvent.frameTimes = np.float32(
            ((currEvent.frameNums - intercept) / framesPerSec)
        )

        ## store timeWindow and frameWindow
        currEvent.timeWindow = timeWindow
        currEvent.frameWindow = [currEvent.firstFrame, currEvent.lastFrame]

        ## append the currEvent to the eventsProcess list
        eventsProcess.append(currEvent)

currSessionProcess = SessionProcess(eventsProcess)
sessionProcessDF = currSessionProcess.toDataFrame()


# ## define the universal start and end times for the entire 30 min session
# universalStartTime = np.int32(eventsProcess[0].OEStartTime) ## start time in milliseconds
# universalEndTime = np.int32(eventsProcess[-1].OEEndTime) ## end time in milliseconds

# ## construct an empty matrix to store the pupil diameter, stimuli, and the spike times for all the cells
# matRows = len(cellEnsemble) + 2 ## number of rows in the matrix; +2 for pupil diameter and stimuli
# matCols = universalEndTime - universalStartTime ## number of columns in the matrix
# dataMatrix = np.zeros((matRows, matCols))

# ## construct the matrix
# for trial in range(len(eventsProcess)): ## i'm calling trial instead of event because events are already defined elsewhere
#     currentEvent = eventsProcess[trial]

#     currentStartTime = currentEvent.OEStartTime
#     currentEndTime = currentEvent.OEEndTime

#     ## convert the start and stop times to columns in the matrix
#     currentStartCol = np.int32(currentStartTime - universalStartTime)
#     currentEndCol = np.int32(currentEndTime - universalStartTime)

#     ## insert frequency of the stimulus in the matrix
#     dataMatrix[0, currentStartCol:currentEndCol] = currentEvent.event.frequency

#     ## insert the pupil diameter in the matrix
#     frameTimeDiff = np.max(np.diff(currentEvent.frameTimes)) ## this is the time difference between the frames in milliseconds
#     for frameID in range(len(currentEvent.pupilDiameter)):
#         frameTime = currentEvent.frameTimes[frameID] ## the time of the frame in milliseconds))
#         frameColStart = np.int32(frameTime - universalStartTime) + 1 ## start column for the frame -> somehow the universalStartTime is 1 ms ahead of the first frame time
#         frameColEnd = np.int32(frameColStart + frameTimeDiff) ## end column for the frame assuming the pupil diameter is constant for the duration of the frame -> approximating now
#         dataMatrix[1, frameColStart:frameColEnd] = currentEvent.pupilDiameter[frameID] ## insert the pupil diameter in the matrix

#     ## now update the matrix with the spiketimes
#     for cellNum in range(len(cellEnsemble)):
#         currSpikeTimes = np.int32((currentEvent.spikeTimes[cellNum]) * ms) ## extract the spike times for the current cell
#         spikeCol = currSpikeTimes - universalStartTime ## convert the spike times to columns in the matrix
#         dataMatrix[cellNum + 2, spikeCol] = 1 ## this one represents the spike at that particular ms


# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

## plot the spikes
# for cellnum in range(dataMatrix[2:].shape[0]):
#     x = np.arange(dataMatrix.shape[1])
#     y = dataMatrix[cellnum + 2] * cellnum
#     ax.plot(x, y, 'k.', markersize=1)

## plot the pupil diameter
## convert nans to zeros for pupil diameter
# pupilCopy = np.copy(dataMatrix[1])
# pupilCopy[np.isnan(pupilCopy)] = 0
# ax.plot(pupilCopy, 'r', markersize=1)

## plot the stimuli
# stimuli = dataMatrix[0].copy()
# stimuli[np.isnan(stimuli)] = 0
# ax.plot(stimuli[0:100000], 'b', markersize=1)

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Cell Number')
# plt.show()

# save the dataMatrix as a .npy file
sessionProcessDF.to_pickle(os.path.join(DATA_PATH_AROUSAL, "sessionProcessDF.pkl"))
