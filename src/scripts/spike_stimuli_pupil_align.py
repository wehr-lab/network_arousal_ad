## import standard libraries 
import os 
import sys 
import glob 
import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt

## add path to custom libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

## add path to custom libraries -> can directly import the libraries, no need to add path 
from utils.extract_vals import extract_value 
from utils.strcmp import strcmp
from behavior import Behavior
from events import Events
from session import Session
from cell import Cell 
from cell_ensemble import CellEnsemble
from events_process import EventsProcess

## define variables 
BONSAI_DIR_PATH = "/Volumes/projects/ToneDecodingArousal/2022-05-09_13-52-25_mouse-0951"

## load the .mat files 
behaviorFilePath = glob.glob(os.path.join(BONSAI_DIR_PATH, "Beh*.mat"))[0]
behaviorFile = Behavior(behaviorFilePath)

headFile = behaviorFile.get_head_data() 
reyeFile = behaviorFile.get_reye_data()

## define the ephys dir 
sessionName = reyeFile["dirName"]
sessionName = extract_value(sessionName, 'dirName')
ephysDirPath = os.path.join(BONSAI_DIR_PATH, sessionName)

## load the sortedUnits file
sortedUnitsFilePath = glob.glob(os.path.join(ephysDirPath, "Sort*.mat"))[0]
sortedUnitsFile = sio.loadmat(sortedUnitsFilePath, struct_as_record=True)
sortedUnitsFile = sortedUnitsFile['SortedUnits']
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

## for sanity check let's plot the spike raster for all the cells 
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for cellnum, cellSpike in zip(cellEnsembleDF["cellnum"], cellEnsembleDF["spiketimes"]):
#     y = np.ones_like(cellSpike) * cellnum
#     ax.plot(cellSpike, y, 'k.', markersize=1)

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Cell Number')
# plt.show()

## Extract stimulus onset and offset times 
ms = 1e3 ## convert time in seconds to milliseconds
timeWindow = [-500, 500] ## time window in milliseconds to extract spikes around the stimulus onset

## define the frame numbers when the first and last stimuli were presented 
sessionFrameNums = reyeFile["TTs"][0][0][0] ## frame numbers when the stimuli were presented
firstEventFrame = sessionFrameNums[0] ## first frame number when the first stimulus was presented
lastEventFrame = sessionFrameNums[-1] ## last frame number when the last stimulus was presented

## extract the time for OpenEphys for the first and last stimulus presentation
OEFirstEventTime = session.get_event(0).soundcard_trigger_timestamp_sec ## OE time in seconds when the first stimulus was presented
OELastEventTime = session.get_event(-1).soundcard_trigger_timestamp_sec ## OE time in seconds when the last stimulus was presented 

## to track the frame numbers with OE time, find the slope and intercept of FrameNumber vs OE time
slope = (lastEventFrame - firstEventFrame) / (OELastEventTime - OEFirstEventTime)
intercept = firstEventFrame - (slope * OEFirstEventTime)

## Store event specific information in a EventsProcess object. This is different from the Events object as it will store the frame numbers, spike times, pupil diameter, etc. for each event. 

eventsProcess = []
for trial in range(len(session)):
    if strcmp(session.get_event(trial).type, ('tone', 'whitenoise', 'silentsound')):
        if hasattr(session.get_event(trial), 'soundcard_trigger_timestamp_sec'):
            eventStartTime = session.get_event(trial).soundcard_trigger_timestamp_sec * ms ## convert time in seconds to milliseconds
        else:
            raise ValueError('Event does not have soundcard_trigger_timestamp_sec attribute.')

        ## now store all the event specific information in the EventsProcess object
        currEvent = EventsProcess(session.get_event(trial)) ## initialize with the Events object
        if not strcmp(session.get_event(trial).type, 'tone'):
            currEvent.event.frequency = np.nan 
        
        ## event (trial) specific start and stop times 
        currEvent.OEStartTime = np.int32(np.ceil(eventStartTime + timeWindow[0])) ## start time in milliseconds
        currEvent.OEEndTime = np.int32(np.ceil(eventStartTime + timeWindow[1]))
        currEvent.freqStart = np.int32(np.ceil(eventStartTime)) ## start time in of stimulus in milliseconds

        ##TODO check if np.ceil is the right function to use here

        currEvent.freqEnd = np.ceil(currEvent.freqStart + currEvent.event.duration) ## end time of stimulus in milliseconds; note that the duration is already in milliseconds

        ## find the frame number for the start and end time of the stimulus
        currEvent.frameRate = extract_value(reyeFile['vid'][0][0]['framerate']) ## lots of nesting, but checked and its correct
        frameRateInMs = (1/ms) * currEvent.frameRate ## frame rate in milliseconds
        currEvent.firstFrame = np.int32(np.ceil(sessionFrameNums[trial] + timeWindow[0] * frameRateInMs)) ## frame number for the start time of the stimulus
        currEvent.lastFrame = np.int32(np.ceil(sessionFrameNums[trial] + timeWindow[1] * frameRateInMs))

        ## extracting the spiketimes now 
        for cellNum in range(len(cellEnsemble)):
            currCell = cellEnsemble.get_cell(cellNum)
            currEvent.spikeTimes.append(currCell.spiketimes[(currCell.spiketimes >= currEvent.OEStartTime/ms) & (currCell.spiketimes <= currEvent.OEEndTime/ms)]) ## extract the spike times within the time window for each cell and append

        ## extract the pupil diameter for the event
        currEvent.pupilDiameter = pupilDiameter[currEvent.firstFrame:currEvent.lastFrame] ## extract the pupil diameter for the event -> note that this returns an array itself

        ## time of frames 
        currEvent.frameNums = np.arange(currEvent.firstFrame, currEvent.lastFrame + 1) ## frame numbers for the event
        currEvent.frameTimes = np.int32(((currEvent.frameNums - intercept) / slope) * ms) ## convert frame numbers to OE time in milliseconds 

        ## store timeWindow and frameWindow
        currEvent.timeWindow = timeWindow
        currEvent.frameWindow = [currEvent.firstFrame, currEvent.lastFrame]

        ## append the currEvent to the eventsProcess list
        eventsProcess.append(currEvent)


## define the universal start and end times for the entire 30 min session 
universalStartTime = np.int32(eventsProcess[0].OEStartTime) ## start time in milliseconds
universalEndTime = np.int32(eventsProcess[-1].OEEndTime) ## end time in milliseconds

## construct an empty matrix to store the pupil diameter, stimuli, and the spike times for all the cells
matRows = len(cellEnsemble) + 2 ## number of rows in the matrix; +2 for pupil diameter and stimuli
matCols = universalEndTime - universalStartTime ## number of columns in the matrix
dataMatrix = np.zeros((matRows, matCols))

## construct the matrix 
for trial in range(len(eventsProcess)): ## i'm calling trial instead of event because events are already defined elsewhere 
    currentEvent = eventsProcess[trial]

    currentStartTime = currentEvent.OEStartTime
    currentEndTime = currentEvent.OEEndTime

    ## convert the start and stop times to columns in the matrix 
    currentStartCol = np.int32(currentStartTime - universalStartTime)
    currentEndCol = np.int32(currentEndTime - universalStartTime)

    ## insert frequency of the stimulus in the matrix
    dataMatrix[0, currentStartCol:currentEndCol] = currentEvent.event.frequency

    ## insert the pupil diameter in the matrix
    frameTimeDiff = np.max(np.diff(currentEvent.frameTimes)) ## this is the time difference between the frames in milliseconds
    for frameID in range(len(currentEvent.pupilDiameter)):
        frameTime = currentEvent.frameTimes[frameID] ## the time of the frame in milliseconds))
        frameColStart = np.int32(frameTime - universalStartTime) + 1 ## start column for the frame -> somehow the universalStartTime is 1 ms ahead of the first frame time
        frameColEnd = np.int32(frameColStart + frameTimeDiff) ## end column for the frame assuming the pupil diameter is constant for the duration of the frame -> approximating now 
        dataMatrix[1, frameColStart:frameColEnd] = currentEvent.pupilDiameter[frameID] ## insert the pupil diameter in the matrix

    ## now update the matrix with the spiketimes 
    for cellNum in range(len(cellEnsemble)):
        currSpikeTimes = np.int32((currentEvent.spikeTimes[cellNum]) * ms) ## extract the spike times for the current cell  
        spikeCol = currSpikeTimes - universalStartTime ## convert the spike times to columns in the matrix
        dataMatrix[cellNum + 2, spikeCol] = 1 ## this one represents the spike at that particular ms


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
np.save(os.path.join(BONSAI_DIR_PATH, "dataMatrix.npy"), dataMatrix)