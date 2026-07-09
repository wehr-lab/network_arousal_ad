## import standard libraries 
import os 
import sys 
import glob 
import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

## add path to custom libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

## import the custom libraries 
from utils.extract_vals import extract_value 
from utils.strcmp import strcmp
from behavior import Behavior
from events import Events
from session import Session
from cell import Cell 
from cell_ensemble import CellEnsemble
from events_process import EventsProcess
import ephys_analysis


## define variables 
EPHYS_DIR_PATH = "/Volumes/projects/5XFAD/Rig3Phys/2025-02-03_8-37-45_mouse-3359/2025-02-03_08-37-56mouse-3359/"

## load the sorted units file
sortedUnitsFilePath = glob.glob(os.path.join(EPHYS_DIR_PATH, "Sort*.mat"))[0]
sortedUnitsFile = sio.loadmat(sortedUnitsFilePath, struct_as_record=True) 
sortedUnitsFile = sortedUnitsFile['SortedUnits']
sortedUnitsFile = sortedUnitsFile.reshape(-1) 

## create a list of Cell objects to pass to CellEnsemble class 
cells = []
for cell in sortedUnitsFile:
    cells.append(Cell(cell))

## create a CellEnsemble object
cellEnsemble = CellEnsemble(cells)

cellEnsembleDF = cellEnsemble.toDataFrame()

# ## for sanity check let's plot the spike raster for all the cells 
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# for cellnum, cellSpike in zip(cellEnsembleDF["cellnum"], cellEnsembleDF["spiketimes"]):
#     y = np.ones_like(cellSpike) * cellnum
#     ax.plot(cellSpike, y, 'k.', markersize=1)

# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Cell Number')
# plt.show()
# plt.close()

## create a matrix of the spike times for each cell
ms = 1e3 
spikeTimes = cellEnsembleDF["spiketimes"].values
# spikeTimes = spikeTimes * ms ## convert to ms -> not doing this because large matrix loading is taking a long time 

minTime = 0
maxtime = np.max([np.max(cellSpike) for cellSpike in spikeTimes]) + 10 ## added 1000 ms so that 1 sec padding added 
numCells = len(spikeTimes)
spikeMatrix = np.zeros((numCells, int(maxtime - minTime)))

for cellnum, cellSpike in enumerate(spikeTimes):
    spikeMatrix[cellnum], _ = np.histogram(cellSpike, bins=np.arange(minTime, maxtime, 1))

## run convolution of the entire data 
spikeMatrixConv = ephys_analysis.convolve_spikes(spikeMatrix, sigma=2) ## why sigma = 2?

## plot the convolved spike matrix
# plt.plot(spikeMatrixConv[0, :])

## subset the data
subsetLengths = np.array(np.arange(0, 125, 25))[1:]
subsetData = {}
for subsetLength in subsetLengths:
    subsetData[subsetLength] = ephys_analysis.subset_data(spikeMatrixConv, percentage=subsetLength)

correlationMatrices = {}
for subsetLength in subsetData.keys():
    correlationMatrices[subsetLength] = np.corrcoef(subsetData[subsetLength])

## plot the correlation matrices
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes = axes.flatten()
for i, subsetLength in enumerate(correlationMatrices.keys()):
    np.fill_diagonal(correlationMatrices[subsetLength], 0)
    sns.heatmap(correlationMatrices[subsetLength], ax=axes[i], cmap="coolwarm", square=True) 
    axes[i].set_title("Subset Length: " + str(subsetLength))
plt.tight_layout()
plt.show()


# ## compute the cosine similarity between the correlation matrices of the subset data and the full length data
# cosineSim = {}

# for subsetLength in correlationMatrices.keys():
#     refCorr = correlationMatrices[100]
#     subsetCorr = correlationMatrices[subsetLength]
#     cosineSim[(subsetLength, 100)] = np.dot(refCorr.flatten(), subsetCorr.flatten()) / (np.linalg.norm(refCorr) * np.linalg.norm(subsetCorr))

# ## plot the cosine similarity between the correlation matrices of the subset data and the full length data
# time = [time[0] for time in cosineSim.keys()]
# cosine = [cosineSim[time] for time in cosineSim.keys()]

# plt.plot(time, cosine)
# plt.xlabel("Time (minutes)")
# plt.ylabel("Cosine similarity")
# plt.title("Cosine similarity between the correlation matrices of the subset data and the full length data")
# plt.show()

## compute the correlation of the correlation matrices 
corrSim = {}

for subsetLength in correlationMatrices.keys():
    refCorr = correlationMatrices[100]
    subsetCorr = correlationMatrices[subsetLength]
    corrSim[(subsetLength, 100)] = np.corrcoef(refCorr.flatten(), subsetCorr.flatten())[0, 1]

## plot the correlatiuon similarity between the correlation matrices of the subset data and the full length data
time = [time[0] for time in cosineSim.keys()]
corr = [corrSim[time] for time in corrSim.keys()]

plt.plot(time, corr)
plt.xlabel("Time (minutes)")
plt.ylabel("Correlation similarity")
plt.title("Correlation similarity between the correlation matrices of the subset data and the full length data")
plt.show()




