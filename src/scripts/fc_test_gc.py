## import standard libraries 
import os 
import sys 
import glob 
import numpy as np 
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg as la
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

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

## subset the data
subsetLengths = np.array(np.arange(0, 125, 25))[1:]
subsetData = {}
for subsetLength in subsetLengths:
    subsetData[subsetLength] = ephys_analysis.subset_data(spikeMatrixConv, percentage=subsetLength)
    subsetData[subsetLength] = subsetData[subsetLength].T

## plot the correlation matrix to compare to GC matrix 
corrMAT = np.corrcoef(subsetData[100].T)

## from here on we will run the GC analysis on the subset data

## first test for stationarity for the full data only 
## plot the spike raster for all the cells
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for cellnum, cellSpike in zip(cellEnsembleDF["cellnum"], cellEnsembleDF["spiketimes"]):
    y = np.ones_like(cellSpike) * cellnum
    ax.plot(cellSpike, y, 'k.', markersize=1)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Cell Number')
plt.show()
# plt.close()



## skipping the stationarity test for now

## run the GC analysis on the subset data
# gcMat = {} ## to store all gc mats for different lengths of time 

gcNetwork = np.zeros((numCells, numCells))

time, numCells = subsetData[100].shape
neurons = [f'Neuron_{i}' for i in range(numCells)]
dfSpikes = pd.DataFrame(subsetData[100], columns=neurons)

## optimal lag for GC
modelVAR = VAR(dfSpikes)
resultsVAR = modelVAR.fit(maxlags=15, ic='aic')
optimalLag = resultsVAR.k_ar
print(f"Optimal lag for GC: {optimalLag}")

## initialize the GC matrix
gcMat = np.zeros((numCells, numCells))

## run the GC analysis
for caused in range(numCells):
    for causing in range(numCells):
        if caused != causing:
            ## run the granger causality test
            testResults = resultsVAR.test_causality(
                caused=neurons[caused],
                causing=neurons[causing],
                kind='f',
            )
            pVal = testResults.pvalue
            gcMat[causing, caused] = 1 if pVal < 0.05 else 0 

## plot the correlation matrix and the GC matrix -> just for visual comparison 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

im1 = axs[0].imshow(corrMAT, cmap='coolwarm', aspect='auto')
axs[0].set_title("Correlation Connectivity")
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(gcMat, cmap='coolwarm', aspect='auto')
axs[1].set_title("Causal Network")
fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()










