## import libraries 
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter
from scipy.ndimage import gaussian_filter1d


def load2P(dataPath):
    F = np.load(os.path.join(dataPath, "F.npy"), allow_pickle=True)
    Fneu = np.load(os.path.join(dataPath, "Fneu.npy"), allow_pickle=True)
    spks = np.load(os.path.join(dataPath, "spks.npy"), allow_pickle=True)
    stat = np.load(os.path.join(dataPath, "stat.npy"), allow_pickle=True)
    ops =  np.load(os.path.join(dataPath, "ops.npy"), allow_pickle=True)
    ops = ops.item() ## converts ops to a dictionary
    iscell = np.load(os.path.join(dataPath, "iscell.npy"), allow_pickle=True)
    return F, Fneu, spks, stat, ops, iscell

def getIndices(iscell, prob=None):
    """Return the indices of the cells that are classified as cells."""
    if prob is not None:
        indices = np.where((iscell[:, 1] >= prob) & (iscell[:, 0] == 1))[0]
    else:
        indices = np.where(iscell[:, 0] == 1)[0]
    return indices

def extractCells(fTraces, indices):
    """Return the ROIs that are cells."""
    return fTraces[indices]


def correct_neuropil(cell_data, npil_data, npil_coeff=0.7):
    """Correct the neuropil contamination in the cell data using the neuropil data."""
    return cell_data - npil_coeff * npil_data


def absFluoresce(data):
    fCorrected = np.zeros(data.shape)
    for ind in range(data.shape[0]): ## loop through the rois 
        if np.min(data[ind, :]) <= 0:
            fCorrected[ind, :] = data[ind, :] + np.abs(np.min(data[ind, :])) + 1
        else:
            fCorrected[ind, :] = data[ind, :] + 1
    return fCorrected


def computeDFF(data, f0=None):
    """Compute the dF/F for the data using the specified baseline."""
    if f0 is None:
        f0 = np.nanmean(data, axis=1)
    return (data - f0[:, np.newaxis]) / f0[:, np.newaxis]

def computeDFFSlide(data, window=100, percentile=20):
    """Compute the dF/F for the data using a sliding window."""
    f0 = percentile_filter(data, percentile=percentile, size=(1, window), mode='wrap')

    numCells, numFrames = data.shape
    dff = np.zeros((numCells, numFrames))

    dff = (data - f0) / f0
    ## trim for the first and last few time points because the f0 values are not accurate for those time points due to window effects
    trim_size = window // 2
    dff = dff[:, trim_size:-trim_size]
    return dff

## convolve the spikes with a gaussian filter
def convolve_spikes(spks, sigma=1):
    convolved_spks = np.zeros(spks.shape)
    for i in range(spks.shape[0]):
        convolved_spks[i, :] = gaussian_filter1d(spks[i, :], sigma)
    return convolved_spks


## define function to plot spikes
def plot_spikes_subset(data, time_points=50, cell_indices=None, y_lim=None):
    """Plot subset of spikes for a subset of cells. This is to see how the spikes look after dF/F correction."""
    if cell_indices is not None:
        cell_indices = np.array(list(cell_indices))
        for i in cell_indices:
            if i >= data.shape[0]:
                raise ValueError("Cell index out of range")
        someCells = np.array(cell_indices)
    else:
        someCells = np.array(list(np.arange(10)))
    
    num_rows = int(len(someCells) // np.sqrt(len(someCells)))
    num_cols = int(np.ceil(len(someCells) / num_rows))
    # print(num_rows, num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))  
    axes = axes.flatten()

    for i, cell_idx in enumerate(someCells):
        axes[i].plot(data[int(cell_idx), :time_points])
        axes[i].set_title("Cell: " + str(someCells[i] + 1))
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("dF/F")
        if y_lim is not None:
            axes[i].set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

def plot_trace(data, cell_indices=None, y_lim=None):
    """Plot the traces of the specified cells. These are the entire dF/F traces."""
    if cell_indices is not None:
        cell_indices = np.array(list(cell_indices))
        for i in cell_indices:
            if i >= data.shape[0]:
                raise ValueError("Cell index out of range")
        someCells = np.array(cell_indices)
    else:
        someCells = np.array(list(np.arange(10)))
    
    num_rows = int(len(someCells) // np.sqrt(len(someCells)))
    num_cols = int(np.ceil(len(someCells) / num_rows))
    # print(num_rows, num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))  
    axes = axes.flatten()

    for i, cell_idx in enumerate(someCells):
        axes[i].plot(data[int(cell_idx)])
        axes[i].set_title("Cell: " + str(someCells[i] + 1))
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("dF/F")
        if y_lim is not None:
            axes[i].set_ylim(y_lim)

    plt.tight_layout()
    plt.show()

def subset_data(data, numPoints=None, percentage=None):
    """Return a subset of the data with the specified number of time points (frames)."""
    if numPoints is None and percentage is None:
        raise ValueError("Either numPoints or percentage should be specified.")
    
    numCells, totalPoints = data.shape
    if numPoints is not None:

        data_subset = np.zeros((numCells, numPoints))

        for i in range(num_cells):
            data_subset[i] = data[i][:numPoints]

        return data_subset

    ## calculate the number of points to subset based on the percentage
    numPoints = int((percentage / 100) * totalPoints)
    
    ## ensure at least one point is selected
    numPoints = max(1, numPoints)

    ## create subset array
    data_subset = np.zeros((numCells, numPoints))

    ## Subset data for each cell
    for i in range(numCells):
        data_subset[i] = data[i][:numPoints]

    return data_subset


def binarize_matrix(matrix, threshold):
    binary_matrix = np.zeros(matrix.shape)
    binary_matrix[matrix >= threshold] = 1
    return binary_matrix