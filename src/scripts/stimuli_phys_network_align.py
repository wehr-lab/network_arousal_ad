import os 
import sys
from pathlib import Path
import scipy.io as sio
import numpy as np
import networkx as nx
import h5py 
import glob 
import matplotlib.pyplot as plt

## import custom modules 
repoDir = Path(__file__).resolve().parent.parent.parent ## dir of the repo root
# print("repoDir:", repoDir)
configDir = repoDir / "config"
sys.path.append(str(configDir))

## define path variables 
from settings import DATA_PATH, CODE_PATH
sys.path.append(str(CODE_PATH["2-photon"]))

from utils.util_funcs import extract_value, strcmp 
from ephys.behavior import Behavior
from ephys.cell import Cell
from ephys.cell_ensemble import CellEnsemble
from ephys.events import Events
from ephys.events_process import EventsProcess
from ephys.session import Session

## load .mat files and typecast to python classes 
DATA_PATH = Path(DATA_PATH["toneDecode"], "2022-05-10_14-02-30_mouse-0956")
# print("DATA_PATH:", DATA_PATH)

## load the .mat files 
behaviorFilePath = glob.glob(os.path.join(DATA_PATH, "Beh*.mat"))[0]
behaviorFile = Behavior(behaviorFilePath)

headFile = behaviorFile.get_head_data() 
reyeFile = behaviorFile.get_reye_data()

## define the ephys dir 
sessionName = reyeFile["dirName"]
sessionName = extract_value(sessionName, 'dirName')
ephysDirPath = os.path.join(DATA_PATH, sessionName)

## load the sortedUnits file
sortedUnitsFilePath = glob.glob(os.path.join(ephysDirPath, "Sort*.mat"))[0]
sortedUnitsFile = sio.loadmat(sortedUnitsFilePath, struct_as_record=True)
sortedUnitsFile = sortedUnitsFile['SortedUnits']
sortedUnitsFile = sortedUnitsFile.reshape(-1)

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
ms = 1e3 ## convert time in seconds to milliseconds
timeWindow = [-1000, 1000] ## time window in ms around stimulus onset to extract spike times

## define the frame numbers when the first and last stimuli were presented 
sessionFrameNums = reyeFile["TTs"][0, 0].reshape(-1) ## frame numbers when the stimuli were presented
firstEventFrame = sessionFrameNums[0] ## first frame number when the first stimulus was presented
lastEventFrame = sessionFrameNums[-1] ## last frame number when the last stimulus was presented

## extract the time for OpenEphys for the first and last stimulus presentation
OEFirstEventTime = session.get_event(0).soundcard_trigger_timestamp_sec ## OE time in seconds when the first stimulus was presented
OELastEventTime = session.get_event(-1).soundcard_trigger_timestamp_sec ## OE time in seconds when the last stimulus was presented 

## Calculate frame rate and create time alignment
print(f"First event frame: {firstEventFrame}, Last event frame: {lastEventFrame}")
print(f"First event time: {OEFirstEventTime:.3f}s, Last event time: {OELastEventTime:.3f}s")

## Calculate frames per second (frame rate)
framesPerSec = (lastEventFrame - firstEventFrame) / (OELastEventTime - OEFirstEventTime)
print(f"Calculated frame rate: {framesPerSec:.2f} frames/sec")

## Create frame number array for pupil data (assuming pupil data starts from frame 0)
pupil_frames = np.arange(len(pupilDiameter))  ## frame numbers for each pupil measurement

## Convert pupil frame numbers to ephys time (seconds)
## Method 1: Linear interpolation based on stimulus events
timePupilEphys_sec = OEFirstEventTime + (pupil_frames - firstEventFrame) / framesPerSec

## Method 2: Alternative calculation using intercept (more explicit)
intercept = firstEventFrame - (framesPerSec * OEFirstEventTime)
timePupilEphys_sec_alt = (pupil_frames - intercept) / framesPerSec

## Convert to milliseconds if needed
timePupilEphys_ms = timePupilEphys_sec * ms

## Create aligned pupil data structure
pupil_aligned = {
    'diameter': pupilDiameter,
    'frames': pupil_frames,
    'time_sec': timePupilEphys_sec,
    'time_ms': timePupilEphys_ms,
    'frame_rate': framesPerSec
}

print(f"Pupil data aligned: {len(pupilDiameter)} samples")
print(f"Time range: {timePupilEphys_sec[0]:.3f}s to {timePupilEphys_sec[-1]:.3f}s")
print(f"Duration: {timePupilEphys_sec[-1] - timePupilEphys_sec[0]:.3f}s")

## Extract spike data aligned to pupil timing
print("\n=== EXTRACTING SPIKE DATA ===")

# Get spike times for all cells
all_spike_times = []
cell_ids = []
spike_counts_binned = []

for i, cell in enumerate(cells):
    spike_times_sec = cell.spike_times_sec  # Assuming spike times are in seconds
    all_spike_times.extend([(t, i) for t in spike_times_sec])  # (time, cell_id) pairs
    cell_ids.append(i)
    print(f"Cell {i}: {len(spike_times_sec)} spikes")

# Sort spikes by time
all_spike_times.sort(key=lambda x: x[0])
spike_times_only = [st[0] for st in all_spike_times]
spike_cell_ids = [st[1] for st in all_spike_times]

print(f"Total spikes across all cells: {len(all_spike_times)}")
print(f"Spike time range: {min(spike_times_only):.3f}s to {max(spike_times_only):.3f}s")

# Create binned spike data to match pupil sampling rate
pupil_sampling_interval = np.median(np.diff(timePupilEphys_sec))  # Median interval between pupil samples
print(f"Pupil sampling interval: {pupil_sampling_interval*1000:.1f}ms")

# Bin spikes into pupil time bins
pupil_time_bins = timePupilEphys_sec
spike_rate_per_cell = np.zeros((len(cells), len(pupil_time_bins)))

for cell_idx in range(len(cells)):
    cell_spikes = [st[0] for st in all_spike_times if st[1] == cell_idx]
    if len(cell_spikes) > 0:
        # Count spikes in each pupil time bin
        spike_counts, _ = np.histogram(cell_spikes, bins=np.concatenate([pupil_time_bins, [pupil_time_bins[-1] + pupil_sampling_interval]]))
        spike_rate_per_cell[cell_idx, :] = spike_counts / pupil_sampling_interval  # Convert to firing rate (Hz)

# Calculate population firing rate
population_firing_rate = np.mean(spike_rate_per_cell, axis=0)
total_spike_count = np.sum(spike_rate_per_cell, axis=0)

print(f"Population firing rate range: {np.min(population_firing_rate):.2f} to {np.max(population_firing_rate):.2f} Hz")

## Create comprehensive aligned data structure
neural_pupil_data = {
    'pupil': {
        'diameter': pupilDiameter,
        'time_sec': timePupilEphys_sec,
        'time_ms': timePupilEphys_ms,
        'frames': pupil_frames
    },
    'spikes': {
        'all_spike_times': spike_times_only,
        'all_spike_cells': spike_cell_ids,
        'spike_rate_per_cell': spike_rate_per_cell,
        'population_firing_rate': population_firing_rate,
        'total_spike_count': total_spike_count,
        'time_bins': pupil_time_bins
    },
    'stimuli': {
        'times_sec': [session.get_event(i).soundcard_trigger_timestamp_sec for i in range(len(sessionFrameNums))],
        'frames': sessionFrameNums
    },
    'frame_rate': framesPerSec
}

## Validation: Check alignment at stimulus times
print("\n=== VALIDATION: Pupil alignment at stimulus times ===")
stimulus_times_sec = [session.get_event(i).soundcard_trigger_timestamp_sec for i in range(len(sessionFrameNums))]
stimulus_frames = sessionFrameNums

for i in range(min(5, len(stimulus_times_sec))):  # Check first 5 stimuli
    # Find closest pupil sample to this stimulus time
    closest_idx = np.argmin(np.abs(timePupilEphys_sec - stimulus_times_sec[i]))
    closest_time = timePupilEphys_sec[closest_idx]
    closest_frame = pupil_frames[closest_idx]
    expected_frame = stimulus_frames[i]
    
    print(f"Stimulus {i+1}: OE time = {stimulus_times_sec[i]:.3f}s, frame = {expected_frame}")
    print(f"  Closest pupil: time = {closest_time:.3f}s, frame = {closest_frame}, error = {abs(closest_time - stimulus_times_sec[i])*1000:.1f}ms")

## Create comprehensive neural-pupil visualization
try:
    stimulus_times_sec = neural_pupil_data['stimuli']['times_sec']
    
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Pupil diameter over time
    plt.subplot(4, 1, 1)
    plt.plot(timePupilEphys_sec, pupilDiameter, 'b-', alpha=0.8, linewidth=1.5, label='Pupil diameter')
    
    # Mark stimulus times
    for i, stim_time in enumerate(stimulus_times_sec[:20]):  # First 20 stimuli
        plt.axvline(stim_time, color='red', alpha=0.4, linestyle='--', linewidth=1)
    
    plt.ylabel('Pupil Diameter\n(pixels)')
    plt.title('Neural Activity and Pupil Dynamics Over Time')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([timePupilEphys_sec[0], timePupilEphys_sec[-1]])
    
    # Plot 2: Population firing rate
    plt.subplot(4, 1, 2)
    plt.plot(pupil_time_bins, population_firing_rate, 'g-', alpha=0.8, linewidth=1, label='Population firing rate')
    plt.fill_between(pupil_time_bins, population_firing_rate, alpha=0.3, color='green')
    
    # Mark stimulus times
    for i, stim_time in enumerate(stimulus_times_sec[:20]):
        plt.axvline(stim_time, color='red', alpha=0.4, linestyle='--', linewidth=1)
    
    plt.ylabel('Population\nFiring Rate (Hz)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([timePupilEphys_sec[0], timePupilEphys_sec[-1]])
    
    # Plot 3: Spike raster plot (sample of cells)
    plt.subplot(4, 1, 3)
    n_cells_to_plot = min(20, len(cells))  # Plot up to 20 cells
    cell_indices_to_plot = np.linspace(0, len(cells)-1, n_cells_to_plot, dtype=int)
    
    for plot_idx, cell_idx in enumerate(cell_indices_to_plot):
        cell_spikes = [st[0] for st in all_spike_times if st[1] == cell_idx]
        if len(cell_spikes) > 0:
            plt.eventplot([cell_spikes], lineoffsets=plot_idx, linelengths=0.8, 
                         colors='black', alpha=0.7, linewidths=0.8)
    
    # Mark stimulus times
    for i, stim_time in enumerate(stimulus_times_sec[:20]):
        plt.axvline(stim_time, color='red', alpha=0.4, linestyle='--', linewidth=1)
    
    plt.ylabel('Cell ID\n(sample)')
    plt.yticks(range(0, n_cells_to_plot, max(1, n_cells_to_plot//5)), 
               [f'{cell_indices_to_plot[i]}' for i in range(0, n_cells_to_plot, max(1, n_cells_to_plot//5))])
    plt.grid(True, alpha=0.3)
    plt.xlim([timePupilEphys_sec[0], timePupilEphys_sec[-1]])
    
    # Plot 4: Cross-correlation analysis
    plt.subplot(4, 1, 4)
    
    # Calculate correlation between pupil diameter and population firing rate
    # First, interpolate firing rate to match pupil sampling
    from scipy import interpolate
    from scipy.stats import pearsonr
    
    # Interpolate population firing rate to pupil time points
    f_interp = interpolate.interp1d(pupil_time_bins, population_firing_rate, 
                                   kind='linear', bounds_error=False, fill_value=0)
    firing_rate_interp = f_interp(timePupilEphys_sec)
    
    # Remove any NaN values
    valid_indices = ~(np.isnan(firing_rate_interp) | np.isnan(pupilDiameter))
    if np.sum(valid_indices) > 10:  # Need enough valid points
        pupil_clean = pupilDiameter[valid_indices]
        firing_clean = firing_rate_interp[valid_indices]
        
        # Calculate sliding window correlation
        window_size = min(1000, len(pupil_clean) // 10)  # 10% of data or 1000 points
        correlations = []
        correlation_times = []
        
        for i in range(0, len(pupil_clean) - window_size, window_size // 2):
            end_idx = i + window_size
            if end_idx <= len(pupil_clean):
                corr, p_val = pearsonr(pupil_clean[i:end_idx], firing_clean[i:end_idx])
                correlations.append(corr)
                correlation_times.append(timePupilEphys_sec[valid_indices][i + window_size//2])
        
        plt.plot(correlation_times, correlations, 'purple', linewidth=2, alpha=0.8, 
                label=f'Pupil-Firing Rate Correlation')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Overall correlation
        overall_corr, overall_p = pearsonr(pupil_clean, firing_clean)
        plt.axhline(y=overall_corr, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Overall r={overall_corr:.3f}, p={overall_p:.3f}')
    
    # Mark stimulus times
    for i, stim_time in enumerate(stimulus_times_sec[:20]):
        plt.axvline(stim_time, color='red', alpha=0.4, linestyle='--', linewidth=1)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Correlation\nCoefficient')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([timePupilEphys_sec[0], timePupilEphys_sec[-1]])
    plt.ylim([-1, 1])
    
    plt.tight_layout()
    plt.show()
    print(f"Comprehensive neural-pupil plot saved to: {DATA_PATH / 'neural_pupil_alignment.png'}")
    
    # Print summary statistics
    print(f"\n=== NEURAL-PUPIL ALIGNMENT SUMMARY ===")
    print(f"Data duration: {timePupilEphys_sec[-1] - timePupilEphys_sec[0]:.1f} seconds")
    print(f"Pupil samples: {len(pupilDiameter)}")
    print(f"Total spikes: {len(all_spike_times)}")
    print(f"Number of cells: {len(cells)}")
    print(f"Number of stimuli: {len(stimulus_times_sec)}")
    if 'overall_corr' in locals():
        print(f"Overall pupil-firing rate correlation: r={overall_corr:.3f}, p={overall_p:.3f}")
    
except Exception as e:
    print(f"Could not create comprehensive plot: {e}")
    import traceback
    traceback.print_exc()

print("run complete!")