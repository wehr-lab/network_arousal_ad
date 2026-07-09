import h5py
import numpy as np


def h5_to_flat_dict(h5_group):
    """Recursively flatten HDF5 structure into a dictionary with dataset names as keys and dataset values as values."""
    flat_dict = {}
    
    for key in h5_group.keys():
        item = h5_group[key]
        
        # If the item is a group, recurse into it
        if isinstance(item, h5py.Group):
            flat_dict.update(h5_to_flat_dict(item))
        # If the item is a dataset, store its value
        elif isinstance(item, h5py.Dataset):
            # Check if the dataset is a scalar (no dimensions)
            if item.shape == ():  # Scalar dataset
                flat_dict[key] = item[()]  # Get scalar value without slicing
            else:
                flat_dict[key] = item[:]  # Read the dataset as an array
    
    return flat_dict


def load_behavior_data(file_path):
    """Load behavior data from an HDF5 file and return it as a dictionary."""
    with h5py.File(file_path, 'r') as h5_file:
        behavior_data = h5_to_flat_dict(h5_file)
    
    return behavior_data


def extract_trials(frequency_array):
    """Extract the trials for each unique frequency in the frequency array."""
    freq_trials = {}
    for freq in np.unique(frequency_array):
        freq_trials[freq] = np.where(frequency_array == freq)[0]
    num_trials = frequency_array.shape[0]
    uniq_freq = np.unique(frequency_array).shape[0]
    freq_np = np.zeros((num_trials, uniq_freq))
    for i, freq in enumerate(freq_trials.keys()):
        freq_np[freq_trials[freq], i] = 1
    return freq_np