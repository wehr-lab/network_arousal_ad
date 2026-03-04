import numpy as np
import h5py
from scipy import interpolate


def extract_value(arr, fieldname=None):
    """Only run this function after importing numpy as np.

    input: arr, a numpy array or list
    output: the very first value from the array or list i.e. arr[0][0]...
    """
    if fieldname == "spiketimes":
        if isinstance(arr, np.ndarray):
            return np.array(arr[0])
        elif isinstance(arr, list):
            return arr  # Return the list as is
        else:
            return None  # Handle unsupported type

    if fieldname == "PupilDiameter":
        return arr[0][0][
            :, 0
        ]  ## I was lazy to write a proper function. This is a temporary solution

    while True:
        ## check if the current item is a NumPy array or a list
        if isinstance(arr, (np.ndarray, list)):
            if len(arr) > 0:  # Ensure the array or list is not empty
                arr = arr[0]
            else:
                return None  # Return None for empty arrays or lists
        # Check if the value is a NumPy integer or float
        elif isinstance(arr, (np.integer, np.floating, np.str_)):
            return arr.item()  # Convert to plain Python type
        # Check if the value is a plain Python integer or float
        elif isinstance(arr, (int, float)):
            return arr
        else:
            return None  # Return None for unsupported types


def strcmp(str1, str2):
    if str2 is not str or type(str2) in [list, tuple]:
        return str1 in str2
    else:
        return str1 == str2


def hdf5ToDict(hdf5FilePath, exclude=None, include=None):
    """Convert a HDF5 file (including groups and datasets) into a nested dictionary.

    Args:
        hdf5FilePath: full path to the HDF5 file
        exclude: list of keys to exclude from loading (default: None)
        include: list of keys to include (if provided, only these keys will be loaded) (default: None)

    Returns:
        A nested dictionary mirroring the HDF5 hierarchy, with datasets loaded into arrays.
        If include is specified, only those keys are loaded.
        If exclude is specified (and include is not), those keys are skipped.
    """
    if exclude is None:
        exclude = []
    if include is None:
        include = []

    def _recurse(h5obj):
        data = {}
        for key, item in h5obj.items():
            # If include list is provided, only process keys in the include list
            if include and key not in include:
                continue

            # Skip this key if it's in the exclude list (only if include is not specified)
            if not include and key in exclude:
                continue

            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                data[key] = _recurse(item)
        return data

    with h5py.File(hdf5FilePath, "r") as f:
        return _recurse(f)


def get_unique_dtypes(arr):
    """Get unique data types from a numpy array or list.

    Args:
        arr: a numpy array or list

    Returns:
        A set of unique data types found in the array.
    """
    if isinstance(arr, np.ndarray):
        return set(arr.dtype.type for x in np.ravel(arr))
    elif isinstance(arr, list):
        return set(type(x) for x in np.ravel(arr))
    else:
        raise TypeError("Input must be a numpy array or a list.")


def coarse_bin_1d(data, bin_size, method="sum"):
    """
    Coarse bin a 1D array by combining adjacent values.

    Parameters:
    -----------
    data : np.ndarray
        1D array to bin
    bin_size : int
        Number of adjacent elements to combine into each bin
    method : str
        How to combine values: 'sum', 'mean', 'max', 'min'

    Returns:
    --------
    binned_data : np.ndarray
        Coarse-binned 1D array
    """
    # Trim data to be divisible by bin_size
    n_bins = len(data) // bin_size
    trimmed_data = data[: n_bins * bin_size]

    # Reshape to (n_bins, bin_size) and apply aggregation
    reshaped = trimmed_data.reshape(n_bins, bin_size)

    if method == "sum":
        return np.sum(reshaped, axis=1)
    elif method == "mean":
        return np.mean(reshaped, axis=1)
    elif method == "max":
        return np.max(reshaped, axis=1)
    elif method == "min":
        return np.min(reshaped, axis=1)
    else:
        raise ValueError("method must be 'sum', 'mean', 'max', or 'min'")


def coarse_bin_overlapping(data, bin_size, step_size=None, method="sum"):
    """
    Coarse bin with overlapping windows (sliding window binning).

    Parameters:
    -----------
    data : np.ndarray
        1D array to bin
    bin_size : int
        Size of each bin window
    step_size : int, optional
        Step between bins (default: bin_size for non-overlapping)
    method : str
        How to combine values: 'sum', 'mean', 'max', 'min'

    Returns:
    --------
    binned_data : np.ndarray
        Overlapping binned 1D array
    """
    if step_size is None:
        step_size = bin_size

    n_bins = (len(data) - bin_size) // step_size + 1
    binned_data = []

    for i in range(n_bins):
        start_idx = i * step_size
        end_idx = start_idx + bin_size
        window = data[start_idx:end_idx]

        if method == "sum":
            binned_data.append(np.sum(window))
        elif method == "mean":
            binned_data.append(np.mean(window))
        elif method == "max":
            binned_data.append(np.max(window))
        elif method == "min":
            binned_data.append(np.min(window))

    return np.array(binned_data)


def coarse_bin_2D(data, bin_size=None, method=None):
    """
    Coarse bin a 2D array by combining adjacent values along columns.

    Parameters:
    -----------
    data : np.ndarray
        2D array to bin (rows, cols)
    bin_size : int, optional
        Number of adjacent columns to combine into each bin (default: 1)
    overlap : int, optional
        Currently not implemented, kept for compatibility
    method : str
        How to combine values: 'sum', 'mean', 'median'

    Returns:
    --------
    binned_data : np.ndarray
        Coarse-binned 2D array with shape (rows, n_bins)
    """
    rows, cols = data.shape
    if bin_size is None:
        bin_size = 1
    if method is None:
        method = "mean"

    # Calculate number of bins (truncate if not evenly divisible)
    n_bins = cols // bin_size

    if n_bins == 0:
        raise ValueError(
            f"bin_size ({bin_size}) is larger than number of columns ({cols})"
        )

    # Trim data to be evenly divisible by bin_size
    trimmed_cols = n_bins * bin_size
    data_trimmed = data[:, :trimmed_cols]

    # Reshape to (rows, n_bins, bin_size) for aggregation
    data_reshape = data_trimmed.reshape(rows, n_bins, bin_size)

    if method == "mean":
        data_binned = np.mean(data_reshape, axis=2)
    elif method == "median":
        data_binned = np.median(data_reshape, axis=2)
    elif method == "sum":
        data_binned = np.sum(data_reshape, axis=2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'median', or 'sum'")

    print(f"Binned 2D array: {data.shape} -> {data_binned.shape}")
    print(f"Trimmed {cols - trimmed_cols} columns to make evenly divisible")

    return data_binned


def coarse_bin_2D_overlapping(data, bin_size, step_size=None, method="mean"):
    """
    Coarse bin a 2D array with overlapping windows along columns.

    Parameters:
    -----------
    data : np.ndarray
        2D array to bin (rows, cols)
    bin_size : int
        Size of each bin window
    step_size : int, optional
        Step between bins (default: bin_size for non-overlapping)
    method : str
        How to combine values: 'sum', 'mean', 'median'

    Returns:
    --------
    binned_data : np.ndarray
        Overlapping binned 2D array
    """
    if step_size is None:
        step_size = bin_size

    rows, cols = data.shape
    n_bins = (cols - bin_size) // step_size + 1

    if n_bins <= 0:
        raise ValueError(f"bin_size ({bin_size}) is too large for {cols} columns")

    binned_data = np.zeros((rows, n_bins))

    for i in range(n_bins):
        start_idx = i * step_size
        end_idx = start_idx + bin_size
        window = data[:, start_idx:end_idx]

        if method == "sum":
            binned_data[:, i] = np.sum(window, axis=1)
        elif method == "mean":
            binned_data[:, i] = np.mean(window, axis=1)
        elif method == "median":
            binned_data[:, i] = np.median(window, axis=1)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'mean', 'median', or 'sum'"
            )

    print(f"Overlapping binned 2D array: {data.shape} -> {binned_data.shape}")
    print(f"Bin size: {bin_size}, Step size: {step_size}")

    return binned_data


def convertBinToSpikeTimes(binMat):
    """
    Convert the binary spikes matrix to spike times.
    """
    outMat = []
    for i in range(binMat.shape[0]):
        spikeTimes = np.where(binMat[i, :] == 1)[0]
        spikeTimes = spikeTimes / 1000  # Convert to seconds
        outMat.append(spikeTimes)
    return outMat


def getAttrList(obj):
    """
    Get a list of attributes of an object excluding private and callable attributes.
    """
    return [
        attr
        for attr in dir(obj)
        if not attr.startswith("_") and not callable(getattr(obj, attr))
    ]


# Convert spike times to spike counts in time bins
def spikeTimes_to_spikeCounts(spike_times_list, bin_size=0.1, time_window=None):
    """
    Convert list of spike times to binned spike counts

    Parameters:
    - spike_times_list: list of arrays, each containing spike times for one neuron
    - bin_size: time bin size in seconds
    - time_window: [start, end] time window, or None for auto-detect
    """
    # Determine time window
    if time_window is None:
        return ValueError("Please provide a time_window as [start, end]")
        # all_spikes = np.concatenate([st for st in spike_times_list if len(st) > 0])
        # time_window = [np.min(all_spikes), np.max(all_spikes)]

    # Create time bins
    time_bins = np.arange(time_window[0], time_window[1] + bin_size, bin_size)

    # Count spikes for each neuron in each bin
    spike_counts = np.zeros((len(spike_times_list), len(time_bins) - 1))

    for neuron_idx, spike_times in enumerate(spike_times_list):
        if len(spike_times) > 0:
            counts, _ = np.histogram(spike_times, bins=time_bins)
            spike_counts[neuron_idx, :] = counts

    return spike_counts  # Shape: (n_neurons, n_time_bins)


def get_consecutive_true_lengths(bool_array):
    """Calculate lengths of consecutive True segments in a boolean array"""
    # Add False padding at start and end
    padded = np.concatenate(([False], bool_array, [False]))

    # Find where values change (transitions)
    transitions = np.where(np.diff(padded))[0]

    # Extract lengths of True segments (every other pair of transitions)
    if (
        len(transitions) > 0 and padded[transitions[0] + 1]
    ):  # First transition is False->True
        true_lengths = transitions[1::2] - transitions[::2]
    else:  # First transition is True->False (starts with True)
        if len(transitions) > 1:
            true_lengths = transitions[1::2] - transitions[::2]
        else:
            true_lengths = np.array([])

    return true_lengths


def interpolate_nans(data):
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(data)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:  # Need at least 2 points to interpolate
        return data

    # Create interpolation function
    interp_func = interpolate.interp1d(
        valid_indices,
        data[valid_mask],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    print(type(interp_func))

    # Apply to all indices
    all_indices = np.arange(len(data))
    interpolated = interp_func(all_indices)

    return interpolated


def list_keys(d, parent_key=""):
    """Recursively list all keys in a nested dictionary."""
    keys = []
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        keys.append(full_key)
        if isinstance(v, dict):
            keys.extend(list_keys(v, full_key))
    return keys


def get_attr_list(obj):
    """
    Get a list of attributes of an object excluding private and callable attributes.
    """
    return [
        attr
        for attr in dir(obj)
        if not attr.startswith("_") and not callable(getattr(obj, attr))
    ]
