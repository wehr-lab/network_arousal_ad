import numpy as np 

def extract_ttl_frames(sync_file_path):
    sync_file = sio.loadmat(sync_file_path)
    sync_info = sync_file['info'].item()
    frames_ttl = np.array(sync_info[0])
    frames_ttl = frames_ttl.flatten()
    send_ttl = frames_ttl[::2]
    receive_ttl = frames_ttl[1::2]
    ttl_frames = np.vstack((send_ttl, receive_ttl)).T
    ttl_frames = ttl_frames[:-1, :] ## remove the last frame because it is not a full frame
    return ttl_frames


def extract_trials(frequency_array):
    freq_trials = {}
    for freq in np.unique(frequency_array):
        freq_trials[freq] = np.where(frequency_array == freq)[0]
    num_trials = frequency_array.shape[0]
    uniq_freq = np.unique(frequency_array).shape[0]
    freq_np = np.zeros((num_trials, uniq_freq))
    for i, freq in enumerate(freq_trials.keys()):
        freq_np[freq_trials[freq], i] = 1
    return (np.unique(frequency_array), freq_np)


def extract_traces(ephys_data, trial_frames, time_range, cell):
    trial_traces = np.zeros((trial_frames.shape[0], time_range[1] - time_range[0]))
    for i, trial in enumerate(trial_frames[:, 0]):
        start_frame = trial + time_range[0]
        end_frame = trial + time_range[1]
        trial_traces[i, :] = ephys_data[cell, start_frame:end_frame]
    return trial_traces


def mean_trace(trial_traces):
    return np.mean(trial_traces, axis=0)

def traces_all_frequencies():
    pass 

def plot_tuning_curve():
    pass

