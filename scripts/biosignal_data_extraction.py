'''
    EXTRACTING BIOSIGNAL DATA FROM VIDEO FILES IN 10MS FRAMES
'''

import os
import json
import numpy as np

def get_biosignal_sample_rate(biosignal_data):
    """
    Calculate the sample rate of the biosignal data.

    Args:
        biosignal_data (dict): A dictionary containing the biosignal data.

    Returns:
        int: The sample rate of the biosignal data.

    Raises:
        ValueError: If the sample rates are not consistent among all channels.
    """

    sample_rate = None
    for _, channels in biosignal_data.items():
        for channel_info in channels.values():
            if sample_rate is None:
                sample_rate = channel_info["Sample Rate"]
            elif sample_rate != channel_info["Sample Rate"]:
                raise ValueError("Sample rates are not consistent among all channels.")
    return sample_rate

def get_biosignal_data_for_frames(cleaned_time, file_path):
    """
    Cleans the biosignal data by keeping only the segments that correspond to the cleaned times.

    Args:
        cleaned_time (ndarray): The cleaned time data in seconds.
        bio_data (dict): The dictionary containing the biosignal data.
        sampling_rate (int): The sampling rate of the biosignal data in Hz (default is 1000Hz).

    Returns:
        dict: Dictionary containing cleaned biosignal data for each channel.
    """

    file = open(file_path)
    bio_data = json.load(file)
    sample_rate = get_biosignal_sample_rate(bio_data)

    cleaned_bio_data = {}

    for __, channels in bio_data.items():
        for channel_name, channel_info in channels.items():
            signal_data = np.array(channel_info['Signal Data']).flatten()
            cleaned_sample_indices = (np.array(cleaned_time) * sample_rate).astype(int)
            cleaned_signal_data = [signal_data[idx] for idx in cleaned_sample_indices if idx < len(signal_data)]
            cleaned_bio_data[channel_name] = cleaned_signal_data
    
    return cleaned_bio_data

'''
    Function for extracting biosignal data with larger window than the default 10ms
'''
# def get_biosignal_data_for_frames(cleaned_time, file_path, frame_duration_ms=20):
#     """
#     Extracts mean and median values for each XX ms interval from the cleaned biosignal data.

#     Args:
#         cleaned_bio_data (dict): The dictionary containing cleaned biosignal data.
#         frame_duration_ms (int): Duration of each frame in milliseconds (default is 10ms).

#     Returns:
#         dict: Dictionary containing mean and median values for each channel.
#     """
#     framed_bio_data = {}

#     file = open(file_path)
#     bio_data = json.load(file)

#     sample_rate = get_biosignal_sample_rate(bio_data)
#     cleaned_bio_data = clean_biosignal_data(cleaned_time, bio_data, sample_rate)

#     for channel_name, signal_data in cleaned_bio_data.items():
#         samples_per_frame = (frame_duration_ms * sample_rate) // 1000

#         num_frames = len(signal_data) // samples_per_frame

#         means = []
#         medians = []
#         for i in range(num_frames):
#             start_idx = i * samples_per_frame
#             end_idx = start_idx + samples_per_frame
#             frame_data = signal_data[start_idx:end_idx]
#             means.append(int(np.mean(frame_data)))
#             medians.append(int(np.median(frame_data)))

#         framed_bio_data[channel_name] = {
#             'mean': means,
#             'median': medians
#         }

#     return framed_bio_data

