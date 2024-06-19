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


def sample_biosignal_data(file_path, frame_duration_ms=10):
    """
    Samples the biosignal data by taking the midpoint value for each frame of the given duration.

    Args:
        file_path (str): The file path to the biosignal data.
        frame_duration_ms (int, optional): The duration of each frame in milliseconds. Default is 10 ms.

    Returns:
        dict: Dictionary containing sampled biosignal data for each channel.
        int: The sampling rate of the biosignal data.
    """

    with open(file_path, 'r') as file:
        bio_data = json.load(file)

    sample_rate = get_biosignal_sample_rate(bio_data)
    frame_duration_s = frame_duration_ms / 1000  # Convert frame duration to seconds
    sampled_bio_data = {}

    for _, channels in bio_data.items():
        for channel_name, channel_info in channels.items():
            signal_data = np.array(channel_info['Signal Data']).flatten()
            num_samples = int(len(signal_data) / sample_rate / frame_duration_s)
            sampled_signal_data = []

            for i in range(num_samples):
                start_index = int(i * frame_duration_s * sample_rate)
                end_index = start_index + int(frame_duration_s * sample_rate)
                if end_index > len(signal_data):
                    break
                frame_signal_data = signal_data[start_index:end_index]
                midpoint_value = frame_signal_data[len(frame_signal_data) // 2]
                sampled_signal_data.append(midpoint_value)

            sampled_bio_data[channel_name] = sampled_signal_data
    
    return sampled_bio_data, sample_rate


def clean_biosignal_data(cleaned_time, sampled_bio_data, sample_rate):
    """
    Cleans the sampled biosignal data by keeping only the segments that correspond to the cleaned times.

    Args:
        cleaned_time (ndarray): The cleaned time data in seconds.
        sampled_bio_data (dict): The dictionary containing the sampled biosignal data.
        sample_rate (int): The sampling rate of the biosignal data.

    Returns:
        dict: Dictionary containing cleaned biosignal data for each channel.
    """
    cleaned_bio_data = {}
    cleaned_time_ms = np.array(cleaned_time) * 1000 
    sampled_times = np.arange(len(next(iter(sampled_bio_data.values())))) * 1000 / sample_rate 

    for channel_name, signal_data in sampled_bio_data.items():
        cleaned_signal_data = []
        for t in cleaned_time_ms:
            frame_index = np.argmin(np.abs(sampled_times - t))
            cleaned_signal_data.append(signal_data[frame_index])
        cleaned_bio_data[channel_name] = cleaned_signal_data
    
    return cleaned_bio_data


def get_biosignal_data_for_frames(cleaned_time, file_path, frame_duration_ms):
    """
    Extracts biosignal data for each frame of the given duration.

    Args:
        cleaned_time (ndarray): The cleaned time data in seconds.
        file_path (str): The file path to the biosignal data.
        frame_duration_ms (int): The duration of each frame in milliseconds.

    Returns:
        dict: Dictionary containing biosignal data for each channel.
    """
    sampled_bio_data, sample_rate = sample_biosignal_data(file_path, frame_duration_ms)
    cleaned_bio_data = clean_biosignal_data(cleaned_time, sampled_bio_data, sample_rate)
    return cleaned_bio_data