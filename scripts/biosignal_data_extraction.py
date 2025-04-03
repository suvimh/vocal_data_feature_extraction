
import os
import json
import numpy as np
import pandas as pd

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
        pd.DataFrame: DataFrame containing sampled biosignal data for each channel.
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
    
    # Convert sampled_bio_data dictionary to DataFrame
    biosignal_df = pd.DataFrame(sampled_bio_data)

    
    return biosignal_df, sample_rate


def clean_biosignal_data(cleaned_time, sampled_bio_data, sample_rate, frame_duration_ms = 10):
    """
    Cleans the sampled biosignal data by keeping only the segments that correspond to the cleaned times.

    Args:
        cleaned_time (ndarray): The cleaned time data in seconds.
        sampled_bio_data (pd.DataFrame): The DataFrame containing the sampled biosignal data.
        sample_rate (int): The sampling rate of the biosignal data.

    Returns:
        pd.DataFrame: DataFrame containing cleaned biosignal data for each channel.
    """
    cleaned_bio_data_df = pd.DataFrame()

    cleaned_time_ms = np.array(cleaned_time) * 1000
    sampled_times = np.arange(0,len(sampled_bio_data) * frame_duration_ms, frame_duration_ms)

    if len(cleaned_time_ms) == len(sampled_bio_data):
      return sampled_bio_data
    
    for channel_name in sampled_bio_data.columns:
        signal_data = sampled_bio_data[channel_name]
        cleaned_signal_data = []

        for t in cleaned_time_ms:
            frame_index = np.argmin(np.abs(sampled_times - t))
            cleaned_signal_data.append(signal_data.iloc[frame_index])

        cleaned_bio_data_df[channel_name] = cleaned_signal_data

    return cleaned_bio_data_df


def get_biosignal_data_for_frames(cleaned_time, file_path, frame_duration_ms):

  """
    Extracts biosignal data for frames.

    Args:
        cleaned_time (float): The time duration for cleaning the biosignal data.
        file_path (str): The path to the biosignal data file.
        frame_duration_ms (int): The duration of each frame in milliseconds.

    Returns:
        pandas.DataFrame: The cleaned biosignal data for the frames.
  """
  sampled_bio_data_df, sample_rate = sample_biosignal_data(file_path, frame_duration_ms)
  if not sampled_bio_data_df.empty:
      cleaned_bio_data_df = clean_biosignal_data(cleaned_time, sampled_bio_data_df, sample_rate, frame_duration_ms)
      return cleaned_bio_data_df
  else:
      empty_frame = pd.DataFrame(index=range(len(cleaned_time)))
      return empty_frame
  