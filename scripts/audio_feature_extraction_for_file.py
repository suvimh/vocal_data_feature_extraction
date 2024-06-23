"""
  Audio feature extraction of all frames of one file using Essentia.
"""

import essentia.standard as es
import pandas as pd
import numpy as np
import scripts.spectral_feature_extraction as spectral
import scripts.pitch_extraction as pitch
import scripts.utils as utils


def spectrum(frames):
    """
    Parameters:
        frames (list): A list of audio frames.

    Returns:
        list: A list of spectra, where each spectrum corresponds to a frame in the input list.
    """
    spectrum = es.Spectrum()
    return [spectrum(frame) for frame in frames]


def rms_energy(frames):
    """
    Calculate the Root Mean Square (RMS) energy of audio frames.

    Parameters:
        frames (np.array): A list of audio frames.

    Returns:
        np.array: An array of RMS energy values for each frame.
    """
    rms = es.RMS()
    return np.array([rms(frame) for frame in frames], dtype=np.float32)


def extract_audio_features_for_frames(audio_filepath, source, reference_audio=True, cleaned_time=None, frame_duration_ms=10):
    """
    Extracts various audio features for each frame of the given audio file and returns them as a DataFrame.
    
    Parameters:
        audio_filepath (str): The path to the audio file.
        source (str): The source of the audio features, used to prefix the column names.
        reference_audio (bool): Whether the audio file is the reference audio file. 
                                (when processing multiple audio sources for the same session)
        cleaned_time (list): The cleaned time stamps corresponding to the extracted features from 
                             reference audio. Must be provided if reference_audio is False.
        frame_duration_ms (int): The size of the frame in milliseconds.
    
    Returns:
        features_df (pd.DataFrame): A DataFrame containing the extracted features for each frame.
        cleaned_time (list): The cleaned time stamps corresponding to the extracted features.
    """
    audio, fs = utils.load_audio(audio_filepath)
    cleaned_time, cleaned_frequencies = get_cleaned_time_and_frequencies(audio, fs, reference_audio, cleaned_time, frame_duration_ms)

    frames = utils.frame_generator(audio, fs, frame_duration_ms)
    cleaned_frames = get_cleaned_frames(frames, cleaned_time, frame_duration_ms)
    validate_cleaned_data(cleaned_frames, cleaned_frequencies)

    features_data = compute_audio_features(cleaned_frames, cleaned_frequencies, cleaned_time, source, fs)

    features_df = pd.DataFrame(features_data)
    return features_df, cleaned_time

def get_cleaned_time_and_frequencies(audio, fs, reference_audio, cleaned_time, frame_duration_ms):
    """
    Calculate cleaned time and frequencies based on the input audio.

    Args:
        audio (np.ndarray): The input audio signal.
        fs (int): The sampling rate of the audio.
        reference_audio (bool): Flag indicating whether a reference audio is provided.
        cleaned_time (np.ndarray or None): The cleaned time values. If None, it must be provided when reference_audio is False.
        frame_duration_ms (float): The duration of each frame in milliseconds.

    Returns:
        np.ndarray: The cleaned time values.
        np.ndarray: The cleaned frequencies.

    Raises:
        ValueError: If cleaned_time is None and reference_audio is False.

    """
    time, frequency, _, _ = pitch.estimate_pitch(audio, fs)
    sampled_times, sampled_frequencies = pitch.sample_pitches(time, frequency, frame_duration_ms)

    if reference_audio:
        cleaned_time, cleaned_frequencies = pitch.clean_sampled_data(sampled_times, sampled_frequencies)
    else:
        if cleaned_time is None:
            raise ValueError("cleaned_time must be provided when reference_audio is False")
        cleaned_frequencies = [sampled_frequencies[np.abs(sampled_times - t).argmin()] for t in cleaned_time]
    
    return np.array(cleaned_time), np.array(cleaned_frequencies)

def get_cleaned_frames(frames, cleaned_time, frame_duration_ms):
    frame_times = np.arange(0, len(frames)) * (frame_duration_ms / 1000)
    cleaned_frames = [frames[np.argmin(np.abs(frame_times - t))] for t in cleaned_time]
    return cleaned_frames

def validate_cleaned_data(cleaned_frames, cleaned_frequencies):
    if len(cleaned_frames) != len(cleaned_frequencies):
        raise ValueError("Number of cleaned frames and cleaned frequencies do not match")

def compute_audio_features(cleaned_frames, cleaned_frequencies, cleaned_time, source, fs):
    """
    Compute audio features for a given audio file.

    Args:
        cleaned_frames (list): List of cleaned audio frames.
        cleaned_frequencies (list): List of cleaned frequencies.
        cleaned_time (list): List of cleaned time values.
        source (str): Source identifier for the audio file.
        fs (int): Sampling rate of the audio file.

    Returns:
        dict: Dictionary containing computed audio features.

    Raises:
        ValueError: If the lengths of the feature lists are not equal.
    """

    spectrum_frames = spectrum(cleaned_frames)
    spec_spread, spec_skew, spec_kurt = spectral.distribution_shape(spectrum_frames)
    tristimulus1, tristimulus2, tristimulus3 = spectral.tristimulus(spectrum_frames, cleaned_frequencies)
    mfcc_values = spectral.mfcc_fb40(spectrum_frames, fs)

    features_data = {
        f"{source}_pitch": np.array(cleaned_frequencies, dtype=np.float32),
        f"{source}_note": [pitch.get_note_for_frequency(freq) for freq in cleaned_frequencies],
        f"{source}_rms_energy": rms_energy(cleaned_frames),
        f"{source}_spec_cent": spectral.spec_cent(spectrum_frames, fs),
        f"{source}_spec_spread": spec_spread,
        f"{source}_spec_skew": spec_skew,
        f"{source}_spec_kurt": spec_kurt,
        f"{source}_spec_slope": spectral.spec_slope(spectrum_frames),
        f"{source}_spec_decr": spectral.spec_decr(spectrum_frames, fs),
        f"{source}_spec_rolloff": spectral.spec_rolloff(spectrum_frames, fs),
        f"{source}_spec_flat": spectral.spec_flat(spectrum_frames),
        f"{source}_spec_crest": spectral.spec_crest(spectrum_frames),
        f"{source}_tristimulus1": tristimulus1,
        f"{source}_tristimulus2": tristimulus2,
        f"{source}_tristimulus3": tristimulus3
    }

    # Add MFCC coefficients
    for i in range(len(mfcc_values)):
        features_data[f"{source}_mfcc_{i+1}"] = mfcc_values[i]

    # Ensure all feature lists have the same length
    lengths = [len(v) for v in features_data.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All features must have the same number of rows")

    return features_data
