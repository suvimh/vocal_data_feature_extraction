"""
  Audio feature extraction of all frames of one file using Essentia.
"""

import numpy as np
import essentia.standard as es
import numpy as np

from scripts.spectral_feature_extraction import *
from scripts.pitch_extraction import *
from scripts.utils import *


def spectrum(frames):
    """
    Compute the spectrum of each frame in the given list of frames.

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
        frames (list): A list of audio frames.

    Returns:
        list: A list of RMS energy values for each frame.
    """
    rms = es.RMS()
    return [rms(frame) for frame in frames]


def extract_audio_features_for_frames(audio_filepath, reference_audio=True, cleaned_time=None, frame_duration_ms=10):
    """
    Extracts various audio features for each frame of the given audio file.

    Parameters:
        audio_filepath (str): The path to the audio file.
        reference_audio (bool): Whether the audio file is the reference audio file. 
                                (when processing multiple audio sources for same session)
        cleaned_time (list): The cleaned time stamps corresponding to the extracted features from 
                             reference audio. Must be provided if reference_audio is False.
        frame_duration_ms (int): The size of the frame in milliseconds.

    Returns:
        features_for_frames (dict): A dictionary containing the extracted features for each frame.
          - 'pitches' (list): The cleaned frequencies of the pitches.
          - 'notes' (list): The musical notes corresponding to the cleaned frequencies.
          - 'rms_energies' (list): The root mean square (RMS) energies of the frames.
          - 'spectrums' (list): The spectra of the frames.
          - 'tristimulus' (list): The tristimulus values calculated from the spectra and cleaned frequencies.
          - 'spec_cents' (list): The spectral centroids of the frames.
          - 'spec_spread' (float): The spectral spread of the frames.
          - 'spec_skew' (float): The spectral skewness of the frames.
          - 'spec_kurt' (float): The spectral kurtosis of the frames.
          - 'spec_slope' (float): The spectral slope of the frames.
          - 'spec_decr' (float): The spectral decrease of the frames.
          - 'spec_rolloff' (float): The spectral rolloff of the frames.
          - 'spec_flat' (float): The spectral flatness of the frames.
          - 'spec_crest' (float): The spectral crest of the frames.
          - 'mfccFB40' (list): The 40-dimensional Mel-frequency cepstral coefficients
                              (MFCCs) of the frames.
        cleaned_time (list): The cleaned time stamps corresponding to the extracted features.
    """
    audio, fs = load_audio(audio_filepath)
    # pitch extraction using crepe on whole file, followed by sampling and cleaning of the output of crepe
    time, frequency, _, _ = estimate_pitch(audio, fs)
    sampled_times, sampled_frequencies = sample_pitches(time, frequency, frame_duration_ms)
   
    if reference_audio:
        cleaned_time, cleaned_frequencies = clean_sampled_data(
            sampled_times, sampled_frequencies
        )
    else:  # clean the sampled data to remove silence based on reference audio
        if cleaned_time is None:
            raise ValueError("cleaned_time must be provided when reference_audio is False")
        cleaned_frequencies = []
        for t in cleaned_time:
            index = np.abs(sampled_times - t).argmin()
            cleaned_frequencies.append(sampled_frequencies[index])
        cleaned_frequencies = np.array(cleaned_frequencies)

    # splitting full audio into frames 
    frames = list(frame_generator(audio, fs, frame_duration_ms))
    frame_times = np.arange(0, len(frames)) * (frame_duration_ms / 1000) 

     # Select frames based on cleaned times from crepe algorithm (to remove silence)
    cleaned_frames = []
    for t in cleaned_time:
        frame_index = np.argmin(np.abs(frame_times - t))
        cleaned_frames.append(frames[frame_index])
    
    if len(cleaned_frames) != len(cleaned_frequencies):
        raise ValueError("Number of cleaned frames and cleaned frequencies do not match")

    rms = es.RMS()
    spectrum = es.Spectrum()
    rms_frames = [rms(frame) for frame in cleaned_frames]
    spectrum_frames = [spectrum(frame) for frame in cleaned_frames]

    spec_spread, spec_skew, spec_kurt = distribution_shape(spectrum_frames)

    features_for_frames = {
        "pitches": cleaned_frequencies,
        "notes": [get_note_for_frequency(freq) for freq in cleaned_frequencies],
        "rms_energies": rms_frames,
        "spectrums": spectrum_frames,
        "tristimulus": tristimulus(spectrum_frames, cleaned_frequencies),
        "spec_cents": spec_cent(spectrum_frames, fs),
        "spec_spread": spec_spread,
        "spec_skew": spec_skew,
        "spec_kurt": spec_kurt,
        "spec_slope": spec_slope(spectrum_frames),
        "spec_decr": spec_decr(spectrum_frames, fs),
        "spec_rolloff": spec_rolloff(spectrum_frames, fs),
        "spec_flat": spec_flat(spectrum_frames),
        "spec_crest": spec_crest(spectrum_frames),
        "mfccFB40": mfcc_fb40(spectrum_frames, fs),
    }

    return features_for_frames, cleaned_time
