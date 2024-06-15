'''
  Spectral feature extraction functions using Essentia.
'''

import numpy as np
import essentia.standard as es
import numpy as np
from scipy.stats import linregress

def spectral_peaks(spectrum_frames):
    """
    Calculate the spectral peaks for each frame in the given spectrum frames.

    Parameters:
    spectrum_frames (list): A list of spectrum frames.

    Returns:
    list: A list of spectral peaks for each frame in the spectrum frames.
    """
    spectral_peaks = es.SpectralPeaks()
    return [spectral_peaks(frame) for frame in spectrum_frames]


def tristimulus(spectrum_frames, sampled_frequencies):
    """
    Compute tristimulus values for each set of harmonic peaks.

    Args:
        spectrum_frames (list): List of spectrum frames.
        sampled_frequencies (list): List of sampled frequencies.

    Returns:
        list: List of tristimulus values for each set of harmonic peaks.
    """
    sp_peaks = spectral_peaks(spectrum_frames)
    harmonic_peaks = es.HarmonicPeaks()
    tristimulus = es.Tristimulus()

    hp_peaks = []

    for (sp_freq, sp_mag), pitch_f0 in zip(sp_peaks, sampled_frequencies):
        # make sure pitch precision is compatible with Essentia
        pitch_f0 = np.float32(pitch_f0)
        # Remove the DC component if the first frequency is zero
        if sp_freq[0] == 0:
            sp_freq = sp_freq[1:]
            sp_mag = sp_mag[1:]

        # Compute harmonic peaks and append to the list
        hp_peaks.append(harmonic_peaks(sp_freq, sp_mag, pitch_f0))

    # Compute and return tristimulus values for each set of harmonic peaks
    return [tristimulus(hp_freq, hp_mag) for (hp_freq, hp_mag) in hp_peaks]


def spec_cent(spectrum_frames, fs):
    """
    Calculate the spectral centroid for each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): List of spectrum frames.
    - fs (int): Sampling frequency.

    Returns:
    - list: List of spectral centroid values for each frame.
    """
    nyquist_frequency = fs / 2
    spectral_centroid = es.Centroid(range=nyquist_frequency)
    return [spectral_centroid(frame) for frame in spectrum_frames]


def distribution_shape(spectrum_frames):
    """
    Calculate the distribution shape features for each frame in the spectrum.

    Parameters:
    spectrum_frames (list): List of spectrum frames.

    Returns:
    tuple: A tuple containing three lists: spec_spread, spec_skew, spec_kurt.
           spec_spread: List of spread values for each frame.
           spec_skew: List of skewness values for each frame.
           spec_kurt: List of kurtosis values for each frame.
    """
    central_moments = es.CentralMoments(mode='sample')
    distribution_shape  = es.DistributionShape()

    spec_spread = []
    spec_skew = []
    spec_kurt = []

    for frame in spectrum_frames:
        c_moments = central_moments(frame)
        spread, skew, kurt = distribution_shape(c_moments)
        spec_spread.append(spread)
        spec_skew.append(skew)
        spec_kurt.append(kurt)

    return spec_spread, spec_skew, spec_kurt


def spec_slope(spectrum_frames):
    """
    Calculate the spectral slope for each spectrum in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectra frames.

    Returns:
    - spec_slopes (list): A list of spectral slopes corresponding to each spectrum frame.
    """
    spec_slopes = []

    for spectrum in spectrum_frames:
        # Compute log-power spectrum
        log_power_spectrum = np.log(np.abs(spectrum) ** 2 + 1e-6)  # Add a small value to avoid log(0)

        # Compute frequency axis
        freq_axis = np.arange(len(spectrum))

        # Perform linear regression
        slope, _, _, _, _ = linregress(freq_axis, log_power_spectrum)

        spec_slopes.append(slope)

    return spec_slopes


def spec_decr(spectrum_frames, fs):
    """
    Calculate the spectral decrease of each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectrum frames.
    - fs (int): The sampling frequency.

    Returns:
    - list: A list of spectral decrease values for each frame.
    """
    nyquist_frequency = fs / 2
    decrease = es.Decrease(range=nyquist_frequency)
    return [decrease(frame) for frame in spectrum_frames]


def spec_rolloff(spectrum_frames, fs):
    """
    Calculate the spectral rolloff for each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectrum frames.
    - fs (int): The sample rate of the audio.

    Returns:
    - list: A list of spectral rolloff values for each frame in the spectrum frames.
    """
    spectral_rolloff = es.RollOff(sampleRate=fs)
    return [spectral_rolloff(frame) for frame in spectrum_frames]


def spec_flat(spectrum_frames):
    """
    Calculate the spectral flatness of each frame in the given spectrum_frames.

    Parameters:
    spectrum_frames (list): A list of spectrum frames.

    Returns:
    list: A list of spectral flatness values for each frame.
    """
    flatness = es.Flatness()
    return [flatness(frame) for frame in spectrum_frames]


def spec_crest(spectrum_frames):
    """
    Calculate the spectral crest factor for each frame in the given spectrum frames.

    Parameters:
    spectrum_frames (list): A list of spectrum frames.

    Returns:
    list: A list of spectral crest factors for each frame.
    """
    crest = es.Crest()
    return [crest(frame) for frame in spectrum_frames]


def mfcc_fb40(spectrum_frames, fs):
    """
    Compute the Mel-frequency cepstral coefficients (MFCC) for a given set of spectrum frames.

    Args:
        spectrum_frames (array-like): A list or numpy array of spectrum frames.
        fs (int): The sample rate of the audio signal.

    Returns:
        list: A list of MFCC coefficients for each spectrum frame.
              MFCCs are with 40 bands and 13 coefficients (Essentia default)

    """
    if isinstance(spectrum_frames, list):
        spectrum_frames = np.array(spectrum_frames)
    input_size = spectrum_frames.shape[1]
    mfcc = es.MFCC(sampleRate=fs, type='magnitude', inputSize=input_size, numberBands=40, numberCoefficients=13)
    return [mfcc(frame)[1] for frame in spectrum_frames]  # [1] to get only MFCC coefficients



