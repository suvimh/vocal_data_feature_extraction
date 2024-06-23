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
          Not changing the return type to np.ndarray to avoid issues with Essentia data types.
    """
    spectral_peaks = es.SpectralPeaks()
    return [spectral_peaks(frame) for frame in spectrum_frames]


def tristimulus(spectrum_frames, cleaned_frequencies):
    """
    Compute tristimulus values for each set of harmonic peaks.

    Args:
        spectrum_frames (list): List of spectrum frames.
        cleaned_frequencies (list): List of cleaned frequencies.

    Returns:
        np.ndarray: Array of tristimulus1 values.
        np.ndarray: Array of tristimulus2 values.
        np.ndarray: Array of tristimulus3 values.
    """
    sp_peaks = spectral_peaks(spectrum_frames)
    harmonic_peaks = es.HarmonicPeaks()
    tristimulus = es.Tristimulus()

    for (sp_freq, sp_mag), pitch_f0 in zip(sp_peaks, cleaned_frequencies):
        # Remove the DC component if the first frequency is zero
        if sp_freq[0] == 0:
            sp_freq = sp_freq[1:]
            sp_mag = sp_mag[1:]

        # Compute harmonic peaks
        hp_freq, hp_mag = harmonic_peaks(sp_freq, sp_mag, pitch_f0)

        # Compute tristimulus values
        ts_values = tristimulus(hp_freq, hp_mag)

        if len(ts_values) != 3:
            raise ValueError("Tristimulus function did not return three values as expected")

        # Append to the list of tristimulus values
        if "tristimulus1" not in locals():
            tristimulus1 = np.array([ts_values[0]])
            tristimulus2 = np.array([ts_values[1]])
            tristimulus3 = np.array([ts_values[2]])
        else:
            tristimulus1 = np.concatenate((tristimulus1, [ts_values[0]]))
            tristimulus2 = np.concatenate((tristimulus2, [ts_values[1]]))
            tristimulus3 = np.concatenate((tristimulus3, [ts_values[2]]))
    
    return tristimulus1, tristimulus2, tristimulus3


def spec_cent(spectrum_frames, fs):
    """
    Calculate the spectral centroid for each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): List of spectrum frames.
    - fs (int): Sampling frequency.

    Returns:
    - np.ndarray: List of spectral centroid values for each frame.
    """
    nyquist_frequency = fs / 2
    spectral_centroid = es.Centroid(range=nyquist_frequency)
    return np.array([spectral_centroid(frame) for frame in spectrum_frames], dtype=np.float32)


def distribution_shape(spectrum_frames):
    """
    Calculate the distribution shape features for each frame in the spectrum.

    Parameters:
    spectrum_frames (list): List of spectrum frames.

    Returns:
    tuple: A tuple containing three np.ndarrays: spec_spread, spec_skew, spec_kurt.
           spec_spread: Array of spread values for each frame.
           spec_skew: Array of skewness values for each frame.
           spec_kurt: Array of kurtosis values for each frame.
    """
    central_moments = es.CentralMoments(mode='sample')
    distribution_shape = es.DistributionShape()

    spec_spread = np.zeros(len(spectrum_frames), dtype=np.float32)
    spec_skew = np.zeros(len(spectrum_frames), dtype=np.float32)
    spec_kurt = np.zeros(len(spectrum_frames), dtype=np.float32)

    for i, frame in enumerate(spectrum_frames):
        c_moments = central_moments(frame)
        spread, skew, kurt = distribution_shape(c_moments)
        spec_spread[i] = spread
        spec_skew[i] = skew
        spec_kurt[i] = kurt

    return spec_spread, spec_skew, spec_kurt


def spec_slope(spectrum_frames):
    """
    Calculate the spectral slope for each spectrum in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectra frames.

    Returns:
    - np.ndarray: Array of spectral slopes corresponding to each spectrum frame.
    """
    spec_slopes = np.zeros(len(spectrum_frames), dtype=np.float32)

    for i, spectrum in enumerate(spectrum_frames):
        freq_axis = np.arange(len(spectrum))
        slope, _, _, _, _ = linregress(freq_axis, spectrum)
        spec_slopes[i] = slope

    return spec_slopes

def spec_decr(spectrum_frames, fs):
    """
    Calculate the spectral decrease of each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectrum frames.
    - fs (int): The sampling frequency.

    Returns:
    - np.ndarray: A list of spectral decrease values for each frame.
    """
    nyquist_frequency = fs / 2
    decrease = es.Decrease(range=nyquist_frequency)
    return np.array([decrease(frame) for frame in spectrum_frames], dtype=np.float32)


def spec_rolloff(spectrum_frames, fs):
    """
    Calculate the spectral rolloff for each frame in the given spectrum frames.

    Parameters:
    - spectrum_frames (list): A list of spectrum frames.
    - fs (int): The sample rate of the audio.

    Returns:
    - np.ndarray: A list of spectral rolloff values for each frame in the spectrum frames.
    """
    spectral_rolloff = es.RollOff(sampleRate=fs)
    return np.array([spectral_rolloff(frame) for frame in spectrum_frames], dtype=np.float32)


def spec_flat(spectrum_frames):
    """
    Calculate the spectral flatness of each frame in the given spectrum_frames.

    Parameters:
    spectrum_frames (list): A list of spectrum frames.

    Returns:
    np.ndarray: A list of spectral flatness values for each frame.
    """
    flatness = es.Flatness()
    return np.array([flatness(frame) for frame in spectrum_frames], dtype=np.float32)


def spec_crest(spectrum_frames):
    """
    Calculate the spectral crest factor for each frame in the given spectrum frames.

    Parameters:
    spectrum_frames (list): A list of spectrum frames.

    Returns:
    np.ndarray: A list of spectral crest factors for each frame.
    """
    crest = es.Crest()
    return np.array([crest(frame) for frame in spectrum_frames], dtype=np.float32)


def mfcc_fb40(spectrum_frames, fs, num_coeffs=13, num_bands=40):
    """
    Compute the Mel-frequency cepstral coefficients (MFCC) for a given set of spectrum frames.

    Args:
        spectrum_frames (array-like): A list or numpy array of spectrum frames.
        fs (int): The sample rate of the audio signal.
        num_coeffs (int): The number of MFCC coefficients to compute.
        num_bands (int): The number of Mel bands to use.
        
    Returns:
        tuple: A tuple of numpy arrays, each containing MFCC coefficients for all frames.
               Each numpy array corresponds to one MFCC coefficient.
    """
    if isinstance(spectrum_frames, list):
        spectrum_frames = np.array(spectrum_frames)

    input_size = spectrum_frames.shape[1]
    mfcc = es.MFCC(sampleRate=fs, type='magnitude', inputSize=input_size, numberBands=num_bands, numberCoefficients=num_coeffs)

    mfcc_coeffs = [np.zeros(len(spectrum_frames), dtype=np.float32) for _ in range(num_coeffs)]

    # Compute MFCC coefficients for each spectrum frame
    for i, frame in enumerate(spectrum_frames):
        mfcc_frame = mfcc(frame)[1]  # Compute MFCC coefficients for the current frame
        for j in range(num_coeffs):
            mfcc_coeffs[j][i] = mfcc_frame[j]  # Store the j-th coefficient for all frames

    return tuple(mfcc_coeffs)
