'''
  Functions used for pitch extraction, sampling and verification
'''

import numpy as np
import matplotlib.pyplot as plt
import crepe
import mir_eval
import numpy as np

def estimate_pitch(audio, fs, voicing_threshold=0.3, use_viterbi=False):
    """
    Estimate the fundamental frequency (pitch) of an audio file using the CREPE algorithm.

    Parameters:
        audio_path (str): The file path to the input audio file.
        voicing_threshold (float, optional): Confidence threshold for voicing. Default is 0.3.
        use_viterbi (bool, optional): Apply Viterbi decoding if True. Default is False.

    Returns:
        time (np.ndarray): Time stamps for each frame in seconds.
        frequency (np.ndarray): Estimated pitch for each frame in Hz.
        confidence (np.ndarray): Confidence of the pitch estimate for each frame.
        activation (np.ndarray): Activation matrix from the CREPE algorithm.
    """
    # Use CREPE to predict pitch
    time, frequency, confidence, activation = crepe.predict(audio, fs, viterbi=use_viterbi)

    # Postprocess the pitch values based on the voicing threshold
    frequency[confidence < voicing_threshold] = 0.0

    return time, frequency, confidence, activation


def plot_pitch(time, frequency, confidence, activation, sampled_time, sampled_frequency):
    """
    Plot pitch tracking information including the fundamental frequency (F0) over time,
    the confidence of the estimates, and an activation matrix representing the salience
    of pitches over time, with sampled points shown as dots.

    Parameters:
        time (array_like): An array of time stamps at which the frequency and confidence 
                           values are estimated.
        frequency (array_like): An array containing estimated fundamental frequency (F0) 
                                values in Hertz (Hz) for each time stamp.
        confidence (array_like): An array containing confidence values associated with 
                                 each F0 estimate.
        activation (array_like): A 2D array representing the activation of different pitch 
                                 bins over time.
        sampled_time (array_like): An array containing the sampled time stamps.
        sampled_frequency (array_like): An array containing the sampled pitch values in Hz.

    Notes:
        This function plots three subplots: The first subplot displays the F0 estimate over time,
        the second subplot shows the confidence of these estimates over time, and the third
        subplot shows the activation matrix with pitch bins in cents over time. A bug fix is
        applied for the pitch calculation as per a known issue in the CREPE repository.

    The function does not return any values but renders a matplotlib figure directly.

    References
    ----------
    .. [1] https://github.com/marl/crepe/issues/2
    """
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(12, 8), sharex=False)
    axes[0].plot(time, frequency, label='Estimated F0')
    axes[0].scatter(sampled_time, sampled_frequency, color='red', s=10, label='Sampled Points')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Estimated F0 (Hz)")
    axes[0].set_title("F0 Estimate Over Time")
    axes[0].legend()

    axes[1].plot(time, confidence)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Confidence")
    axes[1].set_title("Estimate Confidence Over Time")

    im = axes[2].imshow(activation.T, origin="lower", aspect="auto", interpolation='nearest', cmap='jet')
    cbar = fig.colorbar(im, ax=axes[2])
    cbar.set_label('Activation')

    axes[2].set_xticks(np.arange(len(activation))[::500])

    c1 = 32.7  # Hz, fix for a known issue in CREPE
    c1_cent = mir_eval.melody.hz2cents(np.array([c1]))[0]
    c = np.arange(0, 360) * 20 + c1_cent
    freq = 10 * 2 ** (c / 1200)

    axes[2].set_yticks(np.arange(len(freq))[::35])
    axes[2].set_yticklabels([int(f) for f in freq[::35]])
    axes[2].set_ylim([0, 300])
    axes[2].set_xticklabels((np.arange(len(activation))[::500] / 100).astype(int))
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Activation Matrix: 20 Cent Bins Over Time")

    plt.tight_layout()
    plt.show()


def sample_pitches(time, frequency, frame_duration_ms=10):
    """
    Sample the pitch values at regular intervals by taking the value at the midpoint of each window.

    Args:
        time (array-like): The time data.
        frequency (array-like): The frequency data.
        frame_duration_ms (int): The duration of each frame in milliseconds.

    Returns:
        sampled_time (ndarray): The sampled time data.
        sampled_frequency (ndarray): The sampled frequency data.
    """
    total_duration = time[-1]
    sample_interval = frame_duration_ms / 1000
    num_samples = int(total_duration / sample_interval) + 1

    sampled_time = np.linspace(0, total_duration, num_samples, dtype=np.float32)
    sampled_frequency = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        start_time = i * sample_interval
        end_time = start_time + sample_interval
        mid_time = (start_time + end_time) / 2

        # Find the closest index to the mid_time
        closest_index = np.argmin(np.abs(time - mid_time))
        sampled_frequency[i] = frequency[closest_index]

    return sampled_time, sampled_frequency


def clean_sampled_data(sampled_time, sampled_frequency):
    """
    Cleans the sampled time and frequency data by removing zero segments.

    Args:
        sampled_time (array-like): The sampled time data.
        sampled_frequency (array-like): The sampled frequency data.

    Returns:
        cleaned_time (ndarray): The cleaned time data with zero segments removed.
        cleaned_frequency (ndarray): The cleaned frequency data with zero segments removed.
    """
    sampled_time = np.array(sampled_time)
    sampled_frequency = np.array(sampled_frequency)

    non_zero_indices = np.where(sampled_frequency != 0)[0]

    if len(non_zero_indices) == 0:
        return np.array([]), np.array([])

    cleaned_time = []
    cleaned_frequency = []

    in_non_zero_segment = False
    segment_start = 0

    for i in range(len(non_zero_indices)):
        current_index = non_zero_indices[i]

        if not in_non_zero_segment:
            segment_start = current_index
            in_non_zero_segment = True

        if i == len(non_zero_indices) - 1 or non_zero_indices[i+1] != current_index + 1:
            segment_end = current_index
            segment_length = segment_end - segment_start + 1

            if segment_length > 2:
                cleaned_time.extend(sampled_time[segment_start:segment_end + 1])
                cleaned_frequency.extend(sampled_frequency[segment_start:segment_end + 1])

            in_non_zero_segment = False

    return np.array(cleaned_time, dtype=np.float32), np.array(cleaned_frequency, dtype=np.float32)


