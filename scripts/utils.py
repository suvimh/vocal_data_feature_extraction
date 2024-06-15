'''
  General functions for output of audio features to CSV etc.
'''

import essentia.standard as es
import csv
import pandas as pd

LOOKUP_TABLE = "data/frequency_to_note.csv"

def load_audio(audio_path):
    """
    Load audio from the given file path using Essentia.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing the loaded audio data and the sample rate.
    """
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()
    return audio, loader.paramValue('sampleRate')


def frame_generator(audio, sample_rate):
    """
    Generate 10ms frames from an audio signal.

    Args:
        audio (np.ndarray): The audio signal.
        sample_rate (int): The sample rate of the audio signal.

    Returns:
        es.FrameGenerator: The frame generator object.
    """
    window_duration_ms = 10
    frame_size = int(sample_rate * (window_duration_ms / 1000.0))

    # Ensure frame size is even for FFT
    if frame_size % 2 != 0:
        frame_size += 1

    hop_size = frame_size  # For non-overlapping windows, hop_size is equal to frame_size

    return es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)


def create_and_save_lookup_table(csv_file):
    """
    Creates a lookup table of frequencies and corresponding note names and saves it to a CSV file.

    Args:
        csv_file (str): The path to the CSV file where the lookup table will be saved.

    """
    A4_frequency = 440.0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    notes = []

    for octave in range(0, 9):  # From C0 to B8
        for i, note in enumerate(note_names):
            frequency = A4_frequency * 2 ** ((i + (octave - 4) * 12 - 9) / 12)
            notes.append((frequency, f"{note}{octave}"))

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frequency', 'Note'])
        for freq, note in notes:
            writer.writerow([freq, note])


def get_note_for_frequency(frequency, file_path = LOOKUP_TABLE):
    """
    Returns the note name corresponding to a given frequency.

    Parameters:
        frequency (float): The frequency for which to find the note name.

    Returns:
        str: The note name corresponding to the given frequency.

    Raises:
        ValueError: If the 'Frequency' or 'Note' column is not found in the CSV file.
    """
    lookup_table = pd.read_csv(file_path)

    if frequency == 0:
        return "Rest"

    # Check for column names and adjust if necessary
    if 'Frequency' not in lookup_table.columns:
        raise ValueError("Column 'Frequency' not found in the CSV file.")
    if 'Note' not in lookup_table.columns:
        raise ValueError("Column 'Note' not found in the CSV file.")

    lookup_table['difference'] = (lookup_table['Frequency'] - frequency).abs()
    # Find the row with the minimum difference
    closest_row = lookup_table.loc[lookup_table['difference'].idxmin()]
    # Return the note name from the closest row
    return closest_row['Note']

