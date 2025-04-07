'''
    Extract metadata for a file from the folder structure it's in
    All functions work based on the Vocal Data folder structure
'''

import os
import csv 
import pandas as pd


METADATA = "data/metadata.csv"

class PathLengthError(Exception):
    pass


class AudioSourceError(Exception):
    pass


def get_participant_number(file_path):
    """
    Extracts the name from a given file path.

    Args:
        file_path (str): The file path from which to extract the name.

    Returns:
        str: The extracted name.

    Raises:
        PathLengthError: If the path is too short to extract the desired folder.
    """
    path_parts = file_path.split(os.sep)

    if 'inexperienced' in file_path:
      if len(path_parts) >= 5:
        number = path_parts[-5].strip('P')
        return number
      else:
        raise PathLengthError("Path is too short to extract the participant number.")
    else:
      # Extract the sixth folder from the end of the path
      if len(path_parts) >= 6:
          number = path_parts[-6].strip('P')
          return number
      else:
          raise PathLengthError("Path is too short to extract the participant number.")

def get_lowest_note(number, metadata=METADATA):
    """
    Retrieve the lowest note sung by the participant.

    Args:
        number (str): participant number.

    Returns:
        str: lowest note sung by the participant.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Participant number'].strip().lower() == number.strip().lower():
                return row['Lowest']
    return None


def get_experience_level(number, metadata=METADATA):
    """
    Retrieves the experience level associated with a given name from the metadata file.

    Parameters:
    - number (str): The number to search for in the metadata file.
    - metadata (str): The path to the metadata file (default: METADATA).

    Returns:
    - str or None: The experience level associated with the name, or None if not found.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Participant number'].strip().lower() == number.strip().lower():
                return row['Experience level']
    return None


def get_phonation(file_path):
    """
    Extracts the phonation type from the given file path.

    Args:
        file_path (str): The path of the file.

    Returns:
        str: The phonation type extracted from the file path.

    Raises:
        PathLengthError: If the path is too short to extract the desired folder.
    """
    path_parts = file_path.split(os.sep)
    if 'inexperienced' in file_path:
      return 'undefined'
    else:
      # Extract the fifth folder from the end of the path
      if len(path_parts) >= 5:
          phonation = path_parts[-5]
          return phonation
      else:
          raise PathLengthError("Path is too short to extract the desired folder.")


def get_recording_condition(file_path):
    """
    Extracts the recording condition from the given file path.

    Args:
        file_path (str): The path of the file.

    Returns:
        str: The recording condition extracted from the file path.

    Raises:
        PathLengthError: If the path is too short to extract the desired folder.
    """
    path_parts = file_path.split(os.sep)
    # Extract the fourth folder from the end of the path
    if len(path_parts) >= 4:
        recording_condition = path_parts[-4]
        return recording_condition
    else:
        raise PathLengthError("Path is too short to extract the desired folder.")


def get_highest_note(number, metadata=METADATA):
    """
    Retrieve the highest note sung by the participant.

    Args:
        number (str): participant number.

    Returns:
        str: highest note sung by the participant.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Participant number'].strip().lower() == number.strip().lower():
                return row['Highest']
    return None


def get_phrase(file_path):
    """
    Extracts the third folder from the end of the given file path.

    Args:
        file_path (str): The file path from which to extract the folder.

    Returns:
        str: The third folder from the end of the file path.

    Raises:
        PathLengthError: If the file path is too short to extract the desired folder.
    """
    path_parts = file_path.split(os.sep)
    if len(path_parts) >= 3:
        recording_condition = path_parts[-3]
        return recording_condition
    else:
        raise PathLengthError("Path is too short to extract the desired folder.")


def get_clip_number(file_path):
    """
    Extracts the clip number from the given file path.

    Args:
        file_path (str): The path of the file.

    Returns:
        str: The clip number extracted from the file path.

    Raises:
        PathLengthError: If the path is too short to extract the desired folder.
    """
    path_parts = file_path.split(os.sep)
    # Extract the second folder from the end of the path
    if len(path_parts) >= 2:
        recording_condition = path_parts[-2]
        return recording_condition
    else:
        raise PathLengthError("Path is too short to extract the desired folder.")


def extract_metadata(file_path):
    """
    Extracts metadata from the given file path.

    Parameters:
    file_path (str): The path of the file from which to extract metadata.

    Returns:
    dict: A dictionary containing the extracted metadata. The dictionary has the following keys:
        - 'participant_number': The participant number.
        - 'highest_note': The highest note sung by the participant in recordings. 
        - 'lowest_note': The lowest note sung by the participant in recordings.
        - 'experience_level': The singing experience level of the participant.
        - 'phonation': The phonation mode the participant was instructed to sing in.
        - 'recording_condition': The recording condition of the session.
        - 'audio_source': The audio source of the recording.
        - 'phrase': The phrase sung.
        - 'clip_number': The clip number (repetition of phrase).
    """
    number = get_participant_number(file_path)

    file_info = {
        'participant_number' : number,
        'highest_note' : get_highest_note(number),
        'lowest_note' : get_lowest_note(number),
        'experience_level' : get_experience_level(number),
        'phonation' : get_phonation(file_path),
        'recording_condition' : get_recording_condition(file_path),
        'phrase' : get_phrase(file_path),
        'clip_number' : get_clip_number(file_path)
    }

    file_info_df = pd.DataFrame([file_info])

    return file_info_df