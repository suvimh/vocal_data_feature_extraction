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


def get_name(file_path):
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

    # Extract the sixth folder from the end of the path
    if len(path_parts) >= 6:
        name = path_parts[-6]
        name = name.replace('-m', '')
        return name
    else:
        raise PathLengthError("Path is too short to extract the desired folder.")


def get_participant_number(name, metadata=METADATA):
    """
    Retrieves the participant number for a given name from the metadata file.

    Args:
        name (str): The name of the participant.
        metadata (str): The path to the metadata file (default: METADATA).

    Returns:
        str or None: The participant number if found, None otherwise.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'].strip().lower() == name.strip().lower():
                return row['Participant number']
    return None


def get_age(name, metadata=METADATA):
    """
    Retrieves the age of a person based on their name from the given metadata file.

    Args:
        name (str): The name of the person.
        metadata (str): The path to the metadata file (default: METADATA).

    Returns:
        str or None: The age of the person if found, None otherwise.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'].strip().lower() == name.strip().lower():
                return row['Age']
    return None


def get_experience_level(name, metadata=METADATA):
    """
    Retrieves the experience level associated with a given name from the metadata file.

    Parameters:
    - name (str): The name to search for in the metadata file.
    - metadata (str): The path to the metadata file (default: METADATA).

    Returns:
    - str or None: The experience level associated with the name, or None if not found.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'].strip().lower() == name.strip().lower():
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


def get_sex(name, metadata=METADATA):
    """
    Determines the participant sex based on the file path.

    Args:
        file_path (str): The path of the audio file.

    Returns:
        str: Sex of the participant.
    """
    with open(metadata, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Name'].strip().lower() == name.strip().lower():
                return row['Sex']
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
        - 'participant_number': The participant number extracted from the file name.
        - 'age': The age extracted from the file name.
        - 'experience_level': The experience level extracted from the file name.
        - 'phonation': The phonation extracted from the file.
        - 'recording_condition': The recording condition extracted from the file.
        - 'audio_source': The audio source extracted from the file.
        - 'phrase': The phrase extracted from the file.
        - 'clip_number': The clip number extracted from the file.
    """
    name = get_name(file_path)

    file_info = {
        'participant_number' : get_participant_number(name),
        'sex' : get_sex(name),
        'age' : get_age(name),
        'experience_level' : get_experience_level(name),
        'phonation' : get_phonation(file_path),
        'recording_condition' : get_recording_condition(file_path),
        'phrase' : get_phrase(file_path),
        'clip_number' : get_clip_number(file_path)
    }

    file_info_df = pd.DataFrame([file_info])

    return file_info_df