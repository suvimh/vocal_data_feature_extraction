'''
    FUNCTIONS TO MANAGE BULK PROCESSING OF DATA
'''

import os
from scripts.metadata_extraction import extract_metadata
from scripts.audio_feature_extraction_for_file import extract_audio_features_for_frames
from scripts.video_landmark_extraction import get_mediapipe_pose_estimation_for_frames, get_dlib_face_estimation_for_frames
from scripts.biosignal_data_extraction import get_biosignal_data_for_frames
from scripts.csv_out import write_features_df_to_csv
from tqdm.auto import tqdm
import pandas as pd

def process_data_folder(data_directory, csv_out, processed_folders_file, num_folders_to_process, frame_duration_ms=10):
    """
    Process the data folder by extracting features and metadata from files and saving them to a CSV file.

    Args:
        data_directory (str): The path to the directory containing the audio files.
        csv_out (str): The path to the output CSV file.
        processed_folders_file (str): The file that logs processed folders.
        num_folders_to_process (int): The number of folders to process. Set to 0 to process all folders.
        frame_duration_ms (int): The size of the frame in milliseconds. Default is 10.
    """
    # Load the list of already processed folders
    processed_folders = set()
    if os.path.exists(processed_folders_file):
        with open(processed_folders_file, 'r') as log_file:
            processed_folders.update(line.strip() for line in log_file.readlines())

    # Count all folders to be processed
    if num_folders_to_process == 0:
        folders_to_process = []
        for root, dirs, _ in os.walk(data_directory):
            if not dirs:  # Check if there are no subdirectories -- means we are at the last level
                if root not in processed_folders:
                    folders_to_process.append(root)

        total_folders = len(folders_to_process)
    else:
        total_folders = num_folders_to_process

    # Walk through the directory tree and process folders
    with tqdm(total=total_folders, desc="Processing folders") as pbar:
        for root, dirs, _ in os.walk(data_directory):
            if not dirs:  # Check if there are no subdirectories -- means we are at the last level
                if root in processed_folders:
                    pbar.update(1)
                    continue  # Skip already processed folders

                process_folder(root, csv_out, processed_folders_file, processed_folders, frame_duration_ms)
                pbar.update(1)

                if num_folders_to_process != 0 and len(processed_folders) % num_folders_to_process == 0:
                    print(f"Processed {num_folders_to_process} folders. Stopping.")
                    break  # Stop after processing `num_folders_for_processing` folders

            if num_folders_to_process == 0:
                continue 

    pbar.close()


def process_folder(root, csv_out, processed_folders_file, processed_folders, frame_duration_ms):
    """
    Process a single folder by extracting features and saving them to a CSV file.

    Args:
        root (str): The path to the folder to process.
        csv_out (str): The path to the output CSV file.
        processed_folders_file (str): The file that logs processed folders.
        processed_folders (set): A set of already processed folders.
        frame_duration_ms (int): The size of the frame in milliseconds.
    """
    try:
        # Sort files into categories -- need to process mic file as reference file so that all files
        # have the same number of frames based on cleaned_time from mic audio -- removing any silence
        mic_wav_files = [file for file in os.listdir(root) if file.endswith('.wav') and 'mic' in file]
        other_wav_files = [file for file in os.listdir(root) if file.endswith('.wav') and 'mic' not in file]
        mp4_files = [file for file in os.listdir(root) if file.endswith('.mp4')]
        json_files = [file for file in os.listdir(root) if file.endswith('.json')]

        if len(mic_wav_files) != 1:
            raise ValueError(f"Expected 1 mic audio file in folder, found {len(mic_wav_files)}")
        if len(other_wav_files) != 2:
            raise ValueError(f"Expected 2 other audio files in folder, found {len(other_wav_files)}")
        if len(mp4_files) != 2:
            raise ValueError(f"Expected 2 video files in folder, found {len(mp4_files)}")
        if len(json_files) != 1:
            raise ValueError(f"Expected 1 json files in folder, found {len(json_files)}")

        # combine everything into one dataframe 
        features_df, cleaned_time, mic_feature_length = process_mic_audio(root, mic_wav_files[0], frame_duration_ms)
        features_df = process_other_audio(features_df, root, other_wav_files, cleaned_time, mic_feature_length, frame_duration_ms)
        features_df = process_videos(features_df, root, mp4_files, cleaned_time, mic_feature_length)
        features_df = process_biosignal_data(features_df, root, json_files[0], cleaned_time, mic_feature_length, frame_duration_ms)
        
        #output to CSV
        write_features_df_to_csv(csv_out, features_df)

        # Log the folder as processed
        processed_folders.add(root)
        with open(processed_folders_file, 'a') as log_file:
            log_file.write(root + '\n')

    except Exception as e:
        raise ValueError(f"Error ({e}) processing data in folder: {root}")


def process_biosignal_data(features_df, root, json_file, cleaned_time, mic_feature_length, frame_duration_ms):
    """
    Process biosignal data for each file in the given directory.

    Args:
        root (str): The root directory path.
        json_files (list): A list of JSON file names.
        cleaned_time (float): The cleaned time value.
        mic_feature_length (int): The length of the audio features.
        frame_duration_ms (int): The size of the frame in milliseconds.

    Returns:
        dict: A dictionary containing the processed biosignal data.

    Raises:
        ValueError: If the length of biosignal data and audio features is not the same.
    """
    # already checked that just one file
    file_path = os.path.join(root, json_file)
    biosignal_data_df = get_biosignal_data_for_frames(cleaned_time, file_path, frame_duration_ms)
    if len(biosignal_data_df) != mic_feature_length:
        raise ValueError("Biosignal data frame number must match audio feature frame length")
    features_df = pd.concat([features_df, biosignal_data_df], axis=1)
    
    return features_df


def process_videos(features_df, root, mp4_files, cleaned_time, mic_feature_length):
    """
    Process videos and extract pose and face landmarks for each frame.

    Args:
        features_df (pd.DataFrame): The metadata and features.
        root (str): The root directory path.
        mp4_files (list): A list of mp4 file names.
        cleaned_time (list): A list of cleaned time values.
        mic_feature_length (int): The expected length of the audio feature frames.

    Returns:
        tuple: A tuple containing the pose and face landmarks for phone and computer videos.
            - phone_pose_landmarks (pd.DataFrame): The pose landmarks for the phone video.

    Raises:
        ValueError: If the frame number of pose or face landmarks does not match the audio feature frame length.
        ValueError: If the video source of an mp4 file cannot be identified.
    """
    for file in mp4_files:
        file_path = os.path.join(root, file)
        if 'phone' in file or 'computer' in file:
            for source in ["phone", "computer"]:
                if source in file:
                    pose_landmarks_df = get_mediapipe_pose_estimation_for_frames(cleaned_time, file_path, source=source)
                    if len(pose_landmarks_df) != mic_feature_length:
                        raise ValueError(f"{source} pose frame number must match audio feature frame length")
                    features_df = pd.concat([features_df, pose_landmarks_df], axis=1)

                    face_landmarks_df = get_dlib_face_estimation_for_frames(cleaned_time, file_path, source=source)
                    if len(face_landmarks_df) != mic_feature_length:
                        raise ValueError(f"{source} face frame number must match audio feature frame length")
                    features_df = pd.concat([features_df, face_landmarks_df], axis=1)
                    break
        else:
            raise ValueError(f"Cannot identify video source of mp4 file: {file}.")
    return features_df


def process_other_audio(features_df, root, other_wav_files, cleaned_time, mic_feature_length, frame_duration_ms):
    """
    Process other audio files.

    Args:
        features_df (pd.DataFrame): The metadata and features.
        root (str): The root directory path.
        other_wav_files (list): List of other WAV files.
        cleaned_time (float): Cleaned time value.
        mic_feature_length (int): Length of the microphone feature.
        frame_duration_ms (int): The size of the frame in milliseconds.

    Returns:
        tuple: A tuple containing the phone audio features and computer audio features.

    Raises:
        ValueError: If the audio source of a WAV file cannot be identified, or if the audio feature lists have different lengths.
    """
    for file in other_wav_files:
        file_path = os.path.join(root, file)
        if 'phone' in file or 'computer' in file:
            for source in ["phone", "computer"]:
                if source in file:
                    audio_features_df, _ = extract_audio_features_for_frames(file_path, source=source, reference_audio=False, cleaned_time=cleaned_time, frame_duration_ms=frame_duration_ms)
                    check_audio_feature_length(audio_features_df, source)
                    if len(audio_features_df) != mic_feature_length:
                        raise ValueError(f"{source} audio feature frame number must match audio feature frame length")
                    features_df = pd.concat([features_df, audio_features_df], axis=1)
                    break  # Found the correct source, no need to check further
        else:
            raise ValueError(f"Cannot identify audio source of WAV file: {file}.")
    
    return features_df


def process_mic_audio(root, mic_wav_file, frame_duration_ms):
    """
    Process microphone audio files.

    Args:
        root (str): The root directory path.
        mic_wav_files (list): List of microphone audio file names.
        frame_duration_ms (int): The size of the frame in milliseconds.

    Returns:
        tuple: A tuple containing the following elements:
            - metadata_audio_df (pd.DataFrame): The metadata and audio features.
            - cleaned_time (float): Time taken for cleaning the audio.
            - mic_feature_length (int): Length of the microphone audio feature lists.
    """

    file_path = os.path.join(root, mic_wav_file) # only one file -- already checked in prior code
    metadata_df = extract_metadata(file_path)
    mic_audio_features_df, cleaned_time = extract_audio_features_for_frames(file_path, source='mic', reference_audio=True, frame_duration_ms=frame_duration_ms)

    mic_feature_length = check_audio_feature_length(mic_audio_features_df, 'mic')
    # reindex metadata_df to match mic_audio_features_df
    metadata_df = metadata_df.reindex(mic_audio_features_df.index)
    metadata_df_filled = metadata_df.ffill() # Fill forward to copy metadata across all rows
    # Concatenate metadata_df_filled with mic_audio_features_df
    metadata_audio_df = pd.concat([metadata_df_filled, mic_audio_features_df], axis=1)
        
    return metadata_audio_df, cleaned_time, mic_feature_length


def check_audio_feature_length(audio_features_df, source):
    """
    Check the length of audio features extracted from different audio sources.

    Returns:
        int: The length of the audio features (if mic as source).
    """
    lengths = audio_features_df.apply(lambda col: col.dropna().size)

    # Verify all columns have the same length
    feature_length = lengths.iloc[0]
    if not lengths.eq(feature_length).all():
        raise ValueError(f"All {source} audio feature lists must have the same length")
    
    if source == 'mic':
        return feature_length
    else:
        return 
