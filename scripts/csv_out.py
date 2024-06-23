'''
    FUNCTIONS TO HANDLE CSV OUTPUT OF DATA
'''

import csv
import os
import pandas as pd


def write_features_df_to_csv(csv_file, features_df):
    """
    Write the features DataFrame to a CSV file.

    Args:
        csv_file (str): Path to the CSV file to write.
        features_df (pd.DataFrame): DataFrame containing all features.

    Returns:
        None
    """
    features_df.to_csv(csv_file, index=False)





# def write_features_to_csv(output_file, metadata, features_for_audio_sources, features_for_video_sources, biosignal_data):
#     """
#     Write audio features along with metadata to a CSV file.

#     Parameters:
#         output_file (str): The file path of the output CSV file.
#         file_info (dict): Dictionary containing metadata about the file.
#         features_for_audio_sources (dict): dict of dicts containing audio features (per frame) for different audio sources.
#         features_for_video_sources (dict): dict of lists containing pose and face landmarks (per frame) for different video sources.
#         biosignal_data (dict): Dictionary containing biosignal data (per frame) for different channels.
#     """

#     header = construct_csv_header(features_for_audio_sources, features_for_video_sources)
#     file_exists = os.path.exists(output_file)

#     with open(output_file, mode='a', newline='') as file:
#         writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         if not file_exists:
#                 writer.writerow(header)

#         # Iterate over each frame and write the corresponding features to the CSV
#         for frame in range(len(biosignal_data['EMG_1'])):
#             row = write_metadata(metadata, frame)
#             row.extend(write_audio_features(features_for_audio_sources, frame))
#             row.extend(write_video_features(features_for_video_sources, frame))
#             row.extend(write_biosignal_data(biosignal_data, frame))
#             writer.writerow(row)


# def write_metadata(metadata, frame):
#     return [
#         metadata['participant_number'] if 'participant_number' in metadata else None,
#         metadata['sex'] if 'sex' in metadata else None,
#         metadata['age'] if 'age' in metadata else None,
#         metadata['experience_level'] if 'experience_level' in metadata else None,
#         metadata['phonation'] if 'phonation' in metadata else None,
#         metadata['recording_condition'] if 'recording_condition' in metadata else None,
#         metadata['phrase'] if 'phrase' in metadata else None,
#         metadata['clip_number'] if 'clip_number' in metadata else None,
#         frame
#     ]


# def write_audio_features(features_for_audio_sources, frame):
#     row = []
#     for _, features in features_for_audio_sources.items():
#         row.extend([
#             features['pitches'][frame] if 'pitches' in features else None,
#             features['notes'][frame] if 'notes' in features else None,
#             features['rms_energies'][frame] if 'rms_energies' in features else None,
#             # features['spectrums'][frame] if 'spectrums' in features else None,
#             features['tristimulus'][frame] if 'tristimulus' in features else None,
#             features['spec_cents'][frame] if 'spec_cents' in features else None,
#             features['spec_spread'][frame] if 'spec_spread' in features else None,
#             features['spec_skew'][frame] if 'spec_skew' in features else None,
#             features['spec_kurt'][frame] if 'spec_kurt' in features else None,
#             features['spec_slope'][frame] if 'spec_slope' in features else None,
#             features['spec_decr'][frame] if 'spec_decr' in features else None,
#             features['spec_rolloff'][frame] if 'spec_rolloff' in features else None,
#             features['spec_flat'][frame] if 'spec_flat' in features else None,
#             features['spec_crest'][frame] if 'spec_crest' in features else None,
#             features['mfccFB40'][frame] if 'mfccFB40' in features else None,
#         ])
#     return row


# def write_video_features(features_for_video_sources, frame):
#     row = []
#     for source, features in features_for_video_sources.items():
#         row.extend([
#             features[f'{source.lower()} pose'][frame] if f'{source.lower()} pose' in features else None,
#             features[f'{source.lower()} face'][frame] if f'{source.lower()} face' in features else None,
#         ])
#     return row


# def write_biosignal_data(biosignal_data, frame):
#     return [
#         biosignal_data['RESPIRATION_1'][frame] if 'RESPIRATION_1' in biosignal_data else None,
#         biosignal_data['EMG_1'][frame] if 'EMG_1' in biosignal_data else None,
#         biosignal_data['EEG_1'][frame] if 'EEG_1' in biosignal_data else None,
#         biosignal_data['EEG_2'][frame] if 'EEG_2' in biosignal_data else None
#     ]
    

# def construct_csv_header(features_for_audio_sources, features_for_video_sources):
#     """
#     Constructs the header for a CSV file that will store the extracted features.

#     Args:
#         features_for_audio_sources (list): A list of audio sources for which features are extracted.
#         features_for_video_sources (list): A list of video sources for which features are extracted.

#     Returns:
#         list: The constructed CSV header.
#     """
#     metadata_columns =      ["Participant", "Age", "Sex", "Experience level", "Phonation", "Recording Condition",
#                             "Phrase", "Clip Number", "Frame"]
#     audio_feature_columns = ["Pitch", "Note", "RMS Energy",  "Tristimulus", "Spec Centroid", "Spec Spread", 
#                              "Spec Skewness", "Spec Kurtosis", "Spec Slope", "Spec Decrease", "Spec Rolloff", 
#                              "Spec Flatness", "Spec Crest", "MFCC FB40"]
#     video_feature_columns = ["Pose Landmarks", "Face Landmarks"]
#     biosignal_columns =     ["PZT", "EMG", "EEG_1", "EEG_2"]

#     # create full CSV header
#     header = metadata_columns.copy()
#     for source in features_for_audio_sources:
#         for feature in audio_feature_columns:
#             header.append(f"{source} {feature}")
#     for source in features_for_video_sources:
#         for feature in video_feature_columns:
#             header.append(f"{source} {feature}")
#     for channel in biosignal_columns:
#         header.append(channel)
    
#     return header
 
