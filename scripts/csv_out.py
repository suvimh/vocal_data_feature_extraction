'''
    FUNCTIONS TO HANDLE CSV OUTPUT OF DATA
'''

import pandas as pd
import os

def write_features_df_to_csv(csv_file, features_df, predefined_columns=None):
    """
    Append the features DataFrame to a CSV file, creating the file if it doesn't exist.
    If the file is created for the first time, the header will be written based on the predefined columns.

    Args:
        csv_file (str): Path to the CSV file to write.
        features_df (pd.DataFrame): DataFrame containing all features.
        predefined_columns (list): List of predefined column names to use in the header.

    Returns:
        None
    """
    if predefined_columns is None:
        predefined_columns = [
      'participant_number','sex','age','experience_level',
      'phonation','recording_condition','phrase','clip_number','mic_pitch',
      'mic_note','mic_rms_energy','mic_spec_cent','mic_spec_spread',
      'mic_spec_skew','mic_spec_kurt','mic_spec_slope','mic_spec_decr','mic_spec_rolloff',
      'mic_spec_flat','mic_spec_crest','mic_tristimulus1','mic_tristimulus2',
      'mic_tristimulus3','mic_mfcc_1','mic_mfcc_2','mic_mfcc_3','mic_mfcc_4',
      'mic_mfcc_5','mic_mfcc_6','mic_mfcc_7','mic_mfcc_8','mic_mfcc_9','mic_mfcc_10',
      'mic_mfcc_11','mic_mfcc_12','mic_mfcc_13','phone_pitch','phone_note','phone_rms_energy',
      'phone_spec_cent','phone_spec_spread','phone_spec_skew','phone_spec_kurt',
      'phone_spec_slope','phone_spec_decr','phone_spec_rolloff','phone_spec_flat',
      'phone_spec_crest','phone_tristimulus1','phone_tristimulus2','phone_tristimulus3',
      'phone_mfcc_1','phone_mfcc_2','phone_mfcc_3','phone_mfcc_4','phone_mfcc_5','phone_mfcc_6',
      'phone_mfcc_7','phone_mfcc_8','phone_mfcc_9','phone_mfcc_10','phone_mfcc_11','phone_mfcc_12',
      'phone_mfcc_13','computer_pitch','computer_note','computer_rms_energy','computer_spec_cent',
      'computer_spec_spread','computer_spec_skew','computer_spec_kurt','computer_spec_slope',
      'computer_spec_decr','computer_spec_rolloff','computer_spec_flat','computer_spec_crest',
      'computer_tristimulus1','computer_tristimulus2','computer_tristimulus3','computer_mfcc_1',
      'computer_mfcc_2','computer_mfcc_3','computer_mfcc_4','computer_mfcc_5','computer_mfcc_6',
      'computer_mfcc_7','computer_mfcc_8','computer_mfcc_9','computer_mfcc_10','computer_mfcc_11',
      'computer_mfcc_12','computer_mfcc_13','computer_pose_landmark_1_x','computer_pose_landmark_1_y',
      'computer_pose_landmark_1_z','computer_pose_landmark_2_x','computer_pose_landmark_2_y',
      'computer_pose_landmark_2_z','computer_pose_landmark_3_x','computer_pose_landmark_3_y',
      'computer_pose_landmark_3_z','computer_pose_landmark_4_x','computer_pose_landmark_4_y',
      'computer_pose_landmark_4_z','computer_pose_landmark_5_x','computer_pose_landmark_5_y',
      'computer_pose_landmark_5_z','computer_pose_landmark_6_x','computer_pose_landmark_6_y',
      'computer_pose_landmark_6_z','computer_pose_landmark_7_x','computer_pose_landmark_7_y',
      'computer_pose_landmark_7_z','computer_pose_landmark_8_x','computer_pose_landmark_8_y',
      'computer_pose_landmark_8_z','computer_pose_landmark_9_x','computer_pose_landmark_9_y',
      'computer_pose_landmark_9_z','computer_pose_landmark_10_x','computer_pose_landmark_10_y',
      'computer_pose_landmark_10_z','computer_pose_landmark_11_x','computer_pose_landmark_11_y',
      'computer_pose_landmark_11_z','computer_pose_landmark_12_x','computer_pose_landmark_12_y',
      'computer_pose_landmark_12_z','computer_pose_landmark_13_x','computer_pose_landmark_13_y',
      'computer_pose_landmark_13_z','computer_pose_landmark_14_x','computer_pose_landmark_14_y',
      'computer_pose_landmark_14_z','computer_pose_landmark_15_x','computer_pose_landmark_15_y',
      'computer_pose_landmark_15_z','computer_pose_landmark_16_x','computer_pose_landmark_16_y',
      'computer_pose_landmark_16_z','computer_pose_landmark_17_x','computer_pose_landmark_17_y',
      'computer_pose_landmark_17_z','computer_pose_landmark_18_x','computer_pose_landmark_18_y',
      'computer_pose_landmark_18_z','computer_pose_landmark_19_x','computer_pose_landmark_19_y',
      'computer_pose_landmark_19_z','computer_pose_landmark_20_x','computer_pose_landmark_20_y',
      'computer_pose_landmark_20_z','computer_pose_landmark_21_x','computer_pose_landmark_21_y',
      'computer_pose_landmark_21_z','computer_pose_landmark_22_x','computer_pose_landmark_22_y',
      'computer_pose_landmark_22_z','computer_pose_landmark_23_x','computer_pose_landmark_23_y',
      'computer_pose_landmark_23_z','computer_pose_landmark_24_x','computer_pose_landmark_24_y',
      'computer_pose_landmark_24_z','computer_pose_landmark_25_x','computer_pose_landmark_25_y',
      'computer_pose_landmark_25_z','computer_pose_landmark_26_x','computer_pose_landmark_26_y',
      'computer_pose_landmark_26_z','computer_pose_landmark_27_x','computer_pose_landmark_27_y',
      'computer_pose_landmark_27_z','computer_pose_landmark_28_x','computer_pose_landmark_28_y',
      'computer_pose_landmark_28_z','computer_pose_landmark_29_x','computer_pose_landmark_29_y',
      'computer_pose_landmark_29_z','computer_pose_landmark_30_x','computer_pose_landmark_30_y',
      'computer_pose_landmark_30_z','computer_pose_landmark_31_x','computer_pose_landmark_31_y',
      'computer_pose_landmark_31_z','computer_pose_landmark_32_x','computer_pose_landmark_32_y',
      'computer_pose_landmark_32_z','computer_pose_landmark_33_x','computer_pose_landmark_33_y',
      'computer_pose_landmark_33_z','phone_pose_landmark_1_x','phone_pose_landmark_1_y',
      'phone_pose_landmark_1_z','phone_pose_landmark_2_x','phone_pose_landmark_2_y',
      'phone_pose_landmark_2_z','phone_pose_landmark_3_x','phone_pose_landmark_3_y',
      'phone_pose_landmark_3_z','phone_pose_landmark_4_x','phone_pose_landmark_4_y',
      'phone_pose_landmark_4_z','phone_pose_landmark_5_x','phone_pose_landmark_5_y',
      'phone_pose_landmark_5_z','phone_pose_landmark_6_x','phone_pose_landmark_6_y',
      'phone_pose_landmark_6_z','phone_pose_landmark_7_x','phone_pose_landmark_7_y',
      'phone_pose_landmark_7_z','phone_pose_landmark_8_x','phone_pose_landmark_8_y',
      'phone_pose_landmark_8_z','phone_pose_landmark_9_x','phone_pose_landmark_9_y',
      'phone_pose_landmark_9_z','phone_pose_landmark_10_x','phone_pose_landmark_10_y',
      'phone_pose_landmark_10_z','phone_pose_landmark_11_x','phone_pose_landmark_11_y',
      'phone_pose_landmark_11_z','phone_pose_landmark_12_x','phone_pose_landmark_12_y',
      'phone_pose_landmark_12_z','phone_pose_landmark_13_x','phone_pose_landmark_13_y',
      'phone_pose_landmark_13_z','phone_pose_landmark_14_x','phone_pose_landmark_14_y',
      'phone_pose_landmark_14_z','phone_pose_landmark_15_x','phone_pose_landmark_15_y',
      'phone_pose_landmark_15_z','phone_pose_landmark_16_x','phone_pose_landmark_16_y',
      'phone_pose_landmark_16_z','phone_pose_landmark_17_x','phone_pose_landmark_17_y',
      'phone_pose_landmark_17_z','phone_pose_landmark_18_x','phone_pose_landmark_18_y',
      'phone_pose_landmark_18_z','phone_pose_landmark_19_x','phone_pose_landmark_19_y',
      'phone_pose_landmark_19_z','phone_pose_landmark_20_x','phone_pose_landmark_20_y',
      'phone_pose_landmark_20_z','phone_pose_landmark_21_x','phone_pose_landmark_21_y',
      'phone_pose_landmark_21_z','phone_pose_landmark_22_x','phone_pose_landmark_22_y',
      'phone_pose_landmark_22_z','phone_pose_landmark_23_x','phone_pose_landmark_23_y',
      'phone_pose_landmark_23_z','phone_pose_landmark_24_x','phone_pose_landmark_24_y',
      'phone_pose_landmark_24_z','phone_pose_landmark_25_x','phone_pose_landmark_25_y',
      'phone_pose_landmark_25_z','phone_pose_landmark_26_x','phone_pose_landmark_26_y',
      'phone_pose_landmark_26_z','phone_pose_landmark_27_x','phone_pose_landmark_27_y',
      'phone_pose_landmark_27_z','phone_pose_landmark_28_x','phone_pose_landmark_28_y',
      'phone_pose_landmark_28_z','phone_pose_landmark_29_x','phone_pose_landmark_29_y',
      'phone_pose_landmark_29_z','phone_pose_landmark_30_x','phone_pose_landmark_30_y',
      'phone_pose_landmark_30_z','phone_pose_landmark_31_x','phone_pose_landmark_31_y',
      'phone_pose_landmark_31_z','phone_pose_landmark_32_x','phone_pose_landmark_32_y',
      'phone_pose_landmark_32_z','phone_pose_landmark_33_x','phone_pose_landmark_33_y',
      'phone_pose_landmark_33_z','RESPIRATION_1','EMG_1','EEG_1','EEG_2'
  ]

    file_exists = os.path.exists(csv_file)

    # Ensure the DataFrame contains all predefined columns, adding NaN where missing
    features_df = features_df.reindex(columns=predefined_columns)

    if not file_exists:
        # If the file doesn't exist, write the header and the data
        features_df.to_csv(csv_file, mode='w', index=False, header=True)
    else:
        # If the file exists, append the data without the header
        features_df.to_csv(csv_file, mode='a', index=False, header=False)

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
 
