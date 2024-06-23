'''
    FUNCTIONS FOR EXTRACTING POSE AND FACE LANDMARKS FROM VIDEO FILES 
'''

import os
import re
import logging
import json
import cv2
import dlib
import mediapipe as mp
from tqdm.auto import tqdm
import pandas as pd

# Set up logging configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

PROTO_PATH = "data/shape_predictor_68_face_landmarks.dat"


def get_mediapipe_pose_estimation_for_frames(cleaned_time, input_file_path, output_path=None, source='undef', output_video=False, output_json=False):
    """
    Process mediapipe pose estimation for frames.

    Args:
        cleaned_time (float): The cleaned audio time.
        input_file_path (str): The path to the input file.
        output_path (str, optional): The path to the output directory.
        source (str, optional): The source of the landmarks. Defaults to 'undef'.
        output_video (bool, optional): Whether to output a video with landmarks. Defaults to False.
        output_json (bool, optional): Whether to output a JSON file with pose landmarks. Defaults to False.

    Returns:
        list: The cleaned pose landmarks.
    """
    logging.info(f"Processing mediapipe pose estimation for {input_file_path}.")

    if output_video or output_json:
        output_video_path, output_json_path = get_face_pose_output_paths(input_file_path, output_path, landmark_type='mp_pose')

    pose_landmarks_list = get_mediapipe_landmarks(input_file_path, output_video)

    # Clean pose data based on cleaned audio data
    frame_rate = 25  # frames per second -- based on frame rate from shotcut output videos
    reshaped_pose_landmarks = reshape_pose_landmarks_dataframe(pose_landmarks_list, source)
    cleaned_pose_landmarks = align_landmarks(cleaned_time, reshaped_pose_landmarks, frame_rate)
    
    #output json file with landmarks
    if output_json:
        with open(output_json_path, 'w') as json_file:
            json.dump({'pose_landmarks': cleaned_pose_landmarks}, json_file)
        logging.info(f'Pose landmark locations saved to: {output_json_path}')
    #output video file with landmarks on it
    if output_video:
        logging.info(f'Video with landmarks saved to: {output_video_path}')

    return cleaned_pose_landmarks


def get_mediapipe_landmarks(input_video_path, output_video_path=None, output_video=False):
    """
    Extracts landmarks from a video using the MediaPipe Pose model.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str, optional): The path to save the output video file. Defaults to None.
        output_video (bool, optional): Whether to save the output video. Defaults to False.

    Returns:
        np.ndarray: A numpy array of shape (num_frames, num_landmarks, 3) containing the x, y, and z coordinates
                    of the detected landmarks for each frame. The array is of type float32.
    """
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_video:
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose_landmarks_list = []

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar3 = tqdm(total=total_frames, desc='Processing Frames')

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty video frame.")
                break

            image_copy = image.copy()
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            pose_results = pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = []
                for landmark in pose_results.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z if landmark.HasField('z') else None
                    })
                pose_landmarks_list.append(landmarks)
                mp_drawing.draw_landmarks(
                    image_copy,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            if output_video:
                out.write(image_copy)

            pbar3.update(1)

        pbar3.close()
        cap.release()
        if output_video:
            out.release()
        cv2.destroyAllWindows()

    return pd.DataFrame(pose_landmarks_list)


def reshape_pose_landmarks_dataframe(landmark_df, source):
    """
    Reshape the DataFrame containing landmark data from MediaPipe.

    Args:
        df (pd.DataFrame): DataFrame containing landmark data from MediaPipe.

    Returns:
        pd.DataFrame: Reshaped DataFrame with columns Pose_Landmark_{landmark_index}_{coordinate}.
    """
    reshaped_df = pd.DataFrame()

    for landmark, data in landmark_df.items():
        for idx, value in data.items():
            if pd.notna(value):
                for coord in ['x', 'y', 'z']:  # Assuming keys are 'x', 'y', 'z'
                    new_col_name = f"{source}_pose_landmark_{landmark+1}_{coord}"
                    if coord in value:  # Check if coordinate exists in the value dictionary
                        reshaped_df.loc[idx, new_col_name] = value[coord]
                    else:
                        reshaped_df.loc[idx, new_col_name] = None  # Handle missing coordinates

    return reshaped_df


def get_dlib_face_estimation_for_frames(cleaned_time, input_file_path, output_path=None, source='undef', output_video=False, output_json=False):
    """
    Process dlib face landmarks for frames in a video.

    Args:
        cleaned_time (float): The cleaned time in seconds.
        input_file_path (str): The path to the input video file.
        output_path (str, optional): The path to the output directory.
        source (str, optional): The source of the landmarks. Defaults to 'undef'.
        output_video (bool, optional): Whether to save the video with landmarks. Defaults to False.
        output_json (bool, optional): Whether to save the face landmarks as JSON. Defaults to False.

    Returns:
        list: The cleaned face landmarks.
    """
    logging.info(f"Processing dlib face landmarks for {input_file_path}.")

    output_video_path, output_json_path = None, None
    if output_video or output_json:
        output_video_path, output_json_path = get_face_pose_output_paths(input_file_path, output_path, landmark_type='dlib_face')

    face_landmarks_list = get_dlib_face_landmarks(input_file_path, output_video_path, output_video)
    reshaped_face_landmarks = reshape_face_landmarks_dataframe(face_landmarks_list, source)

    # Clean face data based on cleaned audio data
    frame_rate = 25  # frames per second
    cleaned_face_landmarks = align_landmarks(cleaned_time, reshaped_face_landmarks, frame_rate)

    if output_json:
        with open(output_json_path, 'w') as json_file:
            json.dump({'face_landmarks': cleaned_face_landmarks}, json_file)
        logging.info(f'Face landmark locations saved to: {output_json_path}')

    if output_video:
        logging.info(f'Video with landmarks saved to: {output_video_path}')

    return cleaned_face_landmarks


def get_dlib_face_landmarks(input_video_path, output_video_path=None, output_video=False):
    """
    Extracts facial landmarks from a video using dlib.

    Args:
        input_video_path (str): The path to the input video file.
        output_video_path (str, optional): The path to save the output video file with facial landmarks drawn. Defaults to None.
        output_video (bool, optional): Whether to save the output video file. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing facial landmarks for each frame.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PROTO_PATH)  # Path to the dlib model

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_video:
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    face_landmarks_list = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar2 = tqdm(total=total_frames, desc='Processing Frames')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            face_landmarks = [[p.x, p.y, None] for p in landmarks.parts()]
            face_landmarks_list.append(face_landmarks)
            
            if output_video:
                for p in landmarks.parts():
                    cv2.circle(frame, (p.x, p.y), 1, (0, 255, 0), -1)
                out.write(frame)

        pbar2.update(1)

    pbar2.close()
    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()

    return pd.DataFrame(face_landmarks_list)


def reshape_face_landmarks_dataframe(landmarks_df, source):
    """
    Reshape the DataFrame containing face landmark data.

    Args:
        landmarks_df (pd.DataFrame): DataFrame containing face landmark data.
        source (str): Source identifier for the landmarks (e.g., 'mediapipe').

    Returns:
        pd.DataFrame: Reshaped DataFrame with columns Face_Landmark_{landmark_index}_{coordinate}.
    """
    reshaped_df = pd.DataFrame()

    coordinates = ["x", "y", "z"]

    for column_name, column_data in landmarks_df.items():
        for idx, value in enumerate(column_data):
            if value is not None:  # Check if the value is not None
                x, y, z = value  # Extract x, y, z from the list [x, y, z]
                for coord_idx, coord in enumerate(coordinates):
                    new_col_name = f"{source}_face_landmark_{column_name+1}_{coord}"
                    reshaped_df.loc[idx, new_col_name] = locals()[coord]  # Use locals() to access x, y, z by string

    return reshaped_df


def align_landmarks(cleaned_time, landmarks_df, frame_rate):
    """
    Cleans face/pose data by keeping only the segments that correspond to the cleaned audio segments.

    Args:
        cleaned_time (ndarray): The cleaned time data.
        landmarks_df (DataFrame): The DataFrame of face or pose landmarks.
        frame_rate (int): The frame rate of the video.

    Returns:
        DataFrame: The DataFrame of face or pose landmarks aligned with audio frames.
    """
    aligned_landmarks = []
    video_frame_duration_ms = 1000 / frame_rate

    # Convert cleaned_time from seconds to milliseconds
    cleaned_time_ms = cleaned_time * 1000

    # Iterate over cleaned_time and find corresponding frame indices
    for t in cleaned_time_ms:
        frame_index = int(t // video_frame_duration_ms)
        if frame_index < len(landmarks_df):
            aligned_landmarks.append(landmarks_df.iloc[frame_index])

    # Concatenate the aligned landmarks into a single DataFrame
    aligned_df = pd.concat(aligned_landmarks, axis=1).T.reset_index(drop=True)

    return aligned_df


def get_face_pose_output_paths(input_file_path, output_path, landmark_type):
    '''
    Returns the output paths for the video file with landmarks and the JSON file with landmark locations.

    Parameters:
    - input_file_path (str): The path of the input video file.
    - output_path (str): The path where the output files will be saved.
    - landmark_type (str): The type of landmarks to be extracted. Can be either 'face' for dlib face landmarks file or 'pose' for mediapipe pose estimation.

    Returns:
    - output_video_path (str): The path of the output video file with landmarks.
    - output_json_path (str): The path of the output JSON file with landmark locations.
    '''
    input_filename = os.path.basename(input_file_path)

    output_video_name = re.sub(r'(?i)\.mp4$', '', input_filename) + f'_with_{landmark_type}_landmarks.mp4'
    output_video_path = os.path.join(output_path, output_video_name)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    output_json_name = re.sub(r'(?i)\.mp4$', '', input_filename) + f'_{landmark_type}_landmark_locations.json'
    output_json_path = os.path.join(output_path, output_json_name)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    return output_video_path, output_json_path