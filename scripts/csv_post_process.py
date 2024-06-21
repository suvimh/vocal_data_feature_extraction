import pandas as pd
import ast
import re

# dlib - face = 68 landmarks 
# mediapipe - pose = 33 landmarks


def fix_csv_format(CSV_INPUT, CSV_OUTPUT):
    df = pd.read_csv(CSV_INPUT)

    # Expand the 'Tristimulus' and 'MFCC_FB40' columns for each audio source
    df =expand_mfcc_tristiumulus(df, "Phone")
    df =expand_mfcc_tristiumulus(df, "Computer")
    df =expand_mfcc_tristiumulus(df, "Mic")

    # change CSV file to have the correct format for the data -- video landmarks
    df = expand_face_landmarks(df, "Phone Face Landmarks", "Phone Face")
    df = expand_pose_landmarks(df, "Phone Pose Landmarks", "Phone Pose")
    df = expand_face_landmarks(df, "Computer Face Landmarks", "Computer Face")
    df = expand_pose_landmarks(df, "Computer Pose Landmarks", "Computer Pose")

    df.to_csv(CSV_OUTPUT, index=False)


def preprocess_list_string(s):
    # Replace multiple spaces with single space, then replace spaces with commas
    s = re.sub(r'\s+', ' ', s.strip())
    s = s.replace(' ', ',')
    # Remove any leading or trailing commas
    s = s.replace('[,', '[')
    return s

def expand_mfcc_tristiumulus(df, source):
    # Preprocess the 'Tristimulus' column to make it a valid list string
    df[f"{source} Tristimulus"] = df[f"{source} Tristimulus"].apply(preprocess_list_string)
    # Expand the 'Tristimulus' column into separate columns
    tristimulus_cols = df[f"{source} Tristimulus"].apply(lambda x: pd.Series(ast.literal_eval(x)))
    tristimulus_cols.columns = [f"{source} Tristimulus{i+1}" for i in range(tristimulus_cols.shape[1])]

    # Preprocess the 'MFCC_FB40' column to make it a valid list string
    df[f"{source} MFCC FB40"] = df[f"{source} MFCC FB40"].apply(preprocess_list_string)
    # Expand the 'MFCC_FB40' column into separate columns
    mfcc_cols = df[f"{source} MFCC FB40"].apply(lambda x: pd.Series(ast.literal_eval(x)))
    mfcc_cols.columns = [f"{source} MFCC FB40 {i+1}" for i in range(mfcc_cols.shape[1])]

    tristimulus_pos = df.columns.get_loc(f"{source} Tristimulus")
    mfcc_pos = df.columns.get_loc(f"{source} MFCC FB40")

    df = df.drop([f"{source} Tristimulus", f"{source} MFCC FB40"], axis=1)

    # Insert the new columns at the original positions
    for i, col in enumerate(tristimulus_cols.columns):
        df.insert(tristimulus_pos + i, col, tristimulus_cols.iloc[:, i])

    for i, col in enumerate(mfcc_cols.columns):
        df.insert(mfcc_pos + i, col, mfcc_cols.iloc[:, i])
    
    return df

# Define a function to expand landmarks
def expand_face_landmarks(df, column, prefix):
    # Parse the landmark data into lists of pd.Series
    landmarks = df[column].apply(lambda x: ast.literal_eval(x))
    
    if "Face" in column:
        max_landmarks = 68
    else:
        raise ValueError("Invalid prefix")

    # Create a dictionary to store the new columns
    new_cols = {f'{prefix}_Landmark_{i+1}_x': [] for i in range(max_landmarks)}
    new_cols.update({f'{prefix}_Landmark_{i+1}_y': [] for i in range(max_landmarks)})
    new_cols.update({f'{prefix}_Landmark_{i+1}_z': [] for i in range(max_landmarks)})

    for row in landmarks:
        for i in range(max_landmarks):
            if i < len(row) and isinstance(row[i], list) and len(row[i]) == 3:
                x, y, z = row[i]
            else:
                x, y, z = None, None, None
            
            new_cols[f'{prefix}_Landmark_{i+1}_x'].append(x)
            new_cols[f'{prefix}_Landmark_{i+1}_y'].append(y)
            new_cols[f'{prefix}_Landmark_{i+1}_z'].append(z)

    # Convert the dictionary to a DataFrame
    new_cols_df = pd.DataFrame(new_cols)

    # Get the position of the original column
    col_idx = df.columns.get_loc(column)

    # Drop the original column
    df = df.drop(column, axis=1)

    # Insert new columns into the DataFrame at the original column's position
    for i in range(max_landmarks):
        df.insert(col_idx + 3*i, f'{prefix}_Landmark_{i+1}_x', new_cols_df[f'{prefix}_Landmark_{i+1}_x'])
        df.insert(col_idx + 3*i + 1, f'{prefix}_Landmark_{i+1}_y', new_cols_df[f'{prefix}_Landmark_{i+1}_y'])
        df.insert(col_idx + 3*i + 2, f'{prefix}_Landmark_{i+1}_z', new_cols_df[f'{prefix}_Landmark_{i+1}_z'])

    return df

def expand_pose_landmarks(df, column, prefix):
    #print(df[column])
    landmarks = df[column].apply(lambda x: ast.literal_eval(x))

    if "Pose" in column:
        max_landmarks = 33
    else:
        raise ValueError("Invalid prefix")
    
    # Create a dictionary to store the new columns
    new_cols = {f'{prefix}_Landmark_{i+1}_x': [] for i in range(max_landmarks)}
    new_cols.update({f'{prefix}_Landmark_{i+1}_y': [] for i in range(max_landmarks)})
    new_cols.update({f'{prefix}_Landmark_{i+1}_z': [] for i in range(max_landmarks)})

    for data in landmarks:
        for i in range(max_landmarks):
            if i < len(data) and isinstance(data[i], dict) and 'x' in data[i] and 'y' in data[i] and 'z' in data[i]:
                x, y, z = data[i]['x'], data[i]['y'], data[i]['z']
            else:
                x, y, z = None, None, None
            
            new_cols[f'{prefix}_Landmark_{i+1}_x'].append(x)
            new_cols[f'{prefix}_Landmark_{i+1}_y'].append(y)
            new_cols[f'{prefix}_Landmark_{i+1}_z'].append(z)

    # Convert the dictionary to a DataFrame
    new_cols_df = pd.DataFrame(new_cols)

    # Get the position of the original column
    col_idx = df.columns.get_loc(column)

    # Drop the original column
    df = df.drop(column, axis=1)

    # Insert new columns into the DataFrame at the original column's position
    for i in range(max_landmarks):
        if any(new_cols_df[f'{prefix}_Landmark_{i+1}_x'].notna()):
            df.insert(col_idx, f'{prefix}_Landmark_{i+1}_x', new_cols_df[f'{prefix}_Landmark_{i+1}_x'])
            col_idx += 1
        if any(new_cols_df[f'{prefix}_Landmark_{i+1}_y'].notna()):
            df.insert(col_idx, f'{prefix}_Landmark_{i+1}_y', new_cols_df[f'{prefix}_Landmark_{i+1}_y'])
            col_idx += 1
        if any(new_cols_df[f'{prefix}_Landmark_{i+1}_z'].notna()):
            df.insert(col_idx, f'{prefix}_Landmark_{i+1}_z', new_cols_df[f'{prefix}_Landmark_{i+1}_z'])
            col_idx += 1

    return df