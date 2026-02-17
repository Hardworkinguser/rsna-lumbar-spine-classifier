# preprocess.py
import os
import pandas as pd
import numpy as np
import pydicom
import re
import ast
from tqdm import tqdm

def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        if os.path.exists(series_dir):
            images = os.listdir(series_dir)
            image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths

def extract_metadata(dicom_path):
    try:
        ds = pydicom.dcmread(dicom_path)
        return {
            'FilePath': dicom_path,
            'PatientID': ds.PatientID,
            'SeriesDescription': ds.SeriesDescription,
            'ImagePositionPatient': ds.ImagePositionPatient,
            'ImageOrientationPatient': ds.ImageOrientationPatient,
            'PixelSpacing': ds.PixelSpacing,
            'SpacingBetweenSlices': ds.SpacingBetweenSlices if 'SpacingBetweenSlices' in ds else None,
            'Height': ds.Rows,
            'Width': ds.Columns
        }
    except Exception as e:
        print(f"Error reading {dicom_path}: {e}")
        return None

def load_and_prepare_metadata(train_image_paths, metadata_file='dicom_metadata.csv'):
    if os.path.exists(metadata_file):
        print(f"Loading existing metadata from {metadata_file}")
        return pd.read_csv(metadata_file)
    
    print("Extracting metadata from DICOM files...")
    metadata_list = []
    for path in tqdm(train_image_paths):
        meta = extract_metadata(path)
        if meta:
            metadata_list.append(meta)
    
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Saved metadata for {len(metadata_df)} images")
    return metadata_df

def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    for col, val in row.items():
        if col not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = col.split('_')
            condition = ' '.join(word.capitalize() for word in parts[:-2])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(val)
    return pd.DataFrame(data)

def prepare_dataframes(train_csv, label_csv, desc_csv, train_path):
    train = pd.read_csv(train_csv)
    label = pd.read_csv(label_csv)
    train_desc = pd.read_csv(desc_csv)

    new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
    merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
    final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')

    final_merged_df['row_id'] = (
        final_merged_df['study_id'].astype(str) + '_' +
        final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
        final_merged_df['level'].str.lower().str.replace('/', '_')
    )
    final_merged_df['image_path'] = (
        f'{train_path}/train_images/' +
        final_merged_df['study_id'].astype(str) + '/' +
        final_merged_df['series_id'].astype(str) + '/' +
        final_merged_df['instance_number'].astype(str) + '.dcm'
    )

    final_merged_df['severity'] = final_merged_df['severity'].map({'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})
    return final_merged_df

def apply_path_checks(df, train_path):
    def check_exists(p): return os.path.exists(p)
    df['study_exists'] = df['study_id'].apply(lambda sid: check_exists(f'{train_path}/train_images/{sid}'))
    df['series_exists'] = df.apply(lambda row: check_exists(f'{train_path}/train_images/{row["study_id"]}/{row["series_id"]}'), axis=1)
    df['image_exists'] = df['image_path'].apply(check_exists)
    return df[(df['study_exists']) & (df['series_exists']) & (df['image_exists']) & (df['level'] == 'L4/L5')]

def propagate_labels(metadata_df, train_dataf):
    # Your full label propagation logic here (extract_ids, parse_image_position, merging, z-distance <15, top-5 nearest, fillna)
    # ... paste your entire label propagation block from the notebook ...
    # Return filled_metadata at the end
    filled_metadata = metadata_df[metadata_df['severity'].notna()]  # placeholder — replace with your final line
    return filled_metadata
