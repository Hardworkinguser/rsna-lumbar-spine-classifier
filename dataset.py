# dataset.py
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image

def load_dicom(image_path):
    ds = pydicom.dcmread(image_path)
    image = ds.pixel_array
    image = (image / np.max(image) * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    return image

def crop_center(image, x, y, crop_size=64):
    h, w = image.shape[:2]
    left = max(0, x - crop_size // 2)
    top = max(0, y - crop_size // 2)
    right = min(w, x + crop_size // 2)
    bottom = min(h, y + crop_size // 2)
    cropped = Image.fromarray(image).crop((left, top, right, bottom))
    return np.array(cropped)

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = load_dicom(row['image_path'])
        image = crop_center(image, row['x'], row['y'])
        if self.transform:
            image = self.transform(image)
        label = row['severity']  # already mapped to 0,1,2 in preprocess
        return image, torch.tensor(label, dtype=torch.long)
