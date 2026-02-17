# config.py
import torch

# Paths
TRAIN_PATH = '/rsna-2024-lumbar-spine-degenerative-classification2/'
METADATA_FILE = 'dicom_metadata.csv'

# Hyperparameters
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_FOLDS = 5
PATIENCE = 7
ACCUMULATION_STEPS = 4
TEST_SIZE = 0.1
CROP_SIZE = 64
IMAGE_SIZE = (224, 224)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SERIES_DESCRIPTION = "Axial T2"
LEVEL = "L4/L5"

# Label mapping
LABEL_MAP = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
CLASS_NAMES = ['normal_mild', 'moderate', 'severe']
