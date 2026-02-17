# train.py
import argparse
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from utils import set_seed
from config import *
from preprocess import prepare_dataframes, apply_path_checks, propagate_labels, load_and_prepare_metadata, generate_image_paths
from dataset import CustomDataset
from model import CustomConvNeXtWithAttention, FocalLoss
from evaluate import evaluate_model_comprehensive



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=TRAIN_PATH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--folds', type=int, default=NUM_FOLDS)
    parser.add_argument('--save_path', type=str, default="best_axial_t2_model.pth")
    args = parser.parse_args()

    set_seed(SEED)

    # Load and prepare data
    train_csv = args.data_path + 'train.csv'
    label_csv = args.data_path + 'train_label_coordinates.csv'
    desc_csv = args.data_path + 'train_series_descriptions.csv'
    final_df = prepare_dataframes(train_csv, label_csv, desc_csv, args.data_path)
    train_data = apply_path_checks(final_df, args.data_path)

    train_image_paths = generate_image_paths(train_desc, f'{args.data_path}/train_images')
    metadata_df = load_and_prepare_metadata(train_image_paths)
    filled_metadata = propagate_labels(metadata_df, train_data)

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])

    # Class weights
    class_counts = filled_metadata[filled_metadata['series_description'] == SERIES_DESCRIPTION]['severity'].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights /= class_weights.sum()
    class_weights = class_weights.to(DEVICE)

    criterion = FocalLoss(gamma=2, alpha=class_weights)

    # Run training
    metrics, avg_f1, testloader, test_df = train_model_kfold_with_test(
        model_class=CustomConvNeXtWithAttention,
        data_df=filled_metadata,
        series_description=SERIES_DESCRIPTION,
        transform=transform,
        criterion=criterion,
        device=DEVICE,
        num_folds=args.folds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=PATIENCE,
        accumulation_steps=ACCUMULATION_STEPS,
        test_size=TEST_SIZE,
        save_path=args.save_path
    )

    # Final evaluation
    model = CustomConvNeXtWithAttention(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(args.save_path, weights_only=True))
    evaluate_model_comprehensive(model, testloader, criterion, class_names=CLASS_NAMES)

if __name__ == "__main__":
    main()
