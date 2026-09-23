from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.dataset import create_part_a_dataset
from src.models import BaselineCNN


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 512

DENSITY_SIZE = 64

BATCH_SIZE = 2

EPOCHS = 30

LEARNING_RATE = 0.0001

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("BASELINE CNN FINAL TRAINING")
print("=" * 70)

print("Device:", device)
print("Image size:", IMAGE_SIZE)
print("Density size:", DENSITY_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# Dataset
# ============================================================

train_dataset_full, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE,
    density_size=DENSITY_SIZE
)


# ============================================================
# Train / Validation split
# ============================================================

validation_size = int(
    len(train_dataset_full)
    * VALIDATION_SPLIT
)

training_size = (
    len(train_dataset_full)
    - validation_size
)


generator = torch.Generator()

generator.manual_seed(
    RANDOM_SEED
)


train_dataset, validation_dataset = random_split(
    train_dataset_full,
    [
        training_size,
        validation_size
    ],
    generator=generator
)


print("\nDataset split")
print("-" * 70)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = BaselineCNN().to(device)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Best model tracking
# ============================================================

best_validation_mae = float("inf")

best_epoch = 0


models_directory = (
    PROJECT_ROOT
    / "experiments"
    / "models"
)

models_directory.mkdir(
    parents=True,
    exist_ok=True
)


best_model_path = (
    models_directory
    / "baseline_cnn_best.pth"
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    training_loss = 0.0


    for batch in train_loader:

        images = batch["image"].to(
            device
        )

        densities = batch["density"].to(
            device
        )


        optimizer.zero_grad()


        predictions = model(
            images
        )


        loss = criterion(
            predictions,
            densities
        )


        loss.backward()

        optimizer.step()


        training_loss += loss.item()


    training_loss /= len(
        train_loader
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    absolute_errors = []


    with torch.no_grad():

        for batch in validation_loader:

            images = batch["image"].to(
                device
            )

            densities = batch["density"].to(
                device
            )

            ground_truth_counts = (
                batch["count"].to(device)
            )


            predictions = model(
                images
            )


            loss = criterion(
                predictions,
                densities
            )


            validation_loss += loss.item()


            # ----------------------------------------------
            # Crowd-count evaluation
            # ----------------------------------------------

            predicted_counts = (
                predictions.sum(
                    dim=(1, 2, 3)
                )
            )


            errors = torch.abs(
                predicted_counts
                - ground_truth_counts
            )


            absolute_errors.extend(
                errors.cpu().numpy()
            )


    validation_loss /= len(
        validation_loader
    )


    validation_mae = np.mean(
        absolute_errors
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if validation_mae < best_validation_mae:

        best_validation_mae = validation_mae

        best_epoch = epoch + 1


        torch.save(
            model.state_dict(),
            best_model_path
        )


        best_marker = " <-- BEST"


    else:

        best_marker = ""


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {training_loss:.6f} "
        f"Val Loss: {validation_loss:.6f} "
        f"Val MAE: {validation_mae:.4f}"
        f"{best_marker}"
    )


# ============================================================
# Training completed
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL TRAINING COMPLETED"
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation MAE: "
    f"{best_validation_mae:.4f}"
)

print(
    "Best model saved to:"
)

print(
    best_model_path
)

print("=" * 70)