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
from src.models import MCNN


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

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

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


# ============================================================
# Training information
# ============================================================

print("=" * 70)
print("MCNN FINAL TRAINING")
print("=" * 70)

print(
    "Device:",
    device
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Density size:",
    DENSITY_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Learning rate:",
    LEARNING_RATE
)


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


# ============================================================
# Dataset information
# ============================================================

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

model = MCNN().to(
    device
)


# ============================================================
# Loss function
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Best model tracking
# ============================================================

best_validation_mae = float(
    "inf"
)

best_epoch = 0


# ============================================================
# Model directory
# ============================================================

models_directory = (
    PROJECT_ROOT
    / "experiments"
    / "models"
)

models_directory.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Best model path
# ============================================================

best_model_path = (
    models_directory
    / "mcnn_best.pth"
)


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    EPOCHS
):

    # ========================================================
    # Training
    # ========================================================

    model.train()

    training_loss = 0.0


    for batch in train_loader:

        images = batch[
            "image"
        ].to(
            device
        )

        densities = batch[
            "density"
        ].to(
            device
        )


        # ----------------------------------------------------
        # Clear previous gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        predictions = model(
            images
        )


        # ----------------------------------------------------
        # Density-map loss
        # ----------------------------------------------------

        loss = criterion(
            predictions,
            densities
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()


        training_loss += (
            loss.item()
        )


    # --------------------------------------------------------
    # Average training loss
    # --------------------------------------------------------

    training_loss /= len(
        train_loader
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    validation_loss = 0.0

    absolute_errors = []


    with torch.no_grad():

        for batch in validation_loader:

            images = batch[
                "image"
            ].to(
                device
            )

            densities = batch[
                "density"
            ].to(
                device
            )

            ground_truth_counts = batch[
                "count"
            ].to(
                device
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            predictions = model(
                images
            )


            # ------------------------------------------------
            # Validation density loss
            # ------------------------------------------------

            loss = criterion(
                predictions,
                densities
            )


            validation_loss += (
                loss.item()
            )


            # ------------------------------------------------
            # Convert density map to crowd count
            # ------------------------------------------------

            predicted_counts = predictions.sum(
                dim=(1, 2, 3)
            )


            # ------------------------------------------------
            # Calculate absolute counting error
            # ------------------------------------------------

            errors = torch.abs(
                predicted_counts
                - ground_truth_counts
            )


            absolute_errors.extend(
                errors.cpu().numpy().tolist()
            )


    # --------------------------------------------------------
    # Average validation loss
    # --------------------------------------------------------

    validation_loss /= len(
        validation_loader
    )


    # --------------------------------------------------------
    # Calculate validation MAE
    # --------------------------------------------------------

    validation_mae = np.mean(
        absolute_errors
    )


    # ========================================================
    # Save best model
    # ========================================================

    if (
        validation_mae
        < best_validation_mae
    ):

        best_validation_mae = (
            validation_mae
        )

        best_epoch = (
            epoch + 1
        )


        torch.save(
            model.state_dict(),
            best_model_path
        )


        best_marker = (
            " <-- BEST"
        )

    else:

        best_marker = ""


    # ========================================================
    # Print epoch results
    # ========================================================

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

print(
    "\n" + "=" * 70
)

print(
    "MCNN TRAINING COMPLETED"
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

print(
    "=" * 70
)