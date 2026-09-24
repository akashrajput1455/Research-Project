from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_part_a_dataset
from src.models import MobileNetV3SmallCrowdCounter


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 512
DENSITY_SIZE = 64

BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.0001

VALIDATION_SIZE = 30

MODEL_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "models"
)

MODEL_PATH = (
    MODEL_DIR
    / "mobilenetv3_best.pth"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MOBILENETV3-SMALL CROWD COUNTING TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Density size: {DENSITY_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load ShanghaiTech Part A
    # --------------------------------------------------------

    train_dataset, test_dataset = create_part_a_dataset(
        PROJECT_ROOT,
        image_size=IMAGE_SIZE,
        density_size=DENSITY_SIZE
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = (
        len(train_dataset)
        - VALIDATION_SIZE
    )

    validation_size = VALIDATION_SIZE

    train_subset, validation_subset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )

    print()
    print("Dataset split")
    print("-" * 60)

    print(
        f"Training samples:   "
        f"{len(train_subset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_subset)}"
    )

    print(
        f"Test samples:       "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MobileNetV3SmallCrowdCounter().to(
        DEVICE
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Model information")
    print("-" * 60)

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Best model tracking
    # --------------------------------------------------------

    best_validation_mae = float("inf")

    best_epoch = 0

    print()
    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        total_train_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(DEVICE)

            target_density = (
                batch["density"].to(DEVICE)
            )

            optimizer.zero_grad()

            predicted_density = model(images)

            loss = criterion(
                predicted_density,
                target_density
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        total_validation_loss = 0.0

        validation_absolute_errors = []

        with torch.no_grad():

            for batch in validation_loader:

                images = batch["image"].to(DEVICE)

                target_density = (
                    batch["density"].to(DEVICE)
                )

                target_count = (
                    batch["count"].to(DEVICE)
                )

                predicted_density = model(
                    images
                )

                validation_loss = criterion(
                    predicted_density,
                    target_density
                )

                total_validation_loss += (
                    validation_loss.item()
                )

                # Convert density map to count
                predicted_count = (
                    predicted_density.sum(
                        dim=(1, 2, 3)
                    )
                )

                absolute_error = torch.abs(
                    predicted_count
                    - target_count
                )

                validation_absolute_errors.extend(
                    absolute_error
                    .cpu()
                    .numpy()
                    .tolist()
                )

        average_validation_loss = (
            total_validation_loss
            / len(validation_loader)
        )

        validation_mae = (
            sum(validation_absolute_errors)
            / len(validation_absolute_errors)
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation_mae < best_validation_mae:

            best_validation_mae = (
                validation_mae
            )

            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            best_marker = " <-- BEST"

        else:

            best_marker = ""

        # ----------------------------------------------------
        # Epoch output
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: "
            f"{average_train_loss:.6f} "
            f"Val Loss: "
            f"{average_validation_loss:.6f} "
            f"Val MAE: "
            f"{validation_mae:.4f}"
            f"{best_marker}"
        )

    # ========================================================
    # COMPLETED
    # ========================================================

    print()
    print("=" * 60)
    print("MOBILENETV3-SMALL TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Validation MAE: "
        f"{best_validation_mae:.4f}"
    )

    print()
    print("Best model saved to:")

    print(MODEL_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()