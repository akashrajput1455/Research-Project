from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_part_a_dataset
from src.models import BaselineCNN


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 512

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "baseline_cnn_best.pth"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load dataset
# ============================================================

train_dataset, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE
)


# ============================================================
# Recreate validation split
# ============================================================

from torch.utils.data import random_split

generator = torch.Generator()
generator.manual_seed(42)

validation_size = int(
    len(train_dataset) * 0.10
)

training_size = (
    len(train_dataset) - validation_size
)

_, validation_dataset = random_split(
    train_dataset,
    [training_size, validation_size],
    generator=generator
)


# ============================================================
# Load model
# ============================================================

model = BaselineCNN().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


print("=" * 70)
print("BASELINE CNN VALIDATION")
print("=" * 70)

print("Device:", device)
print("Validation samples:", len(validation_dataset))


# ============================================================
# Evaluate validation samples
# ============================================================

absolute_errors = []
squared_errors = []

# Store one example for visualization
example_image = None
example_density = None
example_prediction = None
example_gt_count = None


with torch.no_grad():

    for index in range(len(validation_dataset)):

        sample = validation_dataset[index]

        image = sample["image"].unsqueeze(0).to(device)

        ground_truth_density = (
            sample["density"]
            .unsqueeze(0)
            .to(device)
        )

        ground_truth_count = (
            sample["count"].item()
        )

        prediction = model(image)

        predicted_count = (
            prediction.sum().item()
        )

        error = abs(
            predicted_count -
            ground_truth_count
        )

        squared_error = (
            predicted_count -
            ground_truth_count
        ) ** 2

        absolute_errors.append(error)

        squared_errors.append(squared_error)

        # Save first sample
        if index == 0:

            example_image = (
                sample["image"]
                .cpu()
            )

            example_density = (
                ground_truth_density
                .squeeze(0)
                .squeeze(0)
                .cpu()
            )

            example_prediction = (
                prediction
                .squeeze(0)
                .squeeze(0)
                .cpu()
            )

            example_gt_count = (
                ground_truth_count
            )


# ============================================================
# Calculate metrics
# ============================================================

mae = sum(absolute_errors) / len(
    absolute_errors
)

rmse = (
    sum(squared_errors) /
    len(squared_errors)
) ** 0.5


print("\nValidation Results")
print("-" * 70)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)


# ============================================================
# First validation sample
# ============================================================

predicted_count = example_prediction.sum().item()

print("\nFirst validation sample")
print("-" * 70)

print(
    "Ground-truth count:",
    example_gt_count
)

print(
    "Predicted count:",
    predicted_count
)

print(
    "Absolute error:",
    abs(
        predicted_count -
        example_gt_count
    )
)


# ============================================================
# Visualization
# ============================================================

image_display = (
    example_image
    .permute(1, 2, 0)
)


plt.figure(figsize=(18, 5))


# Original image
plt.subplot(1, 3, 1)

plt.imshow(image_display)

plt.title(
    f"Original Image\nGT Count = {example_gt_count:.0f}"
)

plt.axis("off")


# Ground truth
plt.subplot(1, 3, 2)

plt.imshow(
    example_density,
    cmap="jet"
)

plt.colorbar()

plt.title(
    f"Ground-Truth Density\nSum = {example_density.sum():.2f}"
)

plt.axis("off")


# Prediction
plt.subplot(1, 3, 3)

plt.imshow(
    example_prediction,
    cmap="jet"
)

plt.colorbar()

plt.title(
    f"Predicted Density\nSum = {predicted_count:.2f}"
)

plt.axis("off")


plt.tight_layout()

plt.show()