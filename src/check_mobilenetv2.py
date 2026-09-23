from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_part_a_dataset
from src.models import MobileNetV2CrowdCounter


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 512
DENSITY_SIZE = 64

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "mobilenetv2_best.pth"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("MOBILENETV2 DIAGNOSTIC CHECK")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE,
    density_size=DENSITY_SIZE
)

print()
print("Dataset loaded")
print("-" * 60)
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = MobileNetV2CrowdCounter().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

print()
print("MobileNetV2 model loaded successfully")


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print()
print("Model information")
print("-" * 60)
print(f"Total parameters:     {total_parameters:,}")


# ============================================================
# PREDICTION DIAGNOSTICS
# ============================================================

print()
print("=" * 60)
print("PREDICTION DIAGNOSTICS")
print("=" * 60)

predicted_counts = []

number_of_samples = min(5, len(test_dataset))

with torch.no_grad():

    for index in range(number_of_samples):

        sample = test_dataset[index]

        image = sample["image"].unsqueeze(0).to(DEVICE)
        ground_truth_density = sample["density"]
        ground_truth_count = sample["count"].item()
        image_name = sample["image_name"]

        prediction = model(image)

        predicted_count = prediction.sum().item()

        absolute_error = abs(
            predicted_count - ground_truth_count
        )

        prediction_cpu = prediction.squeeze().cpu()

        minimum_value = prediction_cpu.min().item()
        maximum_value = prediction_cpu.max().item()
        mean_value = prediction_cpu.mean().item()
        standard_deviation = prediction_cpu.std().item()

        density_sum = prediction_cpu.sum().item()

        predicted_counts.append(predicted_count)

        print()
        print(f"Sample {index + 1}")
        print("-" * 60)

        print(
            f"Image name:          {image_name}"
        )

        print(
            f"Image shape:         "
            f"{tuple(sample['image'].shape)}"
        )

        print(
            f"GT density shape:    "
            f"{tuple(ground_truth_density.shape)}"
        )

        print(
            f"Prediction shape:    "
            f"{tuple(prediction.shape)}"
        )

        print(
            f"Ground truth count:  "
            f"{ground_truth_count:.4f}"
        )

        print(
            f"Predicted count:     "
            f"{predicted_count:.4f}"
        )

        print(
            f"Absolute error:      "
            f"{absolute_error:.4f}"
        )

        print()
        print("Prediction statistics")

        print(
            f"Minimum value:       "
            f"{minimum_value:.8f}"
        )

        print(
            f"Maximum value:       "
            f"{maximum_value:.8f}"
        )

        print(
            f"Mean value:          "
            f"{mean_value:.8f}"
        )

        print(
            f"Standard deviation:  "
            f"{standard_deviation:.8f}"
        )

        print(
            f"Density map sum:     "
            f"{density_sum:.4f}"
        )


# ============================================================
# COLLAPSE CHECK
# ============================================================

print()
print("=" * 60)
print("COLLAPSE CHECK")
print("=" * 60)

print()
print("Ground-truth counts:")

for index in range(number_of_samples):

    sample = test_dataset[index]

    print(
        f"Sample {index + 1}: "
        f"{sample['count'].item():.4f}"
    )


print()
print("Predicted counts:")

for index, count in enumerate(predicted_counts):

    print(
        f"Sample {index + 1}: "
        f"{count:.4f}"
    )


average_predicted_count = (
    sum(predicted_counts)
    / len(predicted_counts)
)

predicted_tensor = torch.tensor(
    predicted_counts
)

prediction_std = predicted_tensor.std().item()

print()
print(
    f"Average predicted count: "
    f"{average_predicted_count:.4f}"
)

print(
    f"Prediction std across images: "
    f"{prediction_std:.4f}"
)

if prediction_std < 5:

    print()
    print(
        "WARNING: Predictions are very similar "
        "across images."
    )

    print(
        "The model may have partially collapsed."
    )

else:

    print()
    print(
        "Predictions vary across images."
    )

    print(
        "The model does not appear to be "
        "completely collapsed to one constant count."
    )


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 60)
print("MOBILENETV2 DIAGNOSTIC COMPLETED")
print("=" * 60)