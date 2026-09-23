from pathlib import Path

import torch

from dataset import create_part_a_dataset
from models import MCNN


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\akash\OneDrive\Desktop\Research\CrowdCounting-Lightweight"
)

IMAGE_SIZE = 512
DENSITY_SIZE = 64

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "mcnn_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("MCNN DIAGNOSTIC CHECK")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")

train_dataset, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE,
    density_size=DENSITY_SIZE
)

print()
print("Dataset loaded")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = MCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle both normal state_dict and checkpoint dictionaries
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print()
print("MCNN model loaded successfully")


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("Model information")
print("-" * 60)
print(f"Total parameters:     {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# ============================================================
# CHECK PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("PREDICTION DIAGNOSTICS")
print("=" * 60)

num_samples = min(5, len(test_dataset))

predicted_counts = []
ground_truth_counts = []

with torch.no_grad():

    for index in range(num_samples):

        # ----------------------------------------------------
        # IMPORTANT:
        # ShanghaiTechDataset returns a dictionary
        # ----------------------------------------------------

        sample = test_dataset[index]

        image = sample["image"]
        density = sample["density"]
        gt_count = sample["count"]
        image_name = sample["image_name"]

        # Add batch dimension
        image = image.unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model(image)

        # Remove batch dimension
        prediction = prediction.squeeze(0)

        # ----------------------------------------------------
        # Convert density map to crowd count
        # ----------------------------------------------------

        predicted_count = prediction.sum().item()

        ground_truth_count = gt_count.item()

        # ----------------------------------------------------
        # Prediction statistics
        # ----------------------------------------------------

        prediction_min = prediction.min().item()
        prediction_max = prediction.max().item()
        prediction_mean = prediction.mean().item()
        prediction_std = prediction.std().item()

        prediction_sum = prediction.sum().item()

        predicted_counts.append(
            predicted_count
        )

        ground_truth_counts.append(
            ground_truth_count
        )

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print()
        print(f"Sample {index + 1}")
        print("-" * 60)

        print(f"Image name:          {image_name}")

        print(
            f"Image shape:         "
            f"{tuple(sample['image'].shape)}"
        )

        print(
            f"GT density shape:    "
            f"{tuple(density.shape)}"
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
            f"{abs(predicted_count - ground_truth_count):.4f}"
        )

        print()
        print("Prediction statistics")

        print(
            f"Minimum value:       "
            f"{prediction_min:.8f}"
        )

        print(
            f"Maximum value:       "
            f"{prediction_max:.8f}"
        )

        print(
            f"Mean value:          "
            f"{prediction_mean:.8f}"
        )

        print(
            f"Standard deviation:  "
            f"{prediction_std:.8f}"
        )

        print(
            f"Density map sum:     "
            f"{prediction_sum:.4f}"
        )


# ============================================================
# CHECK WHETHER MODEL IS COLLAPSING
# ============================================================

print()
print("=" * 60)
print("COLLAPSE CHECK")
print("=" * 60)

print()
print("Ground-truth counts:")
for i, count in enumerate(ground_truth_counts, start=1):
    print(f"Sample {i}: {count:.4f}")

print()
print("Predicted counts:")
for i, count in enumerate(predicted_counts, start=1):
    print(f"Sample {i}: {count:.4f}")


# Calculate prediction variation
if len(predicted_counts) > 1:

    predicted_tensor = torch.tensor(
        predicted_counts,
        dtype=torch.float32
    )

    prediction_average = predicted_tensor.mean().item()
    prediction_std_across_images = (
        predicted_tensor.std().item()
    )

    print()
    print(
        f"Average predicted count: "
        f"{prediction_average:.4f}"
    )

    print(
        f"Prediction std across images: "
        f"{prediction_std_across_images:.4f}"
    )

    # --------------------------------------------------------
    # Basic diagnostic interpretation
    # --------------------------------------------------------

    if prediction_std_across_images < 5:

        print()
        print(
            "WARNING: Predictions are very similar "
            "across different images."
        )

        print(
            "This suggests that the MCNN may be "
            "learning a near-constant density map."
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
# FINISHED
# ============================================================

print()
print("=" * 60)
print("MCNN DIAGNOSTIC COMPLETED")
print("=" * 60)