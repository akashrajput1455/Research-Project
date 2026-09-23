from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import create_part_a_dataset
from src.models import MCNN


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 512
DENSITY_SIZE = 64
BATCH_SIZE = 1

MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "mcnn_best.pth"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("MCNN TEST EVALUATION")
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
print("-" * 60)
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples:     {len(test_dataset)}")


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

model = MCNN().to(DEVICE)

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
print("MCNN model loaded successfully")


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
print(f"Total parameters: {total_parameters:,}")


# ============================================================
# MODEL SIZE
# ============================================================

model_size_bytes = MODEL_PATH.stat().st_size
model_size_mb = model_size_bytes / (1024 * 1024)

print(f"Model size:       {model_size_mb:.2f} MB")


# ============================================================
# TEST EVALUATION
# ============================================================

absolute_errors = []
squared_errors = []
inference_times = []

print()
print("=" * 60)
print("RUNNING TEST EVALUATION")
print("=" * 60)

with torch.no_grad():

    for batch_index, batch in enumerate(test_loader):

        images = batch["image"].to(DEVICE)
        ground_truth_counts = batch["count"].to(DEVICE)

        # ----------------------------------------------------
        # Synchronize GPU before timing
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        predictions = model(images)

        # ----------------------------------------------------
        # Synchronize GPU after inference
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        inference_time = end_time - start_time

        inference_times.append(inference_time)

        # ----------------------------------------------------
        # Convert density map to predicted count
        # ----------------------------------------------------

        predicted_counts = predictions.sum(
            dim=(1, 2, 3)
        )

        errors = predicted_counts - ground_truth_counts

        absolute_errors.extend(
            torch.abs(errors).cpu().numpy().tolist()
        )

        squared_errors.extend(
            (errors ** 2).cpu().numpy().tolist()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (batch_index + 1) % 25 == 0:
            print(
                f"Processed {batch_index + 1}/{len(test_loader)} images"
            )


# ============================================================
# CALCULATE METRICS
# ============================================================

mae = sum(absolute_errors) / len(absolute_errors)

rmse = (
    sum(squared_errors) / len(squared_errors)
) ** 0.5

average_inference_time = (
    sum(inference_times) / len(inference_times)
)

fps = 1.0 / average_inference_time


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("MCNN TEST RESULTS")
print("=" * 60)

print(f"Test images:              {len(test_dataset)}")
print(f"MAE:                      {mae:.4f}")
print(f"RMSE:                     {rmse:.4f}")
print(f"Parameters:               {total_parameters:,}")
print(f"Model size:               {model_size_mb:.2f} MB")
print(
    f"Average inference time:   "
    f"{average_inference_time * 1000:.2f} ms"
)
print(f"FPS:                      {fps:.2f}")

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "mcnn_test_results.txt"
)

results_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(results_file, "w") as file:

    file.write("MCNN TEST RESULTS\n")
    file.write("=" * 60 + "\n")

    file.write(
        f"Test images: {len(test_dataset)}\n"
    )

    file.write(
        f"MAE: {mae:.4f}\n"
    )

    file.write(
        f"RMSE: {rmse:.4f}\n"
    )

    file.write(
        f"Parameters: {total_parameters}\n"
    )

    file.write(
        f"Model size: {model_size_mb:.2f} MB\n"
    )

    file.write(
        f"Average inference time: "
        f"{average_inference_time * 1000:.2f} ms\n"
    )

    file.write(
        f"FPS: {fps:.2f}\n"
    )

print()
print(f"Results saved to:")
print(results_file)