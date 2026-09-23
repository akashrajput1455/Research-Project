from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

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

BATCH_SIZE = 1


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("BASELINE CNN TEST EVALUATION")
print("=" * 70)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

_, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE,
    density_size=DENSITY_SIZE
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("Test samples:", len(test_dataset))


# ============================================================
# Load model
# ============================================================

model = BaselineCNN().to(device)


model_path = (
    PROJECT_ROOT
    / "experiments"
    / "models"
    / "baseline_cnn_best.pth"
)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)


model.eval()


# ============================================================
# Number of parameters
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("\nModel information")
print("-" * 70)

print(
    "Total parameters:",
    total_parameters
)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# Model size
# ============================================================

model_size_bytes = model_path.stat().st_size

model_size_mb = (
    model_size_bytes
    / (1024 * 1024)
)


print(
    f"Model size: {model_size_mb:.2f} MB"
)


# ============================================================
# Test evaluation
# ============================================================

absolute_errors = []

squared_errors = []


inference_times = []


print("\nEvaluating test set...")
print("-" * 70)


with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(
            device
        )

        ground_truth_counts = (
            batch["count"].to(device)
        )


        # ----------------------------------------------------
        # Synchronize GPU before timing
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()


        start_time = time.perf_counter()


        predictions = model(
            images
        )


        if device.type == "cuda":

            torch.cuda.synchronize()


        end_time = time.perf_counter()


        inference_time = (
            end_time - start_time
        )


        inference_times.append(
            inference_time
        )


        # ----------------------------------------------------
        # Convert density map to count
        # ----------------------------------------------------

        predicted_counts = (
            predictions.sum(
                dim=(1, 2, 3)
            )
        )


        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        errors = (
            predicted_counts
            - ground_truth_counts
        )


        absolute_errors.extend(
            torch.abs(errors)
            .cpu()
            .numpy()
        )


        squared_errors.extend(
            (errors ** 2)
            .cpu()
            .numpy()
        )


# ============================================================
# Calculate metrics
# ============================================================

mae = np.mean(
    absolute_errors
)


rmse = np.sqrt(
    np.mean(
        squared_errors
    )
)


average_inference_time = np.mean(
    inference_times
)


fps = (
    1.0
    / average_inference_time
)


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("BASELINE CNN TEST RESULTS")
print("=" * 70)

print(
    f"Test images           : {len(test_dataset)}"
)

print(
    f"MAE                   : {mae:.4f}"
)

print(
    f"RMSE                  : {rmse:.4f}"
)

print(
    f"Parameters            : {total_parameters}"
)

print(
    f"Model size            : {model_size_mb:.2f} MB"
)

print(
    f"Average inference time: {average_inference_time * 1000:.2f} ms"
)

print(
    f"FPS                   : {fps:.2f}"
)

print("=" * 70)