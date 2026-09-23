from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from src.density_map import (
    load_points_from_mat,
    calculate_sigmas,
    generate_density_map
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ShanghaiTech"
)


PART = "part_A_final"

IMAGE_SIZE = 512


# ============================================================
# Dataset paths
# ============================================================

IMAGE_DIR = (
    DATASET_ROOT
    / PART
    / "train_data"
    / "images"
)


GT_DIR = (
    DATASET_ROOT
    / PART
    / "train_data"
    / "ground_truth"
)


# ============================================================
# Find first image
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("*.jpg")
)


if len(image_paths) == 0:

    raise RuntimeError(
        f"No images found:\n{IMAGE_DIR}"
    )


image_path = image_paths[0]


# ============================================================
# Find corresponding annotation
# ============================================================

image_name = image_path.stem


gt_path = (
    GT_DIR
    / f"GT_{image_name}.mat"
)


if not gt_path.exists():

    raise RuntimeError(
        f"Ground truth not found:\n{gt_path}"
    )


# ============================================================
# Load image
# ============================================================

image = Image.open(
    image_path
).convert("RGB")


original_width, original_height = (
    image.size
)


print(
    "Image:",
    image_path.name
)

print(
    "Original size:",
    image.size
)


# ============================================================
# Load annotation points
# ============================================================

points = load_points_from_mat(
    gt_path
)


print(
    "Number of annotated people:",
    len(points)
)


# ============================================================
# Resize image
# ============================================================

image_resized = image.resize(
    (
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    Image.Resampling.BICUBIC
)


# ============================================================
# Scale annotation coordinates
# ============================================================

points_scaled = points.copy()


if len(points_scaled) > 0:

    points_scaled[:, 0] *= (
        IMAGE_SIZE
        /
        original_width
    )

    points_scaled[:, 1] *= (
        IMAGE_SIZE
        /
        original_height
    )


# ============================================================
# Calculate sigmas
# ============================================================

sigmas = calculate_sigmas(
    points_scaled
)


# ============================================================
# Generate density map
# ============================================================

density_map = generate_density_map(
    IMAGE_SIZE,
    IMAGE_SIZE,
    points_scaled,
    sigmas
)


# ============================================================
# Count verification
# ============================================================

ground_truth_count = len(
    points_scaled
)


density_map_count = (
    density_map.sum()
)


difference = abs(
    ground_truth_count
    -
    density_map_count
)


print("\n")
print("=" * 60)

print(
    "Ground-truth count:",
    ground_truth_count
)

print(
    "Density-map sum:",
    density_map_count
)

print(
    "Absolute difference:",
    difference
)

print("=" * 60)


# ============================================================
# Visualization
# ============================================================

plt.figure(
    figsize=(16, 5)
)


# ------------------------------------------------------------
# Original image
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    1
)

plt.imshow(
    image_resized
)

plt.title(
    f"Original Image\n"
    f"Count = {ground_truth_count}"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Head annotations
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    2
)

plt.imshow(
    image_resized
)


if len(points_scaled) > 0:

    plt.scatter(
        points_scaled[:, 0],
        points_scaled[:, 1],
        s=5
    )


plt.title(
    "Ground-Truth Head Points"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Density map
# ------------------------------------------------------------

plt.subplot(
    1,
    3,
    3
)

plt.imshow(
    density_map,
    cmap="jet"
)

plt.colorbar()

plt.title(
    f"Ground-Truth Density Map\n"
    f"Sum = {density_map_count:.2f}"
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.show()