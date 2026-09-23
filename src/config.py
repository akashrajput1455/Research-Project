from pathlib import Path
import torch


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data" / "ShanghaiTech"
PART_A_ROOT = DATA_ROOT / "part_A_final"

TRAIN_IMAGE_DIR = PART_A_ROOT / "train_data" / "images"
TRAIN_GT_DIR = PART_A_ROOT / "train_data" / "ground-truth"

TEST_IMAGE_DIR = PART_A_ROOT / "test_data" / "images"
TEST_GT_DIR = PART_A_ROOT / "test_data" / "ground-truth"

VISUALIZATION_DIR = PROJECT_ROOT / "experiments" / "visualizations"


# ============================================================
# Dataset settings
# ============================================================

IMAGE_SIZE = 512

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# ============================================================
# Density-map settings
# ============================================================

KNN_K = 3

BETA = 0.3

FALLBACK_SIGMA = 15.0


# ============================================================
# Hardware
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Create required directories
# ============================================================

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)