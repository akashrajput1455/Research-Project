from pathlib import Path
from src.models import MCNN
import torch

from src.dataset import create_part_a_dataset
# from src.models import BaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = 512


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
# print("BASELINE CNN VERIFICATION")
print("MCNN VERIFICATION")
print("=" * 60)

print("Device:", device)


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

train_dataset, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    # image_size=IMAGE_SIZE
    image_size=512,
    density_size=64
)

sample = train_dataset[0]

image = sample["image"].unsqueeze(0)

print("\nInput information")
print("-" * 60)

print("Image shape:", image.shape)


# --------------------------------------------------
# Create model
# --------------------------------------------------

# model = BaselineCNN().to(device)

model = MCNN().to(device)

image = image.to(device)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

with torch.no_grad():

    prediction = model(image)


print("\nModel information")
print("-" * 60)

print("Prediction shape:", prediction.shape)


# --------------------------------------------------
# Count parameters
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print("Total parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)


# --------------------------------------------------
# Prediction statistics
# --------------------------------------------------

print("\nPrediction statistics")
print("-" * 60)

print("Minimum:", prediction.min().item())
print("Maximum:", prediction.max().item())
print("Mean   :", prediction.mean().item())
print("Sum    :", prediction.sum().item())


print("\nModel verification completed.")