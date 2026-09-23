from pathlib import Path

import matplotlib.pyplot as plt

from src.dataset import create_part_a_dataset


PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = 512


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset, test_dataset = create_part_a_dataset(
    PROJECT_ROOT,
    image_size=IMAGE_SIZE
)


print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print("Training images:", len(train_dataset))
print("Testing images :", len(test_dataset))


# --------------------------------------------------
# Load one sample
# --------------------------------------------------

sample = train_dataset[0]

image = sample["image"]
density = sample["density"]
count = sample["count"]
image_name = sample["image_name"]


print("\nSample information")
print("-" * 60)

print("Image name :", image_name)
print("Image shape:", image.shape)
print("Density shape:", density.shape)
print("GT count   :", count.item())
print("Density sum:", density.sum().item())


difference = abs(
    count.item() - density.sum().item()
)

print("Difference :", difference)


# --------------------------------------------------
# Visualize
# --------------------------------------------------

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

image_display = image.permute(1, 2, 0)

plt.imshow(image_display)

plt.title(
    f"Input Image\nCount = {count.item():.0f}"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(density.squeeze(0), cmap="jet")

plt.colorbar()

plt.title(
    f"Density Map\nSum = {density.sum().item():.2f}"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(image_display)

plt.imshow(
    density.squeeze(0),
    cmap="jet",
    alpha=0.5
)

plt.title("Density Overlay")

plt.axis("off")


plt.tight_layout()

plt.show()