from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from src.density_map import (
    load_points_from_mat,
    calculate_sigmas,
    generate_density_map
)


class ShanghaiTechDataset(Dataset):

    def __init__(
        self,
        image_paths,
        gt_dir,
        image_size=512,
        density_size=64
    ):

        self.image_paths = image_paths
        self.gt_dir = Path(gt_dir)
        self.image_size = image_size
        self.density_size = density_size

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = Path(self.image_paths[index])

        # --------------------------------------------------
        # 1. Load image
        # --------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        # --------------------------------------------------
        # 2. Load ground-truth annotation
        # --------------------------------------------------

        gt_path = self.gt_dir / f"GT_{image_path.stem}.mat"

        points = load_points_from_mat(gt_path)

        # --------------------------------------------------
        # 3. Resize image
        # --------------------------------------------------

        image_tensor = self.image_transform(image)

        # --------------------------------------------------
        # 4. Scale annotation points
        # --------------------------------------------------

        points_scaled = points.copy()

        if len(points_scaled) > 0:

            points_scaled[:, 0] *= (
                self.image_size / original_width
            )

            points_scaled[:, 1] *= (
                self.image_size / original_height
            )

        # --------------------------------------------------
        # 5. Generate 64x64 density map
        # --------------------------------------------------

        density_points = points_scaled.copy()

        if len(density_points) > 0:

            density_points[:, 0] *= (
                self.density_size / self.image_size
            )

            density_points[:, 1] *= (
                self.density_size / self.image_size
            )

        sigmas = calculate_sigmas(
            density_points
        )

        density_map = generate_density_map(
            self.density_size,
            self.density_size,
            density_points,
            sigmas
        )

        density_tensor = torch.from_numpy(
            density_map
        ).unsqueeze(0)

        # --------------------------------------------------
        # 6. Ground-truth count
        # --------------------------------------------------

        count = torch.tensor(
            float(len(points_scaled)),
            dtype=torch.float32
        )

        return {
            "image": image_tensor,
            "density": density_tensor,
            "count": count,
            "image_name": image_path.name
        }


def create_part_a_dataset(
    project_root,
    image_size=512,
    density_size=64
):

    project_root = Path(project_root)

    dataset_root = (
        project_root
        / "data"
        / "ShanghaiTech"
        / "part_A_final"
    )

    train_image_dir = (
        dataset_root
        / "train_data"
        / "images"
    )

    train_gt_dir = (
        dataset_root
        / "train_data"
        / "ground_truth"
    )

    test_image_dir = (
        dataset_root
        / "test_data"
        / "images"
    )

    test_gt_dir = (
        dataset_root
        / "test_data"
        / "ground_truth"
    )

    train_images = sorted(
        train_image_dir.glob("*.jpg")
    )

    test_images = sorted(
        test_image_dir.glob("*.jpg")
    )

    train_dataset = ShanghaiTechDataset(
        train_images,
        train_gt_dir,
        image_size,
        density_size
    )

    test_dataset = ShanghaiTechDataset(
        test_images,
        test_gt_dir,
        image_size,
        density_size
    )

    return train_dataset, test_dataset