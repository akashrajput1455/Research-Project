from pathlib import Path

import numpy as np
from scipy.io import loadmat
import h5py

from PIL import Image


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "ShanghaiTech"
)


# ============================================================
# Find ground-truth directory
# ============================================================

def find_gt_directory(parent):

    possible_names = [
        "ground_truth",
        "ground-truth"
    ]

    for name in possible_names:

        directory = parent / name

        if directory.exists():

            return directory

    return None


# ============================================================
# Load annotation points
# ============================================================

def load_points(mat_path):

    # --------------------------------------------------------
    # Try MATLAB v7.3 / HDF5 format
    # --------------------------------------------------------

    try:

        with h5py.File(
            mat_path,
            "r"
        ) as file:

            if "annPoints" in file:

                points = np.array(
                    file["annPoints"]
                ).T

                return points

    except Exception:

        pass


    # --------------------------------------------------------
    # Try normal MATLAB format
    # --------------------------------------------------------

    try:

        data = loadmat(
            mat_path
        )

        if "annPoints" in data:

            return np.asarray(
                data["annPoints"]
            )

        if "image_info" in data:

            image_info = data[
                "image_info"
            ]

            try:

                points = image_info[
                    0
                ][
                    0
                ][
                    0
                ][
                    0
                ][
                    0
                ]

                return np.asarray(
                    points
                )

            except Exception:

                pass

    except Exception:

        pass


    raise RuntimeError(
        f"Could not read annotation:\n{mat_path}"
    )


# ============================================================
# Analyze one dataset part
# ============================================================

def inspect_part(part_name):

    part_root = (
        DATASET_ROOT
        / part_name
    )

    print("\n")
    print("=" * 70)
    print(f"DATASET: {part_name}")
    print("=" * 70)


    if not part_root.exists():

        print(
            f"❌ Dataset not found:\n{part_root}"
        )

        return


    total_images = 0
    total_annotations = 0

    all_counts = []


    # ========================================================
    # Train + Test
    # ========================================================

    for split in [
        "train_data",
        "test_data"
    ]:

        split_root = (
            part_root
            / split
        )

        image_dir = (
            split_root
            / "images"
        )

        gt_dir = find_gt_directory(
            split_root
        )


        print("\n")
        print("-" * 70)
        print(f"{split}")
        print("-" * 70)


        # ----------------------------------------------------
        # Check directories
        # ----------------------------------------------------

        if not image_dir.exists():

            print(
                f"❌ Images directory missing:\n{image_dir}"
            )

            continue


        if gt_dir is None:

            print(
                f"❌ Ground-truth directory missing:\n"
                f"{split_root}"
            )

            continue


        print(
            "Images:",
            image_dir
        )

        print(
            "Ground truth:",
            gt_dir
        )


        # ----------------------------------------------------
        # Find files
        # ----------------------------------------------------

        image_paths = sorted(
            image_dir.glob("*.jpg")
        )

        gt_paths = sorted(
            gt_dir.glob("*.mat")
        )


        print(
            "\nNumber of images:",
            len(image_paths)
        )

        print(
            "Number of GT files:",
            len(gt_paths)
        )


        # ----------------------------------------------------
        # Check image/GT matching
        # ----------------------------------------------------

        image_names = {
            image.stem
            for image in image_paths
        }

        gt_names = {
            gt.stem.replace(
                "GT_",
                ""
            )
            for gt in gt_paths
        }


        missing_gt = (
            image_names
            -
            gt_names
        )

        extra_gt = (
            gt_names
            -
            image_names
        )


        if len(missing_gt) == 0:

            print(
                "✅ Every image has a GT file."
            )

        else:

            print(
                "⚠️ Missing GT files:",
                len(missing_gt)
            )

            for name in list(
                missing_gt
            )[:10]:

                print(
                    "   ",
                    name
                )


        if len(extra_gt) > 0:

            print(
                "⚠️ Extra GT files:",
                len(extra_gt)
            )


        # ----------------------------------------------------
        # Analyze annotations
        # ----------------------------------------------------

        split_counts = []


        for gt_path in gt_paths:

            try:

                points = load_points(
                    gt_path
                )

                count = len(
                    points
                )

                split_counts.append(
                    count
                )

                all_counts.append(
                    count
                )

            except Exception as error:

                print(
                    "\n⚠️ Error reading:",
                    gt_path.name
                )

                print(
                    error
                )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if len(split_counts) > 0:

            print(
                "\nCrowd statistics:"
            )

            print(
                "Minimum count:",
                min(split_counts)
            )

            print(
                "Maximum count:",
                max(split_counts)
            )

            print(
                "Average count:",
                round(
                    np.mean(
                        split_counts
                    ),
                    2
                )
            )

            print(
                "Median count:",
                round(
                    np.median(
                        split_counts
                    ),
                    2
                )
            )

        # ----------------------------------------------------
        # Inspect first image
        # ----------------------------------------------------

        if len(image_paths) > 0:

            first_image = image_paths[0]

            try:

                image = Image.open(
                    first_image
                )

                print(
                    "\nExample image:"
                )

                print(
                    "File:",
                    first_image.name
                )

                print(
                    "Size:",
                    image.size
                )

                print(
                    "Mode:",
                    image.mode
                )

            except Exception as error:

                print(
                    "Could not open example image:",
                    error
                )


        total_images += len(
            image_paths
        )

        total_annotations += len(
            gt_paths
        )


    # ========================================================
    # Overall statistics
    # ========================================================

    print("\n")
    print("=" * 70)
    print(f"SUMMARY: {part_name}")
    print("=" * 70)

    print(
        "Total images:",
        total_images
    )

    print(
        "Total GT files:",
        total_annotations
    )

    if len(all_counts) > 0:

        print(
            "Overall minimum count:",
            min(all_counts)
        )

        print(
            "Overall maximum count:",
            max(all_counts)
        )

        print(
            "Overall average count:",
            round(
                np.mean(all_counts),
                2
            )
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "ShanghaiTech Dataset Inspection"
    )

    print(
        "Dataset root:",
        DATASET_ROOT
    )


    # --------------------------------------------------------
    # Part A
    # --------------------------------------------------------

    inspect_part(
        "part_A_final"
    )


    # --------------------------------------------------------
    # Part B
    # --------------------------------------------------------

    inspect_part(
        "part_B_final"
    )


    print("\n")
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)