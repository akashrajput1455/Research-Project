import numpy as np

import h5py
from scipy.io import loadmat
from scipy.spatial import cKDTree


# ============================================================
# Configuration
# ============================================================

KNN_K = 3
BETA = 0.3
FALLBACK_SIGMA = 15.0


# ============================================================
# Load annotation points
# ============================================================

def load_points_from_mat(mat_path):

    points = None

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

    except Exception:

        pass


    # --------------------------------------------------------
    # Try standard MATLAB format
    # --------------------------------------------------------

    if points is None:

        try:

            data = loadmat(
                mat_path
            )

            if "annPoints" in data:

                points = data[
                    "annPoints"
                ]

            elif "image_info" in data:

                image_info = data[
                    "image_info"
                ]

                try:

                    points = (
                        image_info[
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
                    )

                except Exception:

                    pass

        except Exception:

            pass


    # --------------------------------------------------------
    # Check result
    # --------------------------------------------------------

    if points is None:

        raise RuntimeError(
            f"Could not load annotations from:\n{mat_path}"
        )


    points = np.asarray(
        points,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Make sure shape is N x 2
    # --------------------------------------------------------

    if points.ndim != 2:

        raise RuntimeError(
            f"Unexpected annotation shape: {points.shape}"
        )


    if points.shape[1] != 2:

        if points.shape[0] == 2:

            points = points.T

        else:

            raise RuntimeError(
                f"Unexpected annotation shape: {points.shape}"
            )


    return points


# ============================================================
# Calculate geometry-adaptive sigma
# ============================================================

def calculate_sigmas(
    points,
    k=KNN_K,
    beta=BETA
):

    number_of_points = len(
        points
    )


    # --------------------------------------------------------
    # No people
    # --------------------------------------------------------

    if number_of_points == 0:

        return np.array(
            [],
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Only one person
    # --------------------------------------------------------

    if number_of_points == 1:

        return np.array(
            [FALLBACK_SIGMA],
            dtype=np.float32
        )


    # --------------------------------------------------------
    # Find nearest neighbours
    # --------------------------------------------------------

    tree = cKDTree(
        points
    )


    effective_k = min(
        k + 1,
        number_of_points
    )


    distances, _ = tree.query(
        points,
        k=effective_k
    )


    sigmas = []


    for distance in distances:

        # First distance is distance to itself
        neighbour_distances = distance[1:]


        if len(
            neighbour_distances
        ) == 0:

            sigma = FALLBACK_SIGMA

        else:

            mean_distance = np.mean(
                neighbour_distances
            )

            sigma = (
                beta
                *
                mean_distance
            )


            # Avoid extremely small kernels
            sigma = max(
                sigma,
                1.0
            )


        sigmas.append(
            sigma
        )


    return np.asarray(
        sigmas,
        dtype=np.float32
    )


# ============================================================
# Generate density map
# ============================================================

def generate_density_map(
    image_height,
    image_width,
    points,
    sigmas
):

    density_map = np.zeros(
        (
            image_height,
            image_width
        ),
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Generate one Gaussian for each person
    # --------------------------------------------------------

    for point, sigma in zip(
        points,
        sigmas
    ):

        x, y = point


        x = int(
            round(x)
        )
        
        y = int(
            round(y)
        )
        
        
        # ----------------------------------------------------
        # Keep points inside density-map boundaries
        # ----------------------------------------------------
        
        x = min(
            max(x, 0),
            image_width - 1
        )
        
        y = min(
            max(y, 0),
            image_height - 1
        )


        if sigma <= 0:

            continue


        # ----------------------------------------------------
        # Gaussian radius
        # ----------------------------------------------------

        radius = max(
            1,
            int(
                3 * sigma
            )
        )


        x_start = max(
            0,
            x - radius
        )

        x_end = min(
            image_width - 1,
            x + radius
        )


        y_start = max(
            0,
            y - radius
        )

        y_end = min(
            image_height - 1,
            y + radius
        )


        # ----------------------------------------------------
        # Create coordinate grid
        # ----------------------------------------------------

        xs = np.arange(
            x_start,
            x_end + 1
        )

        ys = np.arange(
            y_start,
            y_end + 1
        )


        xx, yy = np.meshgrid(
            xs,
            ys
        )


        # ----------------------------------------------------
        # Gaussian function
        # ----------------------------------------------------

        gaussian = np.exp(
            -(
                (
                    (xx - x) ** 2
                    +
                    (yy - y) ** 2
                )
                /
                (
                    2 * sigma ** 2
                )
            )
        )


        # ----------------------------------------------------
        # Normalize Gaussian
        # ----------------------------------------------------

        gaussian_sum = gaussian.sum()


        if gaussian_sum > 0:

            gaussian /= gaussian_sum


        # ----------------------------------------------------
        # Add Gaussian to density map
        # ----------------------------------------------------

        density_map[
            y_start:y_end + 1,
            x_start:x_end + 1
        ] += gaussian.astype(
            np.float32
        )


    return density_map