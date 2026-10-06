import numpy as np

def negative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the negation of each element in the input vector or matrix.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with each element negated.
    """
    return np.negative(x)


def reverse_matrix(x: np.ndarray) -> np.ndarray:
    """
    Returns the input vector or matrix with the order of elements reversed.

    Args:
        x (np.ndarray): A vector (n*1) or matrix (n*n).

    Returns:
        np.ndarray: A matrix with the order of elements reversed.
    """
    # np.flip reverses along every axis, which for a matrix is the same as
    # reversing the flattened element order and reshaping back. "The order of
    # elements reversed" - while still doing the obvious thing to a vector
    return np.flip(np.asarray(x))


def affine_transform(
    x: np.ndarray, alpha_deg: float, scale: tuple[float, float], shear: tuple[float, float],
    translate: tuple[float, float],
) -> np.ndarray:
    """Compute affine transformation

    The four parts are composed as translate(shear(scale(rotate(p)))), i.e. the
    3x3 homogeneous matrix M = T @ Sh @ S @ R, and applied to every 2-D point in
    `x`. Affine composition is not commutative, so this order is part of the
    answer; a different convention gives different numbers for the same inputs.

    `x` holds 2-D points either as rows (shape (N, 2)) or as columns (shape
    (2, N)); the result keeps the layout it was given. A square (2, 2) input is
    read as two points stored row-wise.

    Args:
        x (np.ndarray): vector n*1 or matrix n*n.
        alpha_deg (float): rotation angle in deg.
        scale (tuple[float, float]): x, y scale factor.
        shear (tuple[float, float]): x, y shear factor.
        translate (tuple[float, float]): x, y translation factor.

    Returns:
        np.ndarray: transformed matrix.
    """
    points = np.asarray(x, dtype=float)

    # normalise the input to (N, 2) rows, remembering how to undo it
    if points.ndim == 1:
        if points.size != 2:
            raise ValueError(f"a 2-D point needs 2 coordinates, got {points.size}")
        rows, restore = points.reshape(1, 2), lambda r: r.reshape(2)
    elif points.ndim == 2 and points.shape[1] == 2:
        rows, restore = points, lambda r: r
    elif points.ndim == 2 and points.shape[0] == 2:
        rows, restore = points.T, lambda r: r.T
    else:
        raise ValueError(
            f"expected 2-D points as (N, 2) or (2, N), got shape {points.shape}"
        )

    # build the homogeneous 3x3 blocks
    alpha = np.radians(alpha_deg)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    scale_x, scale_y = scale
    shear_x, shear_y = shear
    translate_x, translate_y = translate

    rotation = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ])
    scaling = np.array([
        [scale_x, 0.0,     0.0],
        [0.0,     scale_y, 0.0],
        [0.0,     0.0,     1.0],
    ])
    shearing = np.array([
        [1.0,     shear_x, 0.0],
        [shear_y, 1.0,     0.0],
        [0.0,     0.0,     1.0],
    ])
    translation = np.array([
        [1.0, 0.0, translate_x],
        [0.0, 1.0, translate_y],
        [0.0, 0.0, 1.0],
    ])

    matrix = translation @ shearing @ scaling @ rotation

    # apply it
    # The homogeneous 1 in the third coordinate is what turns the translation
    # column into an actual shift; a plain 2x2 matrix cannot translate at all
    homogeneous = np.hstack([rows, np.ones((rows.shape[0], 1))])
    transformed = (matrix @ homogeneous.T).T[:, :2]

    return restore(transformed)
