import numpy as np


def get_matrix(n: int, m: int) -> np.ndarray:
    """Create random matrix n * m.

    Args:
        n (int): number of rows.
        m (int): number of columns.

    Returns:
        np.ndarray: matrix n*m.
    """
    return np.random.default_rng().standard_normal((n, m))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrix addition.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: matrix sum.
    """
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Matrix multiplication by scalar.

    Args:
        x (np.ndarray): matrix.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied matrix.
    """
    return np.multiply(x, a)


def dot_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Matrices dot product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix or vector.

    Returns:
        np.ndarray: dot product.
    """
    # Matrix product, not the element-wise one see hadamard_product below
    return np.matmul(x, y)


def identity_matrix(dim: int) -> np.ndarray:
    """Create identity matrix with dimension `dim`. 

    Args:
        dim (int): matrix dimension.

    Returns:
        np.ndarray: identity matrix.
    """
    return np.eye(dim)


def matrix_inverse(x: np.ndarray) -> np.ndarray:
    """Compute inverse matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: inverse matrix.
    """
    x = np.asarray(x)

    if x.ndim == 2 and x.shape[0] == x.shape[1]:
        # A singular matrix raises LinAlgError rather than returning garbage
        return np.linalg.inv(x)

    # Non-square matrices have no inverse; the Moore-Penrose pseudo-inverse is
    # the closest thing that exists, and it agrees with inv() when one exists
    return np.linalg.pinv(x)


def matrix_transpose(x: np.ndarray) -> np.ndarray:
    """Compute transpose matrix.

    Args:
        x (np.ndarray): matrix.

    Returns:
        np.ndarray: transposed matrix.
    """
    return np.transpose(x)


def hadamard_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute hadamard product.

    Args:
        x (np.ndarray): 1st matrix.
        y (np.ndarray): 2nd matrix.

    Returns:
        np.ndarray: Hadamard product.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        # Without this check broadcasting would quietly turn a (3, 1) and a
        # (1, 3) into a 3x3 outer-product-shaped result, which is not Hadamard
        raise ValueError(
            f"Hadamard product needs equal shapes, got {x.shape} and {y.shape}"
        )

    return np.multiply(x, y)


def basis(x: np.ndarray) -> tuple[int]:
    """Compute matrix basis.

    Returns the indexes of a maximal set of linearly independent columns,
    scanning left to right - the same columns the pivots of the reduced row
    echelon form would point at.

    Args:
        x (np.ndarray): matrix.

    Returns:
        tuple[int]: indexes of basis columns.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))

    target_rank = np.linalg.matrix_rank(x)
    chosen: list[int] = []

    for col in range(x.shape[1]):
        if len(chosen) == target_rank:
            break
        candidate = chosen + [col]
        # Keep the column only if it raises the rank, i.e. if it is not already
        # a combination of the columns kept so far. matrix_rank decides this via
        # the SVD, so near-dependent columns are treated as dependent the
        # right call in floating point, where "exactly dependent" never happens
        if np.linalg.matrix_rank(x[:, candidate]) == len(candidate):
            chosen = candidate

    return tuple(chosen)


def norm(x: np.ndarray, order: int | float | str) -> float:
    """Matrix norm: Frobenius, Spectral or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 'fro', 2 or inf.

    Returns:
        float: vector norm
    """
    # atleast_2d so a 1-D input still gets a *matrix* norm: np.linalg.norm would
    # otherwise reject ord='fro' and read ord=2 as the Euclidean vector norm
    return float(np.linalg.norm(np.atleast_2d(np.asarray(x)), ord=order))
