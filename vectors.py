from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.default_rng().standard_normal((dim, 1))


def get_sparse_vector(dim: int) -> sparse.coo_matrix:
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    rng = np.random.default_rng()

    # Keep roughly a third of the entries, but never produce an all-zero vector:
    # a "sparse vector" with no stored values is useless in every test below
    nnz = max(1, dim // 3)
    rows = rng.choice(dim, size=nnz, replace=False)
    values = rng.standard_normal(nnz)

    return sparse.coo_matrix(
        (values, (rows, np.zeros(nnz, dtype=int))), shape=(dim, 1)
    )


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return np.add(x, y)


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return np.multiply(x, a)


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
        raise ValueError(
            f"need one coefficient per vector, got {len(vectors)} vectors "
            f"and {len(coeffs)} coefficients"
        )
    if not vectors:
        raise ValueError("cannot take a linear combination of no vectors")

    # sum(c_i * v_i) in one pass; keeps the shape of the input vectors
    return sum(c * np.asarray(v) for v, c in zip(vectors, coeffs))


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    # np.vdot flattens its arguments, so this works for (n,), (n, 1) and (1, n)
    # alike which matters because get_vector returns a column
    return float(np.vdot(x, y))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    # Flatten first: np.linalg.norm switches to *matrix* norms for 2-D input, and
    # for a (1, n) row vector the matrix 1-norm is not the Manhattan norm
    return float(np.linalg.norm(np.asarray(x).ravel(), ord=order))


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return norm(np.subtract(x, y), order=2)


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Angle between two vectors, in degrees.

    Despite the name, this returns the angle itself and not its cosine:
    0 for parallel vectors, 90 for orthogonal ones, 180 for opposite ones.

    Args:
        x (np.ndarray): 1st vector, shape (n, 1).
        y (np.ndarray): 2nd vector, shape (n, 1).

    Returns:
        float: angle in degrees, in [0, 180].
    """
    norm_x, norm_y = norm(x, 2), norm(y, 2)
    if norm_x == 0.0 or norm_y == 0.0:
        raise ValueError("the angle with a zero vector is undefined")

    cos_alpha = dot_product(x, y) / (norm_x * norm_y)

    # Rounding can push an exactly parallel pair to 1 + 1e-16, and arccos of that
    # is nan. Clip so the result stays a real angle in [0, 180]
    return float(np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0))))


def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1st vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    # Compared against a tolerance scaled by the vector magnitudes: a dot product
    # of 1e-9 means orthogonal for unit vectors but not for vectors of norm 1e6
    scale = norm(x, 2) * norm(y, 2)
    return bool(abs(dot_product(x, y)) <= 1e-9 * max(scale, 1.0))


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    a = np.asarray(a)

    if a.ndim == 2 and a.shape[0] == a.shape[1]:
        # Square: solve exactly. A singular `a` raises LinAlgError on purpose
        # the system has no unique solution and silently returning one of the
        # infinitely many answers is exactly the error this course is about
        return np.linalg.solve(a, b)

    # Over- or under-determined: fall back to the least-squares solution of
    # minimum norm, which is what "solve" can still mean for a non-square system
    return np.linalg.lstsq(a, b, rcond=None)[0]
