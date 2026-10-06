import numpy as np
from scipy import linalg


def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition of a matrix.

    Factors x = P @ L @ U. The permutation P is what makes this work in floating
    point: partial pivoting moves the largest available entry onto the diagonal,
    so the multipliers stored in L stay bounded. Plain Gaussian elimination
    without it loses accuracy as soon as a pivot is small.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            The permutation matrix P, lower triangular matrix L, and upper triangular matrix U.
    """
    p, l, u = linalg.lu(np.asarray(x, dtype=float))
    return p, l, u


def qr_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition of a matrix.

    Factors x = Q @ R with Q orthonormal and R upper triangular.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray]: The orthogonal matrix Q and upper triangular matrix R.
    """
    q, r = np.linalg.qr(np.asarray(x, dtype=float))
    return q, r


def determinant(x: np.ndarray) -> np.ndarray:
    """
    Calculate the determinant of a matrix.

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The determinant of the matrix.
    """
    # Computed from the LU factors, not the cofactor expansion, so the cost is
    # O(n^3) rather than O(n!). Note the determinant says nothing useful about
    # how invertible a matrix is in practice - scale a 100x100 identity by 0.5
    # and the determinant underflows while the matrix stays perfectly conditioned
    return np.linalg.det(np.asarray(x, dtype=float))


def eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and right eigenvectors of a matrix.

    Column i of the returned eigenvector matrix is the unit-norm eigenvector for
    eigenvalue i, so that x @ v[:, i] == w[i] * v[:, i].

    Args:
        x (np.ndarray): The input matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eigenvalues and the right eigenvectors of the matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(x, dtype=float))
    return eigenvalues, eigenvectors


def svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition (SVD) of a matrix.

    Returns the three factors so that x == U @ diag(S) @ V for a square input -
    that is, the third factor is V^T (numpy's `Vh`), despite the name in the
    signature. S is the 1-D vector of singular values in descending order, not
    the diagonal matrix.

    Args:
        x (np.ndarray): The input matrix to decompose.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The matrices U, S, and V.
    """
    u, s, vh = np.linalg.svd(np.asarray(x, dtype=float))
    return u, s, vh
