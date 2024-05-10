import numpy as np
from numpy.linalg import svd, eigh


def align(evecs):
    """
    Aligns the eigenvectors to the principal axes of the tensor.
    """
    # Get the principal axes of the tensor
    u, s, vh = svd(evecs)
    # Align the eigenvectors to the principal axes
    evecs = np.dot(evecs, u)
    return evecs


def test_align():
    d = np.load("Hk_short.npy")
    evals, evecs = eigh(d[21])
    evecs = align(evecs)
    print(evals)
    print(evecs[:, 0:3])


if __name__ == "__main__":
    test_align()
