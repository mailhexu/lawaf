import numpy as np
from numpy.linalg import eigh


def test():
    # random matrix1 which is hermitian
    A = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    A = A + A.T.conj()
    # random matrix2 which is hermitian
    B = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
    B = B + B.T.conj()

    # compute eigenvalues and eigenvectors
    evals1, evecs1 = eigh(A)
    evals2, evecs2 = eigh(B)

    # check if psi1\dagger psi

    P2 = evecs1.conj().T @ evecs2.conj().T
    C = P2 @ B @ P2.conj().T

    evals3, evecs3 = eigh(C)

    print(f"eigenvalues of B: {evals2}")
    print(f"eigenvalues of C: {evals3}")


test()
