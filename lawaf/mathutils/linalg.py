import numpy as np
from scipy.linalg import eigh


def Lowdin(S):
    """
    Calculate S^(-1/2).
    Which is used in lowind's symmetric orthonormalization.
    psi_prime = S^(-1/2) psi
    """
    eigval, eigvec = eigh(S)
    return eigvec @ np.diag(np.sqrt(1.0 / eigval)) @ (eigvec.T.conj())

# unit tests
def test_Lowdin():
    S = np.array([[1.0, 0.5], [0.5, 1.0]])
    S_half = Lowdin(S)
    assert np.allclose(S_half @ S_half, np.linalg.inv(S))
    assert np.allclose(S_half.T.conj(), S_half) 
