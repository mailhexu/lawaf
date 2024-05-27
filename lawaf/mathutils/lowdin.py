import numpy as np


def lowdin_downfold(M, ind_As, e):
    """
    Downfold the Hamiltonian matrix M from the full basis to the active space basis.
    """
    ind_Bs = np.setdiff1d(np.arange(M.shape[0]), ind_As)
    nB = len(ind_Bs)
    M_A = M[ind_As][:, ind_As]
    M_B = M[ind_Bs][:, ind_Bs]
    M_AB = M[ind_As][:, ind_Bs]
    M_BA = M[ind_Bs][:, ind_As]
    # test if M_AM and M_BM are conjugate transpose of each other
    assert np.allclose(M_A.conj().T, M_A)
    assert np.allclose(M_B.conj().T, M_B)
    assert np.allclose(M_AB.conj().T, M_BA)

    evals, evecs = np.linalg.eigh(M)
    evals_A = evals[ind_As]
    evals_B = evals[ind_Bs]
    # evals_B = np.linalg.eigvalsh(M_B)
    # evals_B=[e]
    Meff_A = M_A + M_AB @ np.linalg.inv(np.eye(nB) * evals_A[1] - M_B) @ M_BA
    # Meff_A = M_A + M_AB @ np.linalg.inv(np.diag(evals_B)-M_B) @ M_BA
    return Meff_A


def random_real_hamiltonian(n, symmetric=True):
    """
    Generate a random real Hamiltonian matrix of size n x n.
    param n: int
        Size of the Hamiltonian matrix.
    param symmetric: bool
        If True, the Hamiltonian matrix is symmetric.
    """
    M = np.random.rand(n, n)
    if symmetric:
        M = M + M.T
    return M


def random_unitary(n):
    """
    Generate a random unitary matrix of size n x n.
    param n: int
        Size of the unitary matrix.
    """
    M = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    Q, R = np.linalg.qr(M)
    return Q


def test_random_unitary():
    n = 5
    U = random_unitary(n)
    assert np.allclose(np.dot(U.conj().T, U), np.eye(n))


def random_hamiltonian(n, real=True, hermitian=True):
    """
    Generate a random Hamiltonian matrix of size n x n.
    param n: int
        Size of the Hamiltonian matrix.
    param hermitian: bool
        If True, the Hamiltonian matrix is Hermitian.
    """
    M = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    if hermitian:
        M = (M + M.conj().T) / 2
    return M


def test_lowdin():
    n = 9
    M = random_hamiltonian(n) * 8
    diag = np.arange(n)
    # set the dianogal elements to be diag.
    M[diag, diag] = diag

    ind_As = np.arange(n // 2)
    M_A = lowdin_downfold(M, ind_As, -3)
    evals_M = np.linalg.eigvalsh(M)
    evals_MA = np.linalg.eigvalsh(M_A)
    print(evals_M)
    print(evals_MA)


if __name__ == "__main__":
    test_random_unitary()

    test_lowdin()
