import numpy as np
from scipy.linalg import eigh, fractional_matrix_power, qr, svd


def matrix_power(A, n):
    return fractional_matrix_power(A, n)


def lowdin(H, S):
    Shalf = matrix_power(S, -0.5)
    H = Shalf @ H @ Shalf


def set_random_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)


def gen_random_Hamiltonian(norb, seed=None):
    H = np.random.rand(norb, norb) + 1j * np.random.rand(norb, norb)
    H = H + H.T.conj()
    return H


def gen_random_S(norb):
    S = np.random.rand(norb, norb) + 1j * np.random.rand(norb, norb)
    S = S + S.T.conj()
    S /= 5
    np.fill_diagonal(S, 1)
    return S


def gen_HS(norb, seed):
    set_random_seed(seed)
    H = gen_random_Hamiltonian(norb)
    S = gen_random_S(norb)
    return H, S


def fermi_function(evals, efermi, T=0.1):
    return 1 / (1 + np.exp((evals - efermi) / T))


def scdm(psiT, n):
    # selected columns density matrix
    Q, R, piv = qr(psiT, mode="full", pivoting=True)
    cols = piv[:n]
    return cols


def orth(A, S):
    U, _, VT = svd(A, full_matrices=False)
    return U @ VT
    # Shalf = matrix_power(S, -0.5)
    # return Shalf@A@Shalf


def Hwann(evals, evecs, H, S, n=5, nwann=3):
    if nwann == n:
        efermi = evals[nwann - 1] + 10
    else:
        efermi = (evals[nwann - 1] + evals[nwann]) / 2
    print(f"evals: {evals}")
    print(f"efermi: {efermi}")
    weights = fermi_function(evals, efermi)
    print(f"weights: {weights}")
    psiT = evecs.T.conj()
    print(f"rho: {evecs@ psiT @S *weights[:, None]}")
    psiT = psiT @ S * weights[:, None]
    cols = scdm(psiT, nwann)
    cols = np.sort(cols)
    print(f"cols: {cols}")
    A = psiT[:, cols]
    # A = orth(A, S)
    print(f"A: {A}")
    wann = evecs @ A
    Swann = wann.T.conj() @ S @ wann
    print(f"wann: {wann}")

    wann = evecs @ A
    Hwann2 = wann.T.conj() @ H @ wann
    Hwann = A.T.conj() @ np.diag(evals) @ A
    evals_wann, evecs_wann = eigh(Hwann, Swann)
    evals_wann2, evecs_wann2 = eigh(Hwann2, Swann)

    print(evals)
    print(evals_wann)
    print(evals_wann2)
    print(evecs_wann - evecs_wann2)


def test_Hwann():
    n = 5
    nwann = 5
    H, S = gen_HS(n, None)
    evals, evecs = eigh(H, S)
    Hwann(evals, evecs, H, S, n=n, nwann=nwann)


def test():
    n = 5
    H, S = gen_HS(n, None)
    evals, evecs = eigh(H, S)
    # 1. test if evecs are orthonormal
    r = evecs @ evecs.conj().T @ S
    r1 = evecs @ evecs.conj().T @ S
    print(r)
    print(r1)


if __name__ == "__main__":
    test_Hwann()
    # test()
