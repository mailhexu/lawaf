import numpy as np


def normalize():
    size1 = 4
    size2 = 3

    a = np.random.random((size1, size2))
    #a = a + a.T

    Q, R = np.linalg.qr(a, mode='reduced')
    print(f"{Q=}")
    print(f"{R=}")

    print(R.conj().T @ R)

    print(Q.conj().T@Q)

    U, E, VT = np.linalg.svd(a, full_matrices=False)
    A=U@VT

    print(A.conj().T@A)
    


normalize()
