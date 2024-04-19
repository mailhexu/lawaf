import numpy as np
from ase.dft.kpoints import monkhorst_pack
from scipy.linalg import eigh
from lawaf.utils.kpoints import kmesh_to_R


def HR_to_k(HR, Rlist, kpts):
    # Hk[k,:,:] = sum_R (H[R] exp(i2pi k.R))
    phase = np.exp(2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    Hk = np.einsum("rlm, kr -> klm", HR, phase)
    return Hk


def Hk_to_R(Hk, Rlist, kpts, kweights):
    phase = np.exp(-2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    HR = np.einsum("klm, kr, k->rlm", Hk, phase, kweights)
    return HR


def modify_one_kpoint(evals, evecs, func):
    """
    return the Hamiltonian with same eigen vectors but modified eigen values.
    evals and evecs are the  original eigen values and vectors.
    return H: the new Hamiltonian
    """
    new_evals = func(evals)
    new_Hk = evecs @ np.diag(new_evals) @ evecs.T.conj()
    return new_Hk


def force_ASR_kspace(HR, Rlist, kmesh, maxrk=np.sqrt(3) * 0.40):
    """
    Force the acoustic at Gamma while preserve the zone boundary phonon frequency.

    """
    kpts = monkhorst_pack(kmesh)
    nkpt = len(kpts)
    kweights = np.ones(nkpt, dtype=float) / nkpt
    Rlist = kmesh_to_R(kmesh)
    Hks = HR_to_k(HR, Rlist, kpts)
    sumH = np.diag(np.sum(HR, axis=(0, 1)))
    for ik, k in enumerate(kpts):
        # TODO: Move k into first BZ
        factor = 1.0 - 1.0 / maxrk * np.linalg.norm(k)
        Hks[ik] -= factor * sumH
    new_HR = Hk_to_R(Hks, Rlist, kpts, kweights)
    return new_HR


def ftest(x):
    y = x
    y[-1] += 1
    return y


def test_modify_one_kpoint():
    H0 = np.random.random((3, 3))
    H0 = H0 + H0.conj().T
    evals, evecs = eigh(H0)
    print(evecs)

    print("evals: ", evals)
    new_H = modify_one_kpoint(evals, evecs, ftest)
    new_evals, new_evecs = eigh(new_H)
    print(new_evals)
    print(new_evecs)


class HamModifier:
    def __init__(self, HR, Rlist):
        self.HR = HR
        self.Rlist = Rlist

    def modify(self, func, kmesh, keepR=True):
        kpts = monkhorst_pack(kmesh)
        nkpt = len(kpts)
        kweights = np.ones(nkpt, dtype=float) / nkpt
        Rlist = kmesh_to_R(kmesh)
        if keepR and (not np.all(self.Rlist == Rlist)):
            raise ValueError("The kmesh given is not consistent with the Rlist")

        Hks = HR_to_k(self.HR, self.Rlist, kpts)
        for ik, k in enumerate(kpts):
            Hk = Hks[ik]
            evals, evecs = eigh(Hk)
            Hks[ik] = modify_one_kpoint(evals, evecs, func)

        new_HR = Hk_to_R(Hks, Rlist, kpts, kweights)
        return new_HR, Rlist


if __name__ == "__main__":
    test_modify_one_kpoint()
