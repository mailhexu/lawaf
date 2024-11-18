import numpy as np


def HR_to_k(HR, Rlist, kpts):
    # Hk[k,:,:] = sum_R (H[R] exp(i2pi k.R))
    phase = np.exp(2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    Hk = np.einsum("rlm, kr -> klm", HR, phase)
    return Hk


def Hk_to_R(Hk, Rlist, kpts, kweights, Rdeg=None):
    if Rdeg is None:
        Rdeg = np.ones(Rlist.shape[0], dtype=float)
    phase = np.exp(-2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    HR = np.einsum("klm, kr, k, r->rlm", Hk, phase, kweights, Rdeg)
    return HR


def k_to_R(kpts, Rlist, Mk, kweights=None, Rdeg=None):
    """
    Transform k-space wavefunctions to real space.
    params:
        kpts: k-points
        Rlist: list of R vectors
        Mk: matrix of shape [nkpt, n1, n2] in k-space.

    return:
        MR: matrix of shape [nR, n1, n2], the matrix in R-space.

    """
    Rlist = np.array(Rlist)
    if Rdeg is None:
        Rdeg = np.ones(Rlist.shape[0], dtype=float)
    nkpt, n1, n2 = Mk.shape
    if kweights is None:
        kweights = np.ones(nkpt, dtype=float) / nkpt
    # phase=np.exp(-2.0j*np.pi*np.tensordot(kpts, Rlist, axes=([1], [1])))
    # phase = np.exp(-2.0j * np.pi * np.einsum("kd, rd-> kr", kpts, Rlist))
    # MR = np.einsum("klm, kr, k, r -> rlm", Mk, phase, kweights, Rdeg)
    # return MR

    nkpt, n1, n2 = Mk.shape
    nR = Rlist.shape[0]
    MR = np.zeros((nR, n1, n2), dtype=complex)
    for iR, R in enumerate(Rlist):
        for ik in range(nkpt):
            MR[iR] += (
                Mk[ik]
                * np.exp(-2.0j * np.pi * np.dot(kpts[ik], R))
                * kweights[ik]
                * Rdeg[iR]
            )
    return MR


def R_to_k(kpts, Rlist, MR):
    """
    Transform real-space wavefunctions to k-space.
    params:
        kpts: k-points
        Rlist: list of R vectors
        MR: matrix of shape [nR, n1, n2] in R-space.

    return:
        Mk: matrix of shape [nkpt, n1, n2], the matrix in k-space.

    """
    # phase = np.exp(2.0*np.pi*1j*np.tensordot(kpts, Rlist, axes=([1], [1])))
    # phase = np.exp(2.0j * np.pi * np.einsum("kd, rd-> kr", kpts, Rlist))
    # Mk = np.einsum("rlm, kr -> klm", MR, phase)
    # return Mk

    nR, n1, n2 = MR.shape
    nkpt = kpts.shape[0]
    Mk = np.zeros((nkpt, n1, n2), dtype=complex)
    for iR, R in enumerate(Rlist):
        for ik in range(nkpt):
            Mk[ik] += MR[iR] * np.exp(2.0j * np.pi * np.dot(kpts[ik], R))
    return Mk


def R_to_onek(kpt, Rlist, MR):
    """
    Transform real-space wavefunctions to k-space.
    params:
        kpt: k-point
        Rlist: list of R vectors
        MR: matrix of shape [nR, n1, n2] in R-space.

    return:
        Mk: matrix of shape [n1, n2], the matrix in k-space.

    """
    # phase = np.exp(2.0j * np.pi * np.dot(Rlist, kpt))
    # Mk = np.einsum("rlm, r -> lm", MR, phase)
    # return Mk
    n1, n2 = MR.shape[1:]
    Mk = np.zeros((n1, n2), dtype=complex)
    for iR, R in enumerate(Rlist):
        Mk += MR[iR] * np.exp(2.0j * np.pi * np.dot(kpt, R))
    return Mk
