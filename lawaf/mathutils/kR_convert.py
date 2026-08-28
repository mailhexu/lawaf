import numpy as np


def HR_to_k(HR, Rlist, kpts, Rdeg=None):
    # Hk[k,:,:] = sum_R Rdeg[R] * H[R] exp(i2pi k.R)
    if Rdeg is None:
        Rdeg = np.ones(Rlist.shape[0], dtype=float)
    phase = np.exp(2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    Hk = np.einsum("rlm, kr, r->klm", HR, phase, Rdeg)
    return Hk


def Hk_to_R(Hk, Rlist, kpts, kweights):
    """Full (unweighted) DFT coefficients; Rdeg is NOT folded into the
    stored values -- it is metadata applied at every R-sum (see R_to_k)."""
    phase = np.exp(-2.0j * np.pi * np.tensordot(kpts, Rlist, axes=([1], [1])))
    HR = np.einsum("klm, kr, k->rlm", Hk, phase, kweights)
    return HR


def k_to_R(kpts, Rlist, Mk, kweights=None):
    """
    Transform k-space tensor to real space (full Fourier coefficients).

    MR[R,:,:] = sum_k w_k Mk[k,:,:] exp(-2 pi i k.R)

    Rdeg is NOT folded into the stored values; pass it to the R-space
    consumers (R_to_k, R_to_onek, HR_to_k) instead.
    """
    Rlist = np.array(Rlist)
    nkpt, n1, n2 = Mk.shape
    if kweights is None:
        kweights = np.ones(nkpt, dtype=float) / nkpt
    phase = np.exp(-2.0j * np.pi * np.einsum("kd, rd->kr", kpts, Rlist))
    MR = np.einsum("klm, kr, k -> rlm", Mk, phase, kweights)
    return MR


def R_to_k(kpts, Rlist, MR, Rdeg=None):
    """
    Transform real-space tensor to k space.

    Mk[k,:,:] = sum_R Rdeg[R] MR[R,:,:] exp(2 pi i k.R)

    Rdeg carries the R-grid degeneracy weights (e.g. 1/2 on the aliased
    boundary images of an even-mesh Wigner-Seitz grid); it defaults to 1.
    """
    if Rdeg is None:
        Rdeg = np.ones(Rlist.shape[0], dtype=float)
    nR, n1, n2 = MR.shape
    nkpt = kpts.shape[0]
    Mk = np.zeros((nkpt, n1, n2), dtype=complex)
    for iR, R in enumerate(Rlist):
        for ik in range(nkpt):
            Mk[ik] += MR[iR] * np.exp(2.0j * np.pi * np.dot(kpts[ik], R)) * Rdeg[iR]
    return Mk


def R_to_onek(kpt, Rlist, MR, Rdeg=None):
    """
    Transform real-space tensor to a single k point (weighted sum over R).

    Mk[:,:] = sum_R Rdeg[R] MR[R,:,:] exp(2 pi i k.R)

    See R_to_k for the Rdeg convention.
    """
    if Rdeg is None:
        Rdeg = np.ones(Rlist.shape[0], dtype=float)
    phase = np.exp(2.0j * np.pi * np.einsum("rd, d->r", Rlist, kpt)) * Rdeg
    return np.einsum("rlm, r->lm", MR, phase)
