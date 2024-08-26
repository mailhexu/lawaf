"""
Use Amn, kpt, and eig to calculate the Hamiltonian matrix
"""

import numpy as np
import wannier90io as wio
from HamiltonIO.lawaf import LawafHamiltonian as EWF
from scipy.linalg import svd

from lawaf.mathutils.kR_convert import k_to_R
from lawaf.utils.kpoints import build_Rgrid_with_degeneracy


def othogonalize(Amn):
    U, S, VT = svd(Amn)
    return U @ VT


def Amn_to_hamk(Amn, kpts, eig, orthogonize=False):
    nkpt, nband, nwann = Amn.shape
    hamk = np.zeros((nkpt, nwann, nwann), dtype=complex)
    for ik, kpt in enumerate(kpts):
        if orthogonize:
            Ak = othogonalize(Amn[ik])
        else:
            Ak = Amn[ik]
        hamk[ik] = Ak.T.conj() @ np.diag(eig[ik]) @ Ak
    return hamk


class HamBuilder:
    def __init__(self, Amn, kpts, eig, kweights=None):
        self.Amn = Amn
        self.kpts = kpts
        self.eig = eig
        self.nkpt, self.nband, self.nwann = Amn.shape
        if kweights is None:
            self.kweights = np.ones(self.nkpt) / self.nkpt
        else:
            self.kweights = kweights

    def get_Hamiltonian(self, Rmesh=None, Rlist=None, Rdeg=None, orthogonize=False):
        if Rmesh is not None:
            Rlist, Rdeg = build_Rgrid_with_degeneracy(Rmesh)
        Hk = Amn_to_hamk(self.Amn, self.kpts, self.eig, orthogonize=orthogonize)
        HR = k_to_R(self.kpts, Rlist, Hk, kweights=self.kweights, Rdeg=Rdeg)
        return EWF(
            wannR=None,
            HwannR=HR,
            SwannR=None,
            Rlist=Rlist,
            Rdeg=Rdeg,
            atoms=None,
            wann_names=None,
            is_orthogonal=True,
        )


def parse_win(text):
    w = wio.parsed_win_raw(text)
    if "kpoints" in w:
        kpts = w["kpoints"]
    else:
        kpts = None
    if "cell_cart" in w:
        cell = w["cell_cart"]
    else:
        cell = None
    return kpts, cell


class WannHamBuilder(HamBuilder):
    @classmethod
    def load_from_files(cls, prefix="t14o_DS2_w90", **kwargs):
        amnfile = prefix + ".amn"
        eigfile = prefix + ".eig"
        winfile = prefix + ".win"
        with open(amnfile, "r") as f:
            amn = wio.read_amn(f)
        with open(eigfile, "r") as f:
            eig = wio.read_eig(f)
        with open(winfile, "r") as f:
            text = f.read()
        parsed_win = wio.parse_win_raw(text)
        kpts = parsed_win["kpoints"]
        return cls(amn, kpts, eig, **kwargs)


if __name__ == "__main__":
    wh = WannHamBuilder.load_from_files()
