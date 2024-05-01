"""
Wannierizer Module: For building Wannier functions and Hamiltonians.
"""
import copy
import numpy as np
from scipy.linalg import qr, svd
from scipy.special import erfc
from netCDF4 import Dataset
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from lawaf.utils.kpoints import kmesh_to_R, build_Rgrid
from lawaf.lwf.lwf import LWF
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BasicWannierizer:
    """
    Basic Wannier function builder
    variables:
    """
    evals: np.ndarray = None
    wfn: np.ndarray = None
    positions: np.ndarray = None
    kpts: np.ndarray = None
    nwann: int = None
    weight_func: Callable = None
    exclude_bands: Optional[Tuple[int]] = None
    wfn_anchor: Optional[np.ndarray] = None
    ndim: int = 3
    nkpt: int = None
    nbaisis: int = None
    nband: int = None
    is_orthogonal: bool = False
    S: np.ndarray = None
    has_phase: bool = True
    build_Rgrid: Optional[np.ndarray] = None
    has_nac: bool = False
    Hks: Optional[np.ndarray] = None
    Hshorts: Optional[np.ndarray] = None
    Hlongs: Optional[np.ndarray] = None


class Wannierizer(BasicWannierizer):
    """
    General Wannier function builder
    """

    def __init__(
        self,
        evals,
        wfn,
        positions,
        kpts,
        nwann,
        weight_func,
        kshift=None,
        atoms=None,
        kweights=None,
        Sk=None,
        has_phase=True,
        Rgrid=None,
        exclude_bands=None,
        wfn_anchor=None,
    ):
        self.evals = np.array(evals)
        self.kpts = np.array(kpts, dtype=float)
        self.wfn_anchor = wfn_anchor
        self.ndim = self.kpts.shape[1]
        self.nkpt, self.nbasis, self.nband = np.shape(wfn)

        if Sk is None:
            self.is_orthogonal = True
        else:
            self.S = Sk
            self.is_orthogonal = False
        # exclude bands
        if exclude_bands is None:
            exclude_bands = []
        self.ibands = tuple([i for i in range(self.nband) if i not in exclude_bands])
        self.nband = len(self.ibands)
        self.nwann = nwann
        self.positions = positions
        # kpts
        self.nkpt = self.kpts.shape[0]
        self.kmesh, self.k_offset = get_monkhorst_pack_size_and_offset(self.kpts)
        if not kweights:
            self.kweights = np.ones(self.nkpt, dtype=float) / self.nkpt
        else:
            self.kweights = kweights
        self.weight_func = weight_func
        self.atoms = atoms
        # Rgrid
        self.Rgrid = Rgrid
        self._prepare_Rlist()
        self.nR = self.Rlist.shape[0]

        # calculate occupation functions
        self.occ = self.weight_func(self.evals[:, self.ibands])

        # remove e^ikr from wfn
        self.has_phase = has_phase
        if not has_phase:
            self.psi = wfn
        else:
            self._remove_phase(wfn)

        self.Amn = np.zeros((self.nkpt, self.nband, self.nwann), dtype=complex)

        self.wannk = np.zeros((self.nkpt, self.nbasis, self.nwann), dtype=complex)
        self.Hwann_k = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)

        self.HwannR = np.zeros((self.nR, self.nwann, self.nwann), dtype=complex)
        self.wannR = np.zeros((self.nR, self.nbasis, self.nwann), dtype=complex)

    def get_wannier(self):
        """
        Calculate Wannier functions
        """
        self.prepare()
        self.get_Amn()
        self.get_wannk_and_Hk()
        lwf = self.k_to_R()
        lwf.atoms = copy.deepcopy(self.atoms)
        return lwf

    def set_nac_Hks(self, Hks, Hshorts, Hlongs):
        """set  Hamiltonians including splited Hks, Hshorts and Hlongs."""
        self.has_nac = True
        self.Hks = Hks
        self.Hshorts = Hshorts
        self.Hlongs = Hlongs

    def set_nac_params(self, born, dielectic, factor):
        """set  Hamiltonians including splited Hks, Hshorts and Hlongs."""
        self.has_nac = True
        self.born = born
        self.dielectic = dielectic
        self.factor = factor

    def get_wannier_nac(self):
        """
        Calculate Wannier functions but using non-analytic correction.
        """
        self.prepare()
        self.get_Amn()
        self.get_wannk_and_Hk_nac()
        lwf = self.k_to_R()
        lwf.atoms = copy.deepcopy(self.atoms)
        lwf.set_born_from_full(self.born, self.dielectic, self.factor)
        return lwf

    def prepare(self):
        """
        Do some preparation for calculating Wannier functions.
        """
        pass

    def get_psi_k(self, ikpt):
        """
        return the psi for the ikpt'th kpoint. excluded bands are removed.
        Note: This could be overrided so that the psi.
        """
        return self.psi[ikpt][:, self.ibands]
        # return self.psi[ikpt, :, self.ibands]  # A bug in numpy found!! The matrix get transposed.
        # Not a bug, but defined behavior (though confusing).

    def get_eval_k(self, ikpt):
        """
        return the eigen values with excluded bands removed
        """
        return self.evals[ikpt, self.ibands]

    def _remove_phase_k(self, wfnk, k):
        # phase = np.exp(-2j * np.pi * np.einsum('j, kj->k', k, self.positions))
        # return wfnk[:, :] * phase[:, None]
        psi = np.zeros_like(wfnk)
        for ibasis in range(self.nbasis):
            phase = np.exp(-2j * np.pi * np.dot(k, self.positions[ibasis, :]))
            psi[ibasis, :] = wfnk[ibasis, :] * phase
        return psi

    def _remove_phase(self, wfn):
        self.psi = np.zeros_like(wfn)
        for ik, k in enumerate(self.kpts):
            self.psi[ik, :, :] = self._remove_phase_k(wfn[ik, :, :], k)

    def _prepare_Rlist(self):
        if self.Rgrid is None:
            self.Rlist = kmesh_to_R(self.kmesh)
        else:
            self.Rlist = build_Rgrid(self.Rgrid)

    def find_k(self, kpt):
        """
        Find the most close k point to kpt from the kpts list.
        TODO: use PBC.
        """
        kpt = np.array(kpt)
        ns = np.linalg.norm(self.kpts - kpt[None, :], axis=1)
        ik = np.argmin(ns)
        return ik

    def get_Amn_one_k(self, ik):
        """
        Calcualte Amn matrix for one k point
        """
        raise NotImplementedError("The get_Amn_one_k method is should be overrided.")

    def get_Amn(self):
        """
        Calculate all Amn matrix for all k.
        """
        for ik in range(self.nkpt):
            self.Amn[ik, :, :] = np.array(self.get_Amn_one_k(ik), dtype=complex)
        return self.Amn

    def get_wannk_and_Hk(self, shift=0.0):
        """
        calculate Wannier function and H in k-space.
        """
        for ik in range(self.nkpt):
            self.wannk[ik] = self.get_psi_k(ik) @ self.Amn[ik, :, :]
            h = (
                self.Amn[ik, :, :].T.conj()
                @ np.diag(self.get_eval_k(ik) + shift)
                @ self.Amn[ik, :, :]
            )
            self.Hwann_k[ik] = h
        return self.wannk, self.Hwann_k

    def get_wannk_and_Hk_nac(self):
        """
        calculate Wannier function and H in k-space
        but onlythe short range part of the Hk. Ham is the
        short range part of the Hamiltonian in the original basis.
        """
        print("Using short range part of DM to build Hamiltonian")
        for ik in range(self.nkpt):
            Ham = self.Hshorts[ik] + self.Hlongs[ik]
            self.wannk[ik] = self.get_psi_k(ik) @ self.Amn[ik, :, :]
            psik = self.get_psi_k(ik)
            psiA = psik @ self.Amn[ik, :, :]
            self.Hwann_k[ik] = psiA.T.conj() @ Ham @ psiA
        return self.Hwann_k

    def get_wannier_centers(self):
        self.wann_centers = np.zeros((self.nwann, 3), dtype=float)
        for iR, R in enumerate(self.Rlist):
            c = self.wannR[iR, :, :]
            self.wann_centers += (c.conj() * c).T.real @ self.positions + R[None, :]
            # self.wann_centers+=np.einsum('ij, ij, jk', c.conj())#(c.conj()*c).T.real@self.positions  + R[None, :]
        # print(f"Wannier Centers: {self.wann_centers}")

    def _assure_normalized(self):
        """
        make sure that all the Wannier functions are normalized
        # TODO: should we use overlap matrix for non-orthogonal basis?
        """
        for iwann in range(self.nwann):
            norm = np.trace(self.wannR[:, :, iwann].conj().T @ self.wannR[:, :, iwann])

    def k_to_R(self):
        """
        Calculate Wannier function and Hamiltonian from K space to R space.
        """
        for iR, R in enumerate(self.Rlist):
            for ik, k in enumerate(self.kpts):
                phase = np.exp(-2j * np.pi * np.dot(R, k))
                self.HwannR[iR] += self.Hwann_k[ik, :, :] * phase * self.kweights[ik]
                self.wannR[iR] += self.wannk[ik, :, :] * phase * self.kweights[ik]
        self._assure_normalized()
        self.get_wannier_centers()
        return LWF(
            self.wannR,
            self.HwannR,
            self.Rlist,
            cell=np.eye(3),
            wann_centers=self.wann_centers,
            atoms=copy.deepcopy(self.atoms),
        )

    def save_Amnk_nc(self, fname):
        """
        Save Amn matrices into a netcdf file.
        """
        root = Dataset(fname, "w")
        ndim = root.createDimension("ndim", self.ndim)
        nwann = root.createDimension("nwann", self.nwann)
        nkpt = root.createDimension("nkpt", self.nkpt)
        nband = root.createDimension("nband", self.nband)
        kpoints = root.createVariable("kpoints", float, dimensions=(ndim, nkpt))
        Amnk = root.createVariable("Amnk", float, dimensions=(nkpt, nband, nwann))
        kpoints[:] = self.kpts
        Amnk[:] = self.Amn
        root.close()

def occupation_func(ftype=None, mu=0.0, sigma=1.0):
    """
    Return a Weight function.
    """
    if ftype in [None, "unity"]:

        def func(x):
            return np.ones_like(x, dtype=float)

    elif ftype == "Fermi":

        def func(x):
            return 0.5 * erfc((x - mu) / sigma)

    elif ftype == "Gauss":

        def func(x):
            return np.exp(-1.0 * (x - mu) ** 2 / sigma**2)

    elif ftype == "window":

        def func(x):
            return 0.5 * erfc((x - mu) / 0.01) - 0.5 * erfc((x - sigma) / 0.01)

    elif ftype == "linear":

        def func(x):
            return x

    else:
        raise NotImplementedError("function type %s not implemented." % ftype)
    return func


def Amnk_to_Hk(Amn, psi, Hk0, kpts):
    """
    For a given Amn, psi, Hk0,
    """
    Hk_prim = []
    for ik, k in enumerate(kpts):
        wfn = psi[ik, :, :] @ Amn[ik, :, :]
        hk = wfn.T.conj() @ Hk0[ik, :, :] @ wfn
        Hk_prim.append(hk)
    return np.array(Hk_prim)


def Hk_to_Hreal(Hk, kpts, kweights, Rpts):
    nbasis = Hk.shape[1]
    nk = len(kpts)
    nR = len(Rpts)
    for iR, R in enumerate(Rpts):
        HR = np.zeros((nR, nbasis, nbasis), dtype=complex)
        for ik, k in enumerate(kpts):
            phase = np.exp(-2j * np.pi * np.dot(R, k))
            HR[iR] += Hk[ik] * phase * kweights[ik]
    return HR

