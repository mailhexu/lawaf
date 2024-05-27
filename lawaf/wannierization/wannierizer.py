"""
Wannierizer Module: Getting the Unitary matrix Amnk.
"""
import copy
import numpy as np
from scipy.linalg import qr, svd
from scipy.special import erfc
from netCDF4 import Dataset
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from lawaf.lwf.lwf import LWF
from typing import Callable, Optional, Tuple, Union, List
from dataclasses import dataclass
from lawaf.params import WannierParams
from lawaf.mathutils.occupation_functions import occupation_func


@dataclass
class BasicWannierizer:
    """
    Basic Wannier function builder
    variables:
    """

    nwann: int = None
    weight_func: Callable = None
    exclude_bands: Optional[Tuple[int]] = None
    wfn_anchor: Optional[np.ndarray] = None
    ndim: int = 3
    nkpt: int = None
    nbaisis: int = None
    nband: int = None
    is_orthogonal: bool = False
    params: WannierParams = None

    evecs: np.ndarray = None
    evals: np.ndarray = None
    positions: np.ndarray = None
    kpts: np.ndarray = None
    kweights: np.ndarray = None

    # has_phase: bool = True
    # build_Rgrid: Optional[np.ndarray] = None
    # has_nac: bool = False
    # Hks: Optional[np.ndarray] = None
    # Hshorts: Optional[np.ndarray] = None
    # Hlongs: Optional[np.ndarray] = None
    # S: np.ndarray = None

    def __init__(
        self,
        params: WannierParams,
        evals,
        evecs,
        kpts,
        kweights,
        Sk=None,
        kmesh=None,
        k_offset=None,
    ):
        """
        Initialize the Wannierizer class from the parameters.
        """
        self.evecs = evecs
        self.evals = np.array(evals)
        self.kpts = np.array(kpts, dtype=float)
        self.kweights = np.array(kweights, dtype=float)

        self.ndim = self.kpts.shape[1]
        self.nkpt, self.nbasis, self.nband = np.shape(evecs)
        self.nwann = params.nwann
        self.kshift = params.kshift

        if Sk is None:
            self.is_orthogonal = True
        else:
            self.S = Sk
            self.is_orthogonal = False

        self.params = params
        # exclude bands
        if self.params.exclude_bands is None:
            self.params.exclude_bands = []
        self.ibands = tuple(
            [i for i in range(self.nband) if i not in self.params.exclude_bands]
        )
        self.nband = len(self.ibands)
        if kmesh is not None:
            self.kmesh = kmesh
        else:
            self.kmesh = self.params.kmesh
            # get_monkhorst_pack_size_and_offset(self.kpts)
        self.weight_func = params.weight_func


class Wannierizer(BasicWannierizer):
    """
    General Wannier function builder
    """

    def __init__(
        self,
        evals,
        evecs,
        kpts,
        kweights,
        params: WannierParams,
        Sk=None,
        kmesh=None,
        k_offset=None,
    ):
        super().__init__(
            params, evals, evecs, kpts, kweights, Sk=Sk, kmesh=kmesh, k_offset=k_offset
        )

        # kpts
        self.weight_func = self.params.weight_func

        if isinstance(self.params.weight_func, str):
            self.weight_func = occupation_func(
                self.params.weight_func, *(self.params.weight_func_params)
            )

        # Rgrid
        # self.Rgrid = Rgrid
        # self._prepare_Rlist()
        # self.nR = self.Rlist.shape[0]

        # calculate occupation functions
        self.occ = self.weight_func(self.evals[:, self.ibands])

        self.Amn = np.zeros((self.nkpt, self.nband, self.nwann), dtype=complex)

        self.wannk = np.zeros((self.nkpt, self.nbasis, self.nwann), dtype=complex)
        # self.wannR = np.zeros((self.nR, self.nbasis, self.nwann), dtype=complex)
        self.Hwann_k = np.zeros((self.nkpt, self.nwann, self.nwann), dtype=complex)
        # self.HwannR = np.zeros((self.nR, self.nwann, self.nwann), dtype=complex)

        self.set_params(params)

    def set_params(self, params):
        pass

    def get_wannier(self, Rlist=None):
        """
        Calculate Wannier functions
        """
        self.prepare()
        self.get_Amn()
        self.get_wannk_and_Hk()
        if Rlist is not None:
            lwf = self.k_to_R(Rlist=Rlist)
            # lwf.atoms = copy.deepcopy(self.atoms)
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
        return self.evecs[ikpt][:, self.ibands]
        # return self.psi[ikpt, :, self.ibands]  # A bug in numpy found!! The matrix get transposed.
        # Not a bug, but defined behavior (though confusing).

    def get_eval_k(self, ikpt):
        """
        return the eigen values with excluded bands removed
        """
        return self.evals[ikpt, self.ibands]

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

    def get_wannier_centers(self, wannR, Rlist, positions):
        nR = len(Rlist)
        wann_centers = np.zeros((self.nwann, 3), dtype=float)
        for iR, R in enumerate(Rlist):
            c = wannR[iR, :, :]
            wann_centers += (c.conj() * c).real @ positions + R[None, :]
            # wann_centers+=np.einsum('ij, ij, jk', c.conj())#(c.conj()*c).T.real@self.positions  + R[None, :]
        # print(f"Wannier Centers: {self.wann_centers}")
        return wann_centers

    def _assure_normalized(self, wannR):
        """
        make sure that all the Wannier functions are normalized
        # TODO: should we use overlap matrix for non-orthogonal basis?
        """
        for iwann in range(self.nwann):
            norm = np.trace(wannR[:, :, iwann].conj().T @ wannR[:, :, iwann])

    def k_to_R(self, Rlist):
        """
        Calculate Wannier function and Hamiltonian from K space to R space.
        """
        wannR = np.zeros((len(Rlist), self.nbasis, self.nwann), dtype=complex)
        HwannR = np.zeros((len(Rlist), self.nwann, self.nwann), dtype=complex)
        for iR, R in enumerate(Rlist):
            for ik, k in enumerate(self.kpts):
                phase = np.exp(-2j * np.pi * np.dot(R, k))
                HwannR[iR] += self.Hwann_k[ik, :, :] * phase * self.kweights[ik]
                wannR[iR] += self.wannk[ik, :, :] * phase * self.kweights[ik]
        self._assure_normalized(wannR)
        # wann_centers=self.get_wannier_centers(wannR, Rlist, positions)
        return LWF(
            wannR=wannR,
            HwannR=HwannR,
            Rlist=Rlist,
            cell=np.eye(3),
            # wann_centers=wann_centers,
            # atoms=copy.deepcopy(atoms),
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
