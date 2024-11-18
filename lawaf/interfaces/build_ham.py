"""
Use Amn, kpt, and eig to calculate the Hamiltonian matrix
"""

import numpy as np
import wannier90io as wio
from ase import Atoms
from HamiltonIO.lawaf import LawafHamiltonian as EWF
from scipy.linalg import svd

from lawaf.mathutils.kR_convert import k_to_R
from lawaf.utils.kpoints import build_Rgrid_with_degeneracy


def othogonalize(Amn):
    #Amn[ 60:, :] = 0
    U, S, VT = svd(Amn, full_matrices=False)
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
    def __init__(self, Amn, kpts, eig, kweights=None, **kwargs):
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


unit_to_angstrom = {"bohr": 0.52917721067, "angstrom": 1.0}


def get_unit(d):
    unit = d.get("units", "angstrom")
    if unit is None:
        unit = "angstrom"
    return unit


def parse_win(text):
    w = wio.parse_win_raw(text)
    if "kpoints" in w:
        kpts = w["kpoints"]["kpoints"]
    else:
        kpts = None
    if "unit_cell_cart" in w:
        uc = w["unit_cell_cart"]
        unit = get_unit(uc)
        factor = unit_to_angstrom[unit.lower()]
        a1 = np.array(uc["a1"]) * factor
        a2 = np.array(uc["a2"]) * factor
        a3 = np.array(uc["a3"]) * factor
        cell = np.array([a1, a2, a3])
    else:
        cell = None

    species = []
    xred = None
    xcart = None
    if "atoms_frac" in w:
        xred = []
        atoms_list = w["atoms_frac"]["atoms"]
        for a in atoms_list:
            species.append(a["species"])
            xred.append(np.array(a["basis_vector"]))
        xred = np.array(xred)

    if "atoms_cart" in w:
        xcart = []
        unit = get_unit(w["atoms_cart"])
        factor = unit_to_angstrom[unit.lower()]
        atoms_list = w["atoms_cart"]["atoms"]
        for a in atoms_list:
            species.append(a["species"])
            xcart.append(np.array(a["basis_vector"]) * factor)
        xcart = np.array(xcart)
    return kpts, cell, species, xred, xcart


def load_w90_files(prefix="t14o_DS2_w90", **kwargs):
    amnfile = prefix + ".amn"
    eigfile = prefix + ".eig"
    winfile = prefix + ".win"
    with open(amnfile, "r") as f:
        amn = wio.read_amn(f)
        print(np.abs(amn[0, :, 0]))
    with open(eigfile, "r") as f:
        eig = wio.read_eig(f)
    with open(winfile, "r") as f:
        text = f.read()
    parsed_win = wio.parse_win_raw(text)
    print(parsed_win)
    kpts, cell, species, xred, xcart = parse_win(text)
    # print(f"cell: {cell}")
    # print(f"species: {species}")
    # print(f"xred: {xred}")
    # print(f"xcart: {xcart}")
    if xcart is not None:
        atoms = Atoms(symbols=species, positions=xcart, cell=cell)
    else:
        atoms = Atoms(symbols=species, scaled_positions=xred, cell=cell)
    obj = HamBuilder(amn, kpts, eig, **kwargs)
    return obj, atoms


if __name__ == "__main__":
    load_w90_files()
