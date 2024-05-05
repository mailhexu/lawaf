import os
import json
import numpy as np
from lawaf.scdm.lwf import LWF
from minimulti.electron.basis2 import BasisSet, Basis
from lawaf.utils.symbol import symbol_number
from minimulti.electron.Hamiltonian import atoms_model
from ase.atoms import Atoms
import matplotlib.pyplot as plt


def read_basis(fname):
    """
    Read basis from json file.
    """
    bdict = dict()
    with open(fname) as myfile:
        c = json.load(myfile)
    atoms = Atoms(
        c["chemical_symbols"], scaled_positions=c["atom_xred"], cell=c["cell"]
    )
    orbs = c["Orb_names"]
    Efermi = c["Efermi"]
    bset = BasisSet()
    sdict = symbol_number(atoms)
    for orb in orbs:
        sn, label, spin = orb.split("|")
        label = label.split("Z")[0][1:]
        site = sdict[sn]
        bset.append(Basis(site=site, label=label, spin=spin, index=0))
    bset.set_atoms(atoms)
    return bset, Efermi


def read_wf(path, lwf_fname, json_fname="Downfold.json"):
    bset, Efermi = read_basis(os.path.join(path, json_fname))
    lwf = LWF.load_nc(os.path.join(path, lwf_fname))
    return lwf_to_model(bset, lwf, Efermi)


def lwf_to_model(bset, lwf, Efermi):
    atoms = bset.atoms
    bset = bset.add_spin_index()
    model = atoms_model(atoms=atoms, basis_set=bset, nspin=2)
    hop = {}
    for iR, R in enumerate(lwf.Rlist):
        R = tuple(R)
        hr = np.zeros(
            (
                lwf.nwann * 2,
                lwf.nwann * 2,
            ),
            dtype="complex",
        )
        hr[::2, ::2] = lwf.hoppings[tuple(R)] / 2
        hr[1::2, 1::2] = lwf.hoppings[tuple(R)] / 2
        hop[R] = hr

    en = np.zeros(lwf.nwann * 2, dtype=float)
    en[::2] = lwf.site_energies + Efermi
    en[1::2] = lwf.site_energies + Efermi
    model._hoppings = hop
    model._site_energies = en
    return model


def run(alpha, U, J):
    model = read_wf(
        path="DF",
        lwf_fname=f"Downfolded_hr_{alpha:.1f}.nc",
        json_fname=f"Downfold_{alpha:.1f}.json",
    )
    run_model(U, J)


def run_model(model: atoms_model, alpha, U, J, plot=False):
    model.set(nel=4)
    model.set_kmesh([4, 4, 4])
    model.set_Hubbard_U(Utype="Kanamori", Hubbard_dict={"V": {"U": U, "J": J}})
    model.scf_solve()
    model.save_result(
        pfname=f"Results_Kanamori/result_U{U:.2f}_J{J:.2f}_alpha{alpha:.2f}.pickle"
    )
    if plot:
        model.plot_band(
            kvectors=[
                [0, 0, 0],
                [0, 0.5, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0],
                [0, 0, 0],
                [0, 0.25, 0.5],
                [0.5, 0.25, 0.5],
                [0.5, 0, 0],
                [0, 0, 0],
                [0.0, 0.25, -0.5],
                [0.5, 0.25, -0.5],
            ],
            knames="GYCZGBDZGAE",
            shift_fermi=True,
        )
        figname = f"Results_Kanamori/result_U{U:.2f}_J{J:.2f}_alpha{alpha:.2f}.png"
        plt.savefig(figname)
        plt.show()
        plt.close()


if __name__=="__main__":
    run_model()
