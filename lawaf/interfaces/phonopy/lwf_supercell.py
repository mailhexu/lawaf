import copy
import os

import numpy as np
from ase.io import read
from scipy.sparse import csr_matrix, dok_matrix, load_npz, save_npz

# from pyDFTutils.ase_utils import vesta_view
# from lawaf.interface.phonon.lwf import LWF
from lawaf.lwf.lwf_supercell import (
    MyLWFSC as _MyLWFSC,
    build_lwf_lattice_mapping_matrix as _build_mapping,
    lwf_to_disp as _lwf_to_disp,
)
from lawaf.lwf.lwf import LWF
from lawaf.plot.mcif import write_mcif
from lawaf.utils.supercell import SupercellMaker




def write_lwf_cif(
    lwf=None,
    lwf_fname=None,
    sc_matrix=np.diag([2, 2, 2]),
    center=True,
    amp=1.0,
    prefix="LWF",
    list_lwf=None,
):
    if lwf is None:
        mylwf = LWF.load_nc(fname=lwf_fname)
    else:
        mylwf = lwf
    scmaker = SupercellMaker(sc_matrix=sc_matrix, center=center)
    mylwfsc = _MyLWFSC(mylwf, scmaker)
    # nwan=scmaker.ncell*3
    nlwf = mylwfsc.nlwf
    nlwf_sc = scmaker.ncell * nlwf
    if list_lwf is None:
        list_lwf = list(range(nlwf))
    elif isinstance(list_lwf, int):
        list_lwf = [list_lwf]

    atoms_lwfs = []
    disps = []
    for i in list_lwf:
        amps = np.zeros((nlwf_sc,))
        amps[i] = amp
        atoms, disp = mylwfsc.get_distorted_atoms(amps)
        atoms_lwfs.append(atoms)
        disps.append(disp)
        atoms.set_pbc(True)
        write_mcif(f"{prefix}_{i:04d}.cif", atoms, vectors=disp, factor=1)
    return atoms
