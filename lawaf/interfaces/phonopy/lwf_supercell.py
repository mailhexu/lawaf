import copy
import os

import numpy as np
from ase.io import read
from scipy.sparse import csr_matrix, dok_matrix, load_npz, save_npz

# from pyDFTutils.ase_utils import vesta_view
# from lawaf.interface.phonon.lwf import LWF
from lawaf.plot.mcif import write_mcif
from lawaf.utils.supercell import SupercellMaker


def build_lwf_lattice_mapping_matrix(mylwf, scmaker):
    # prim_atoms = mylwf.atoms
    nR, natom3, nlwf = mylwf.wannR.shape
    # mapping matirx: natom3_sc * nlwf_sc
    natom3_sc = scmaker.ncell * natom3
    nlwf_sc = scmaker.ncell * nlwf
    # mapping_mat = np.zeros((natom3_sc, nlwf_sc), dtype=float)
    mapping_mat = dok_matrix((natom3_sc, nlwf_sc), dtype=float)
    print(natom3_sc, nlwf_sc)
    for iwann in range(nlwf):
        for icell, Rsc in enumerate(scmaker.sc_vec):
            iwann_sc = scmaker.sc_i_to_sci(i=iwann, ind_Rv=icell, n_basis=nlwf)
            for iRwann, Rwann in enumerate(mylwf.Rlist):
                for j in range(natom3):
                    val = np.real(mylwf.wann_disps[iRwann, j, iwann])
                    if abs(val) > 1e-4:
                        sc_j, sc_R = scmaker.sc_jR_to_scjR(j, Rwann, Rsc, natom3)
                        mapping_mat[sc_j, iwann_sc] = val
    return csr_matrix(mapping_mat)


def lwf_to_disp(mapping_mat, lwfamp):
    return mapping_mat @ lwfamp


class MyLWFSC:
    def __init__(self, lwf, scmaker, mapping_file=None, scatoms_file=None):
        self.lwf = lwf

        nR, self.natom3, self.nlwf = self.lwf.wannR.shape
        self.natom = self.natom3 // 3
        self.scmaker = scmaker
        self.natom_sc = self.natom * self.scmaker.ncell
        if mapping_file is not None and os.path.exists(mapping_file):
            self.mapping_mat = load_npz(mapping_file)
        else:
            self.mapping_mat = build_lwf_lattice_mapping_matrix(lwf, scmaker)
            if mapping_file is not None:
                save_npz(mapping_file, self.mapping_mat)
        self.prim_atoms = self.lwf.atoms
        if scatoms_file is not None and os.path.exists(scatoms_file):
            self.sc_atoms = read(scatoms_file)
        else:
            self.sc_atoms = scmaker.sc_atoms(self.prim_atoms)
            # write(scatoms_file, self.sc_atoms, vasp5=True)

    def get_distorted_atoms(self, amp):
        disp = (self.mapping_mat @ amp).reshape((self.natom_sc, 3))
        atoms = copy.deepcopy(self.sc_atoms)
        positions = atoms.get_positions() + disp
        atoms.set_positions(positions)
        # write('datoms.vasp', atoms, vasp5=True, sort=True)
        # write_atoms_to_netcdf('datoms.nc', atoms)
        return self.sc_atoms, disp


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
    mylwfsc = MyLWFSC(mylwf, scmaker)
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
