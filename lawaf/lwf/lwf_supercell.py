from lawaf.lwf.lwf import LWF
from lawaf.plot.mcif import write_mcif
import numpy as np
from minimulti.utils.supercell import SupercellMaker
from pyDFTutils.ase_utils import vesta_view
from netCDF4 import Dataset
from ase.io import read, write
from scipy.sparse import dok_matrix, csr_matrix, save_npz, load_npz
from ase.units import Bohr
from ase import Atoms
import copy
import os


def write_atoms_to_netcdf(fname, atoms: Atoms):
    root = Dataset(fname, "w")
    natom = len(atoms)
    natom_id = root.createDimension(dimname="natom", size=natom)
    three_id = root.createDimension(dimname="three", size=3)

    cell_id = root.createVariable("cell", float, ("three", "three"))
    numbers_id = root.createVariable("numbers", int, ("natom",))
    xcart_id = root.createVariable("xcart", float, ("natom", "three"))

    root.variables["cell"][:] = atoms.get_cell()
    root.variables["numbers"][:] = atoms.get_atomic_numbers()
    root.variables["xcart"][:] = atoms.get_positions()
    root.close()


def read_atoms_from_netcdf(fname):
    root = Dataset(fname, "r")
    cell = root.variables["cell"][:]
    numbers = root.variables["numbers"][:]
    positions = root.variables["xcart"][:]
    return Atoms(numbers=numbers, positions=positions, cell=cell)


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
            for Rwann, iRwann in mylwf.Rdict.items():
                for j in range(natom3):
                    val = np.real(mylwf.wannR[iRwann, j, iwann])
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
    listlwf=None,
):
    if lwf is None:
        mylwf = LWF.load_nc(fname=lwf_fname)
    scmaker = SupercellMaker(sc_matrix=sc_matrix, center=center)
    mylwfsc = MyLWFSC(mylwf, scmaker)
    # nwan=scmaker.ncell*3
    nlwf = mylwfsc.nlwf
    nlwf_sc = scmaker.ncell * nlwf
    if listlwf is None:
        listlwf = list(range(nlwf))
    elif isinstance(listlwf, int):
        listlwf = [listlwf]

    atoms_lwfs = []
    disps = []
    for i in listlwf:
        amps = np.zeros((nlwf_sc,))
        amps[i] = amp
        atoms, disp = mylwfsc.get_distorted_atoms(amps)
        atoms_lwfs.append(atoms)
        disps.append(disp)
        atoms.set_pbc(True)
        write_mcif(f"{prefix}_{i:04d}.cif", atoms, vectors=disp, factor=1)
    return atoms


# write_lwf_cif(lwf_fname="Downfolded_hr.nc", listlwf=[0,1,2], sc_matrix=np.diag([3,3,3]))


def lwf_to_atoms(mylwf: LWF, scmat, amplist):
    print(mylwf.get_wann_largest_basis)
    patoms = mylwf.atoms
    scmaker = SupercellMaker(scmat, center=True)
    scatoms = scmaker.sc_atoms(patoms)
    positions = scatoms.get_positions()
    print(positions)
    nR, natom3, nlwf = mylwf.wannR.shape
    scnatom = len(scatoms)

    displacement = np.zeros_like(positions.flatten())
    for R, iwann, ampwann in amplist:
        iwann = int(iwann) - 1
        for Rwann, iRwann in mylwf.Rdict.items():
            for j in range(natom3):
                sc_j, sc_R = scmaker.sc_jR_to_scjR(j, Rwann, R, natom3)
                amp = ampwann * mylwf.wannR[iRwann, j, iwann]
                displacement[sc_j] += amp.real
    sc_pos = positions + displacement.reshape((scnatom, 3))
    scatoms.set_positions(sc_pos)
    return scatoms


def load_amplist(fname):
    root = Dataset(fname, "r")
    nlwf = root.dimensions["nlwf"].size
    Rlist = root.variables["lwf_rvec"][:].filled(np.nan)
    ilwf = root.variables["ilwf_prim"][:]
    amps = list(root.variables["lwf"][:][-1, :] * Bohr)
    amplist = []
    for i in range(nlwf):
        amplist.append([tuple(Rlist[i]), ilwf[i], amps[i]])
    return amplist


def get_atoms(Q1, Q2, sc_mat=np.diag([1, 1, 6])):
    amplist = [[(0, 0, 0), 0, Q1], [(0, 0, 0), 1, Q2]]
    mylwf = LWF.load_nc(fname="./lwf.nc")
    atoms = lwf_to_atoms(
        mylwf,
        scmat=sc_mat,
        amplist=amplist,
    )
    atoms.set_pbc(True)
    # write(f'supercell_{Q1:.1f}_{Q2:.1f}.vasp', atoms, vasp5=True, sort=True)
    write(f"223/supercell_{Q1:.1f}_{Q2:.1f}.cif", atoms)
    # vesta_view(atoms)


# for Q1 in [-0.1, 0.0, 0.1]:
#    for Q2 in [-0.1, 0.0, 0.1]:
#        get_atoms(Q1, Q2)


def m(ix, iz):
    r = np.exp(2.0j * np.pi * np.dot([0.5, 0, 0.5], [ix, 0, iz]))
    print(r)
    return r


def gen_domain(size, interface=2, mo=-1):
    amplist = []
    amp = 0.20
    for ix in range(2):
        for iz in range(size):
            if iz < interface:
                amplist.append([(ix, 0, iz), 0, m(ix, iz) * amp * mo])
                amplist.append([(ix, 0, iz), 1, m(ix, iz) * amp * mo])
            else:
                amplist.append([(ix, 0, iz), 0, m(ix, iz) * amp])
                amplist.append([(ix, 0, iz), 1, m(ix, iz) * amp])

    mylwf = LWF.load_nc(fname="./lwf.nc")
    atoms = lwf_to_atoms(
        mylwf,
        scmat=np.diag([2, 1, size]),
        amplist=amplist,
    )
    atoms.set_pbc(True)
    # write(f'supercell_{Q1:.1f}_{Q2:.1f}.vasp', atoms, vasp5=True, sort=True)
    # write(f'antidomain.vasp', atoms, vasp5=True, sort=True)
    write("Rdomain.vasp", atoms, vasp5=True, sort=True)
    vesta_view(atoms)
