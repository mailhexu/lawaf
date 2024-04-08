from banddownfolder.scdm.lwf import LWF
import numpy as np
from minimulti.utils.supercell import SupercellMaker
from pyDFTutils.ase_utils import vesta_view
from netCDF4 import Dataset
from ase.io import read, write
from ase.units import Bohr
import copy
from scipy.sparse import coo_matrix
from ase import Atoms


def lwf_to_atoms(mylwf: LWF, scmaker, amplist, thr=0.01):
    patoms = mylwf.atoms
    scatoms = scmaker.sc_atoms(patoms)
    positions = scatoms.get_positions()
    nR, natom3, nlwf = mylwf.wannR.shape
    scnatom = len(scatoms)
    displacement = np.zeros_like(positions.flatten())
    #stds = np.zeros_like(positions.flatten())
    for (R, iwann, ampwann) in amplist:
        iwann = int(iwann) - 1
        for Rwann, iRwann in mylwf.Rdict.items():
            for j in range(natom3):
                clwf = mylwf.wannR[iRwann, j, iwann]
                if abs(clwf.real) > thr:
                    sc_j, sc_R = scmaker.sc_jR_to_scjR(j, Rwann, R, natom3)
                    amp = ampwann * clwf
                    displacement[sc_j] += amp.real
                    #stds[sc_j] += std.real
    sc_pos = positions + displacement.reshape((scnatom, 3))
    scatoms.set_positions(sc_pos)
    return scatoms, displacement


def lwf_to_atoms_batch(mylwf: LWF,
                       scmaker,
                       fname,
                       itimes,
                       out_prefix,
                       thr=0.01):
    amplist = load_amplist(fname)
    patoms = mylwf.atoms
    scatoms = scmaker.sc_atoms(patoms)
    scatoms.set_pbc(True)
    write(f'scatoms.cif', scatoms)
    write(f'scatoms.vasp', scatoms)
    positions = scatoms.get_positions()
    nR, natom3, nlwf = mylwf.wannR.shape
    scnatom = len(scatoms)
    npos = np.product(positions.shape)
    ntime = len(itimes)
    print(npos, ntime)
    displacement = np.zeros((ntime, npos), dtype=float)
    #stds = np.zeros_like(positions.flatten())

    for Rwann, iRwann in mylwf.Rdict.items():
        for j in range(natom3):
            for R, iwann, ampwann in zip(*amplist):
                iwann = int(iwann) - 1
                clwf = mylwf.wannR[iRwann, j, iwann]
                if abs(clwf.real) > thr:
                    sc_j, sc_R = scmaker.sc_jR_to_scjR(j, Rwann, R, natom3)
                    displacement[:, sc_j] += ampwann[itimes] * clwf.real
    for it, itime in enumerate(itimes):
        p = copy.deepcopy(positions)
        print(displacement[it])
        disp = displacement[it, :].reshape((scnatom, 3))
        sc_pos = p + disp
        d = copy.deepcopy(scatoms)
        d.set_positions(sc_pos)
        d.set_pbc(True)
        write(f'cifs/{out_prefix}_{itime:03d}.cif', d)
        np.save(f'cifs/{out_prefix}_{itime:03d}_disp.npy', disp)
    return scatoms  #, stds


def load_amplist(fname, itime=None):
    root = Dataset(fname, 'r')
    nlwf = root.dimensions['nlwf'].size
    Rlist = root.variables['lwf_rvec'][:].filled(np.nan)
    ilwf = root.variables['ilwf_prim'][:]
    amps = root.variables['lwf'][:][:, :] * Bohr  # itime, iwann
    root.close()
    return Rlist, ilwf, amps[itime, :]


def load_map(fname):
    root = Dataset(fname, 'r')
    ndisp = root.dimensions["natom3"].size
    nlwf = root.dimensions["nlwf"].size
    idisp = root.variables["lwf_latt_map_id_displacement"][:]
    ilwf = root.variables["lwf_latt_map_id_lwf"][:]
    vals = root.variables['lwf_latt_map_values'][:]
    root.close()
    mat_map = coo_matrix((vals, (idisp - 1, ilwf - 1)), shape=(ndisp, nlwf))
    return mat_map


def load_ref_atoms(fname):
    root = Dataset(fname, 'r')
    ndisp = root.dimensions["natom3"].size
    ref_cell = root.variables["ref_cell"][:] * Bohr
    ref_xcart = root.variables["ref_xcart"][:] * Bohr
    zion = root.variables["zion"][:]
    root.close()
    return ref_cell, ref_xcart, zion


def get_atoms(fname, itime=None):
    ref_cell, ref_xcart, zion = load_ref_atoms(fname)
    dmap = load_map(fname)
    amp = load_amplist(fname, itime)[2]
    disp = dmap @ amp
    natom = len(disp) // 3
    xcart = ref_xcart + disp.reshape(natom, 3)
    atoms = Atoms(numbers=zion, positions=xcart, cell=ref_cell)
    write_atoms_to_netcdf('sample.nc',atoms)
    return atoms


def write_atoms_to_netcdf(fname, atoms: Atoms, nd=5):
    root = Dataset(fname, 'w')
    natom = len(atoms)
    natom_id = root.createDimension(dimname='natom', size=natom)
    three_id = root.createDimension(dimname='three', size=3)

    cell_id = root.createVariable("cell",
                                  float, ('three', 'three'),
                                  zlib=True,
                                  least_significant_digit=nd)
    numbers_id = root.createVariable("numbers", int, ('natom', ))
    xcart_id = root.createVariable("xcart",
                                   float, ('natom', 'three'),
                                   zlib=True,
                                   least_significant_digit=nd)

    root.variables['cell'][:] = atoms.get_cell()
    root.variables['numbers'][:] = atoms.get_atomic_numbers()
    root.variables['xcart'][:] = atoms.get_positions()
    root.close()


def read_atoms_from_netcdf(fname):
    root = Dataset(fname, 'r')
    cell = root.variables['cell'][:]
    numbers = root.variables['numbers'][:]
    positions = root.variables['xcart'][:]
    return Atoms(numbers=numbers, positions=positions, cell=cell)


atoms=get_atoms(fname="./lwf.out_T0007_lwfhist.nc", itime=30)
vesta_view(atoms)


def get_atoms(fname='./lwf.out_lwfhist.nc',
              scmaker=SupercellMaker(np.diag([16, 16, 16])),
              out_prefix=None,
              itime=None):
    print(f"Getting atoms from file {fname}, itime:{itime}")
    amplist = load_amplist(fname, itime=itime)
    mylwf = LWF.load_nc(fname='./VO2_othermodes.nc')
    atoms = lwf_to_atoms(
        mylwf,
        scmaker=scmaker,
        amplist=amplist,
    )
    #amplist=[[(0, 0, 0), 0, 1],
    #         [(0, 0, 0), 1, 1]])
    atoms.set_pbc(True)
    print(f"Writting to file: {out_prefix}_{itime:03d}.cif")
    write(f'{out_prefix}_{itime:03d}.cif', atoms)
    return (fname, itime)
    #np.savetxt(out_prefix + '.std', stds)


def main():
    import concurrent.futures
    scmaker = SupercellMaker(np.diag([32, 32, 32]))
    itimes = list(range(1, 20, 1))
    mylwf = LWF.load_nc(fname='./VO2_othermodes.nc')

    with concurrent.futures.ProcessPoolExecutor(max_workers=13) as executor:
        for i in range(1, 21, 1):
            T = (i - 1) * 30
            fname = f"lwf.out_T{i:04d}_lwfhist.nc"
            #lwf_to_atoms_batch(mylwf, scmaker, fname, itimes, out_prefix= f'T{T}K', thr=0.01)
            futures = executor.submit(lwf_to_atoms_batch, mylwf, scmaker,
                                      fname, itimes, f'n{i}', 0.03)
        for x in concurrent.futures.as_completed(futures):
            print(x, result())


#load_map("lwf.out_T0061_lwfhist.nc")

#main()
