from banddownfolder.scdm.lwf import LWF
import numpy as np
from minimulti.utils.supercell import SupercellMaker
from pyDFTutils.ase_utils import vesta_view


def lwf_to_atoms(mylwf: LWF, scmat, amplist):
    patoms = mylwf.atoms
    scmaker = SupercellMaker(scmat)
    scatoms = scmaker.sc_atoms(patoms)
    positions = scatoms.get_positions()
    nR,  natom3, nlwf = mylwf.wannR.shape
    scnatom = len(scatoms)

    print(mylwf.wannR[mylwf.Rdict[(0,0,0)],:, 0].real)

    displacement = np.zeros_like(positions.flatten())
    for (R, iwann, ampwann) in amplist:
        for Rwann, iRwann in mylwf.Rdict.items():
            for j in range(natom3):
                sc_j, sc_R = scmaker.sc_jR_to_scjR(j, Rwann, R, natom3)
                #print(f"{mylwf.wannR[iRwann, iwann, j].real: .2f}")
                amp = ampwann * mylwf.wannR[iRwann,j, iwann]
                #print(f"{amp:.2f}")
                displacement[sc_j] += amp.real
    sc_pos = positions + displacement.reshape((scnatom, 3))
    #print(displacement.reshape((scnatom, 3))[:6])
    scatoms.set_positions(sc_pos)
    return scatoms
    #print(displacement)


def test():
    mylwf = LWF.load_nc(fname='./Downfolded_hr.nc')
    atoms = lwf_to_atoms(mylwf,
                         scmat=np.diag([5, 5, 5]),
                         amplist=[[(0, 0, 0), 0, 1],
                                  [(0, 0, 0), 1, 1]])
    vesta_view(atoms)


test()
