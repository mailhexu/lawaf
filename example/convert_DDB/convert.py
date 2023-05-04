from abipy.dfpt.converters import abinit_to_phonopy
from abipy.dfpt.anaddbnc import AnaddbNcFile
import numpy as np


def convert(fname, outfname):
    nc = AnaddbNcFile.from_file(fname)
    phonon = abinit_to_phonopy(nc, supercell_matrix=np.eye(3)*4)
    phonon.save(settings={'force_constants': True})


convert(fname="run_anaddb.nc", outfname='')
