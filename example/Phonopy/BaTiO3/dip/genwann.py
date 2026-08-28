import numpy as np
from phonopy import load
from ase.io import write
from lawaf import NACPhonopyDownfolder
from lawaf.plot import plot_band
import matplotlib.pyplot as plt

fname = 'phonopy_params.yaml'
#phonon=load(fname, is_nac=True)
#print(phonon.nac_params)
#exit()
params = dict(
    method='scdmk',
    nwann=3,  # selected_basis=[2,5],
    anchors={(.0, .0, .0): (0, 1, 2)},
    use_proj=False,
    weight_func='Gauss',
    weight_func_params=(-0.25, 9.4),
    kmesh=(2, 2, 2),
)
downfolder = NACPhonopyDownfolder(phonopy_yaml=fname, mode="IFC",
                                  nac_params={"method": "wang"},
                                  born_filename="BORN", params=params)
#phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
#downfolder=PhonopyDownfolder(force_sets_filename="FORCE_SETS",
#         #born_filename="./BORN",
#         unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3, mode="DM")
lwf = downfolder.downfold()
lwf.write_to_netcdf("Downfolded_hr.nc")
lwf.save_txt("Downfolded_hr.txt")

ax = downfolder.plot_band_fitting(
    kvectors=np.array([[0., 0., 0.],
                       [0.5, 0.0, 0.],
                       [0.5, 0.5, 0.0],
                       [0.5, 0.5, 0.5],
                       [0.5, 0.0, 0.0],
                       [0.0, 0.0, 0],
                       [0.5, 0.5, 0.5]
                       ]),
    npoints=100,
    unit_factor=15.6 * 33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True,
    knames=['$\\Gamma$', 'X', 'M', 'R', 'X', '$\\Gamma$', "R"],
    show=False,
    fix_LOTO=True,
)
plt.savefig('LWF_PTO.pdf')
plt.show()
