import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt

fname = 'phonopy_params.yaml'
#phonon=load(fname, is_nac=True)
# print(phonon.nac_params)
# exit()
downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="IFC")
#phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
# downfolder=PhonopyDownfolder(force_sets_filename="FORCE_SETS",
#          #born_filename="./BORN",
#          unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3, mode="DM")
downfolder.downfold(method='projected', nwann=3,  # selected_basis=[2,5],
                    anchors={(.0, .0, .0): (0, 1, 2)},
                    use_proj=False, mu=-10, sigma=5.4, weight_func='Gauss', kmesh=(4,4,4), gamma=True)
write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax = downfolder.plot_band_fitting(kvectors=np.array([[0., 0., 0.],
                                                     [0.5, 0.0, 0.],
                                                     [0.5, 0.5, 0.0],
                                                     [0.5, 0.5, 0.5],
                                                     [0.5, 0.0, 0.0],
                                                     [0.0, 0.0, 0],
                                                     [0.5, 0.5, 0.5]
                                                     ]), npoints=80,
                                  knames=['$\\Gamma$', 'X', 'M', 'R', 'X', '$\\Gamma$', "R"], show=False)
plt.savefig('LWF_PTO.pdf')
plt.show()
