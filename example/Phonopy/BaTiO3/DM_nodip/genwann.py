import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt

fname = "phonopy_params.yaml"
# phonon=load(fname, is_nac=True)
# print(phonon.nac_params)
# exit()
params = dict(
    method="scdmk",
    nwann=3,  # selected_basis=[9, 10, 11],
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    use_proj=True,
    weight_func_params=(0, 0.06),
    weight_func="unity",
    kmesh=(4, 4, 4),
    gamma=True,
    kshift=(0.000, 0.000, 0.000),
)
downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM")
downfolder.set_parameters(**params)
# phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
# downfolder=PhonopyDownfolder(force_sets_filename="FORCE_SETS",
#          #born_filename="./BORN",
#          unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3, mode="DM")
downfolder.downfold()
# write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax = downfolder.plot_band_fitting(
    kvectors=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0],
            [0.5, 0.5, 0.5],
        ]
    ),
    npoints=300,
    unit_factor=15.6 * 33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True,
    knames=["$\\Gamma$", "X", "M", "R", "X", "$\\Gamma$", "R"],
    show=False,
)
plt.savefig("LWF_BTO.pdf")
plt.show()
