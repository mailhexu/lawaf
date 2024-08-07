import numpy as np
from phonopy import load
from ase.io import write
from lawaf import NACPhonopyDownfolder, PhonopyDownfolder
import matplotlib.pyplot as plt

fname = "phonopy_params.yaml"
params = dict(
    method="scdmk",
    # method="projected",
    nwann=3,
    # selected_basis=[9, 10, 11],
    # nwann=15,
    # selected_basis=list(range(15)),
    anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
    use_proj=True,
    weight_func="unity",
    # weight_func="Fermi",
    weight_func_params=(-1, 1),
    kmesh=(6, 6, 6),
    gamma=True,
    kshift=(0.000, 0.000, 0.000),
    enhance_Amn=0,
)

downfolder = NACPhonopyDownfolder(
    phonopy_yaml=fname,
    mode="DM",
    params=params,
    nac_params={"method": "wang"},
    born_filename="BORN",
)
downfolder.set_parameters(**params)
for i in range(len(downfolder.kpts)):
    print(i, downfolder.kpts[i])
downfolder.downfold()
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
    npoints=100,
    unit_factor=15.6 * 33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True,
    knames=["$\\Gamma$", "X", "M", "R", "X", "$\\Gamma$", "R"],
    show=False,
    fix_LOTO=True,
    # plot_nonac=True,
)
plt.savefig("LWF_BTO.pdf")
plt.show()
