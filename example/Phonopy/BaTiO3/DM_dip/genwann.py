import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt

fname = "phonopy_params.yaml"
params = dict(
    method="projected",
    nwann=3,   
    selected_basis=[9, 10, 11],
    #anchors={(0.0, 0.0, 0.0): (0, 1, 14)},
    use_proj=True,
    weight_func_params=(0, 600),
    weight_func="unity",
    kmesh=(2,2,2),
    gamma=True,
    kshift=(0.000, 0.001, 0.002),
)
downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM")
downfolder.set_parameters(**params)
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
    npoints=30,
    unit_factor=15.6 * 33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True,
    knames=["$\\Gamma$", "X", "M", "R", "X", "$\\Gamma$", "R"],
    show=False,
)
plt.savefig("LWF_BTO.pdf")
plt.show()
