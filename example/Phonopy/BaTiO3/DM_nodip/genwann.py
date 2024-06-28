import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt


def gen_wann(name, method, use_proj, **kwargs):
    fname = "phonopy_params.yaml"
    params = dict(
        method="scdmk",
        nwann=3,  # selected_basis=[9, 10, 11],
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func_params=(800, 100.010),
        weight_func="unity",
        kmesh=(4, 4, 4),
        gamma=True,
        kshift=(0.000, 0.000, 0.000),
        # enhance_Amn=-2,
    )
    params.update(kwargs)
    downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM", params=params)
    lwf = downfolder.downfold()
    lwf.write_to_netcdf(f"lwf_{name}.nc")
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
    plt.savefig(f"{name}.pdf")
    plt.show()


def run_all():
    pass


run_all()
