import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt


def gen_wann(name, **kwargs):
    fname = "phonopy_params.yaml"
    params = dict(
        method="scdmk",
        nwann=3,  # selected_basis=[9, 10, 11],
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func_params=(0, 100.010),
        weight_func="unity",
        kmesh=(2, 2, 2),
        gamma=True,
        kshift=(0.000, 0.000, 0.000),
        # enhance_Amn=-2,
    )
    params.update(kwargs)
    downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM", params=params)
    lwf = downfolder.downfold()
    lwf.write_to_netcdf(f"results/lwf_{name}.nc")
    lwf.write_to_cif(
        sc_matrix=np.eye(3) * 3,
        list_lwf=[0, 1, 2],
        prefix=f"results/{name}",
        center=True,
    )
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
    plt.savefig(f"results/{name}.pdf")
    plt.show()


def run_all():
    # gen_wann("scdmk_unity_noproj", method="scdmk", use_proj=False, weight_func="unity")
    # gen_wann("scdmk_Fermi_noproj", method="scdmk", use_proj=False, weight_func="Fermi")
    gen_wann("scdmk_unity_proj", method="scdmk", use_proj=True, weight_func="unity")


run_all()
