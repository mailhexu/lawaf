import numpy as np
from phonopy import load
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt


def gen_wann(name, **kwargs):
    fname = "phonopy_params.yaml"
    params = dict(
        method="scdmk",
        nwann=3,
        # selected_basis=[9, 10, 11],
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func_params=(0, 100.010),
        weight_func="unity",
        kmesh=(4, 4, 4),
        gamma=True,
        kshift=(0.000, 0.000, 0.000),
        # enhance_Amn=-2,
    )
    params.update(kwargs)

    downfolder = PhonopyDownfolder(phonopy_yaml=fname, mode="DM", params=params)
    lwf = downfolder.downfold()
    lwf.write_to_netcdf(f"results/lwf_{name}.nc")
    kmesh = params["kmesh"]
    lwf.write_to_cif(
        sc_matrix=np.diag(np.array(kmesh) + 1),
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
        ylabel="Frequency ($cm^{-1}$)",
        evals_to_freq=True,
        knames=["$\\Gamma$", "X", "M", "R", "X", "$\\Gamma$", "R"],
        show=False,
    )
    plt.savefig(f"results/{name}.pdf")
    plt.show()


def run_all_scdmk():
    # gen_wann("scdmk_unity_noproj_k222", method="scdmk", use_proj=False, weight_func="unity", kmesh=(2, 2, 2))
    # gen_wann("scdmk_Fermi0_noproj_k222", method="scdmk", use_proj=False, weight_func="Fermi", weight_func_params=(0, 50.0), kmesh=(2, 2, 2))
    # gen_wann("scdmk_Fermi200_noproj_k222", method="scdmk", use_proj=False, weight_func="Fermi", weight_func_params=(200, 50.0), kmesh=(2, 2, 2))
    # gen_wann("scdmk_Fermi400_noproj_k222", method="scdmk", use_proj=False, weight_func="Fermi", weight_func_params=(400, 50.0), kmesh=(2, 2, 2))
    # gen_wann("scdmk_unity_noproj_k444", method="scdmk", use_proj=False, weight_func="unity", kmesh=(4, 4, 4))

    # gen_wann("scdmk_unity_proj_k222", method="scdmk", use_proj=True, weight_func="unity", kmesh=(2, 2, 2))
    # gen_wann("scdmk_Fermi0_proj_k222", method="scdmk", use_proj=True, weight_func="Fermi", weight_func_params=(0, 50.0), kmesh=(2, 2, 2))
    # gen_wann("scdmk_unity_proj_k444", method="scdmk", use_proj=True, weight_func="unity",kmesh=(4, 4, 4))

    # gen_wann("scdmk_Fermi0_proj_k444", method="scdmk", use_proj=True, weight_func="Fermi",kmesh=(4, 4, 4), weight_func_params=(-0, 250.0))
    # gen_wann("scdmk_Fermi0_proj_k444", method="scdmk", use_proj=True, weight_func="Fermi",kmesh=(4, 4, 4), weight_func_params=(-0, 50.0))
    gen_wann(
        "scdmk_unity_proj_k889",
        method="scdmk",
        use_proj=True,
        weight_func="unity",
        kmesh=(6, 6, 6),
        weight_func_params=(-0, 50.0),
    )
    # gen_wann("scdmk_Fermi0_proj_k666", method="scdmk", use_proj=True, weight_func="Fermi",kmesh=(6,6,6), weight_func_params=(-0, 50.0))
    # gen_wann("scdmk_unity_proj_k666", method="scdmk", use_proj=True, weight_func="unity",kmesh=(6, 6, 6), weight_func_params=(-0, 250.0))

    pass


def run_all_pwf():
    # gen_wann("pwf_unity_atom_k222", method="projected", use_proj=False,  weight_func="unity", kmesh=(2, 2, 2), selected_basis=[9, 10, 11])
    # gen_wann("pwf_unity_mode_k222", method="projected", use_proj=False,  weight_func="unity", kmesh=(2, 2, 2))
    # gen_wann("pwf_Fermi0_mode_k222", method="projected", use_proj=False,  weight_func="Fermi", kmesh=(2, 2, 2), weight_func_params=(0, 50.0))
    # gen_wann("pwf_unity_mode_k444", method="projected", use_proj=False,  weight_func="unity", kmesh=(4, 4, 4), weight_func_params=(0, 50.0))
    gen_wann(
        "pwf_Fermi0_mode_k444",
        method="projected",
        use_proj=False,
        weight_func="Fermi",
        kmesh=(4, 4, 4),
        weight_func_params=(0, 50.0),
    )
    gen_wann(
        "pwf_Fermi200_mode_k444",
        method="projected",
        use_proj=False,
        weight_func="Fermi",
        kmesh=(4, 4, 4),
        weight_func_params=(200, 50.0),
    )
    gen_wann(
        "pwf_Fermi500_mode_k444",
        method="projected",
        use_proj=False,
        weight_func="Fermi",
        kmesh=(4, 4, 4),
        weight_func_params=(500, 50.0),
    )
    # gen_wann("pwf_Fermi500_mode_k666", method="projected", use_proj=False,  weight_func="Fermi", kmesh=(6, 6, 6), weight_func_params=(500, 50.0))
    # gen_wann("pwf_Fermi500_mode_k888", method="projected", use_proj=False,  weight_func="Fermi", kmesh=(8, 8, 8), weight_func_params=(500, 50.0))
    # gen_wann("pwf_Fermi0_mode_k888", method="projected", use_proj=False,  weight_func="Fermi", kmesh=(8, 8, 8), weight_func_params=(0, 50.0))
    pass


def run_all_pwf_atom_projector():
    # gen_wann("pwf_unity_mode_k444_atom", method="projected", use_proj=False,  weight_func="unity", kmesh=(4, 4, 4), weight_func_params=(0, 50.0),selected_basis=[9, 10, 11], anchors=None)
    # gen_wann("pwf_Fermi0_mode_k444_atom", method="projected", use_proj=False,  weight_func="Fermi", kmesh=(4, 4, 4), weight_func_params=(0, 50.0),selected_basis=[9, 10, 11], anchors=None)
    pass


# run_all_scdmk()
run_all_pwf()
# run_all_pwf_atom_projector()
