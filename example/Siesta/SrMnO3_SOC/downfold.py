from lawaf.interfaces import SiestaDownfolder
import numpy as np


def main():
    params = dict(
        method="projected",
        kmesh=[5, 5, 5],
        # nwann=4,
        weight_func="Gauss",
        weight_func_params=(3, 3),
        use_proj=True,
        # selected_basis=None,
        # anchors={(0.0, 0.0, 0): [46, 47, 48, 49]},
        exclude_bands=[],
        # selected_orbdict={"Mn":["3dxy", "3dyz", "3dxz"]}
        # selected_orbdict={"Mn":["3dx2-y2", "3dz2"]}
        selected_orbdict={"Mn": ["3d"], "O": ["2p"]},
        # selected_orbdict={"Mn": ["3d"]},
        enhance_Amn=1,
    )

    downfolder = SiestaDownfolder(fdf_fname="siesta.fdf", params=params)
    downfolder.downfold()
    downfolder.plot_band_fitting(
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=[-6, 8],
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig="Downfolded_band.png",
        show=True,
    )


if __name__ == "__main__":
    main()
