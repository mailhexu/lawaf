import numpy as np

from lawaf.interfaces import SiestaDownfolder


def main():
    params = dict(
        method="projected",
        kmesh=[6, 6, 6],
        # nwann=4,
        weight_func="window",
        weight_func_params=(-8, 6.5, 0.001),
        use_proj=False,
        # selected_basis=None,
        # anchors={(0.0, 0.0, 0): [46, 47, 48, 49]},
        exclude_bands=[],
        # selected_orbdict={"Mn":["3dxy", "3dyz", "3dxz"]}
        # selected_orbdict={"Mn":["3dx2-y2", "3dz2"]}
        selected_orbdict={"Mn": ["3d"], "O": ["2p"]},
        # selected_orbdict={"Mn": ["3d"]},
        # nwann=28,
        enhance_Amn=0,
    )

    downfolder = SiestaDownfolder(fdf_fname="siesta.fdf", params=params)
    wann = downfolder.downfold()
    wann.save_pickle("wannier.pickle")
    downfolder.plot_band_fitting(
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=[-10, 8],
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig="Downfolded_band.png",
        show=True,
    )


if __name__ == "__main__":
    main()
