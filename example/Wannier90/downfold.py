from lawaf import W90Downfolder
import numpy as np


def main():
    # Read From Wannier90 output
    params = dict(
        method="scdmk",
        kmesh=(4, 4, 4),
        nwann=5,
        weight_func="Gauss",
        weight_func_params=(10.0, 3.0),
        selected_basis=None,
        # anchors={(0, 0, 0): (12,13)},
        # anchors={(0, 0, 0): (9, 10, 11)},
        use_proj=False,
    )

    model = W90Downfolder(
        folder="./SMO_wannier", prefix="abinito_w90_down", params=params
    )

    # Downfold the band structure.
    model.downfold()

    # Plot the band structure.
    model.plot_band_fitting(
        kvectors=np.array(
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        knames=["$\Gamma$", "X", "M", "$\Gamma$", "R"],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=None,
        fullband_color="blue",
        downfolded_band_color="green",
        marker="o",
        ax=None,
        savefig="Downfolded_band.png",
        show=True,
    )


if __name__ == "__main__":
    main()
