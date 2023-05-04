from banddownfolder.scdm.downfolder import SislDownfolder
import numpy as np


def main():
    downfolder = SislDownfolder(
        folder='.', fdf_file='siesta.fdf', format="sparse", nbands=50)
    downfolder.downfold(
        method='scdmk',
        kmesh=[5, 5, 5],
        nwann=4,
        weight_func='window',
        mu=-1,
        sigma=7,
        use_proj=False,
        selected_basis=None,
        anchors={(.0, .0, 0): [46, 47, 48, 49]},
        exclude_bands=[],
        write_hr_nc="Downfolded_hr.nc",
        write_hr_txt="Downfolded_hr.txt",
    )

    downfolder.plot_band_fitting(
        kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0],
                           [.5, .5, .5]]),
        knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
        supercell_matrix=None,
        npoints=100,
        efermi=None,
        erange=[-6, 8],
        fullband_color='blue',
        downfolded_band_color='green',
        marker='o',
        ax=None,
        savefig='Downfolded_band.png',
        show=True)


if __name__ == "__main__":
    main()
