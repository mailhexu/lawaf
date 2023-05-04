from banddownfolder import W90Downfolder
import numpy as np

def main():
    # Read From Wannier90 output
    model = W90Downfolder(folder='./SMO_wannier',
                          prefix='abinito_w90_down')

    # Downfold the band structure.
    model.downfold(
        method='scdmk',
        kmesh=(3, 3, 3),
        nwann=2,
        weight_func='Gauss',
        mu=10.0,
        sigma=3.0,
        selected_basis=None,
        anchors={(0, 0, 0): (12,13)},
        use_proj=True,
        write_hr_nc='Downfolded_hr.nc',
        write_hr_txt='Downfolded_hr.txt')

    # Plot the band structure.
    model.plot_band_fitting(
                          kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                             [0.5, 0.5, 0], [0, 0, 0],
                                             [.5, .5, .5]]),
                          knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                          supercell_matrix=None,
                          npoints=100,
                          efermi=None,
                          erange=None,
                          fullband_color='blue',
                          downfolded_band_color='green',
                          marker='o',
                          ax=None,
                          savefig='Downfolded_band.png',
                          show=True)

if __name__ == "__main__":
    main()


