import numpy as np
from ase.io import write
from lawaf import PhonopyDownfolder
import matplotlib.pyplot as plt

fname = 'phonopy_params.yaml'
params = dict(
    method='scdmk',
    nwann=3,  # selected_basis=[2,5],
    anchors={(.0, .0, .0): (0, 1, 2)},
    use_proj=True,
    weight_func='Gauss',
    weight_func_params=(-0.25, 9.4),
    kmesh=(3, 3, 3),
)
downfolder = PhonopyDownfolder(phonopy_yaml=fname, params=params)
downfolder.downfold(write_hr_nc="Downfolded_hr.nc", write_hr_txt="Downfolded_hr.txt")
write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax = downfolder.plot_band_fitting(kvectors=np.array([[0., 0., 0.],
                                                     [0.5, 0.0, 0.],
                                                     [0.5, 0.5, 0.0],
                                                     [0.5, 0.5, 0.5],
                                                     [0.5, 0.0, 0.0],
                                                     [0.0, 0.0, 0],
                                                     [0.5, 0.5, 0.5]
                                                     ]),
                                  knames=['$\\Gamma$', 'X', 'M', 'R', 'X', '$\\Gamma$', "R"], show=False)
plt.savefig('LWF_PTO.pdf')
plt.show()
