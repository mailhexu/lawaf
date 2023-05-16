import numpy as np
from phonopy import load
from ase.io import write
from lawaf.scdm import PhonopyDownfolder
from lawaf.plot import plot_band
from lawaf.wrapper.phonondownfolderwrapper import PhonopyDownfolderWrapper
import matplotlib.pyplot as plt

fname = 'phonopy_params.yaml'
#phonon=load(fname, is_nac=True)
#print(phonon.nac_params)
#exit()
downfolder=PhonopyDownfolder(phonopy_yaml=fname, mode="DM", has_nac=True)
#phonon=load(force_sets_filename="FORCE_SETS", born_filename="./BORN", unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3 )
#downfolder=PhonopyDownfolder(force_sets_filename="FORCE_SETS", 
#          #born_filename="./BORN", 
#          unitcell_filename="POSCAR-unitcell",supercell_matrix=np.eye(3)*3, mode="DM")
lwf=downfolder.downfold(method='scdmk',nwann=3, #selected_basis=[2,5], 
                    anchors={(.0,.0,.0):(0,1, 2)},
                    use_proj=True, mu=-200, sigma=300.4, weight_func='Gauss', kmesh=(4,4,4) )

m=PhonopyDownfolderWrapper(downfolder)

ax = plot_band(
    m, 
    kvectors=np.array([[0. , 0. , 0. ],
           [0.5, 0.0, 0. ],
           [0.5, 0.5, 0.0],
           [0.5 , 0.5 , 0.5 ],
           [0.5, 0.0, 0.0],
           [0.0,0.0,0],
           [0.5,0.5,0.5]                           
           ]), npoints=80,
    knames=['$\\Gamma$', 'X','M', 'R', 'X', '$\\Gamma$', "R"],
    efermi=0,
    color="green",
    alpha=0.5,
    marker=".",
    fix_LOTO=True,
    unit_factor=15.6*33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True,

)


print(lwf.get_wannier_born())
write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax=downfolder.plot_band_fitting(ax=ax, kvectors=np.array([[0. , 0. , 0. ],
           [0.5, 0.0, 0. ],
           [0.5, 0.5, 0.0],
           [0.5 , 0.5 , 0.5 ],
           [0.5, 0.0, 0.0],
           [0.0,0.0,0],
           [0.5,0.5,0.5]                           
           ]),
           npoints=80,
    knames=['$\\Gamma$', 'X','M', 'R', 'X', '$\\Gamma$', "R"], plot_original=True, plot_downfolded=False, show=False, fix_LOTO=True ,
     unit_factor=15.6*33.6,
    ylabel="Frequency (cm^-1)",
    evals_to_freq=True, )
plt.savefig('phonon.pdf')
plt.show()
