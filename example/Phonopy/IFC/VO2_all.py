import numpy as np
from ase.io import write
from banddownfolder.scdm import PhonopyDownfolder
from banddownfolder.scdm.lwf import LWF
import matplotlib.pyplot as plt


symmetrize_LWF=None

fname = '/home/hexu/projects/VO2/phonopy_params.yaml'
downfolder=PhonopyDownfolder(phonopy_yaml=fname, mode='ifc')
basis=list(range(18))
#basis.remove(2)
#basis.remove(5)
downfolder.downfold(method='scdmk',nwann=18, selected_basis=basis, 
                    #anchors={(.5,0.0,.5):(1,0)},
                    use_proj=False, mu=-20, sigma=10, weight_func='Gauss', kmesh=(3,3,5), post_func=symmetrize_LWF)
#downfolder.ewf=symmetrize_LWF(downfolder.ewf)
#downfolder.ewf=symmetrize_LWF(downfolder.ewf)
#print(downfolder.ewf.Rlist[0])
#print(downfolder.ewf.HwannR[0])

write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax=downfolder.plot_band_fitting(kvectors=np.array([[0. , 0. , 0. ],
           [0.5, 0.5 , 0. ],
           [0.5, 0.5, 0.5],
           [0. , 0. , 0.5 ],
           [0.5, 0.0, 0.5],
           [0.5,0.5,0],
           [0,0,0],
           [0.0,0.0,0.5]                           
           ]),
    knames=['$\\Gamma$', 'M','A', 'Z','R', 'M', '$\\Gamma$', 'Z'], show=False)
plt.savefig('LWF_VO2_reverse.pdf')
plt.show()
