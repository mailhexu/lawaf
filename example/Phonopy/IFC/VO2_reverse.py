import numpy as np
from ase.io import write
from banddownfolder.scdm import PhonopyDownfolder
from banddownfolder.scdm.lwf import LWF
import matplotlib.pyplot as plt


symmetrize_LWF=None

fname = '/home/hexu/projects/VO2/phonopy_params.yaml'
downfolder=PhonopyDownfolder(phonopy_yaml=fname, mode='ifc')
basis=list(range(18))
basis.remove(2)
basis.remove(5)
downfolder.downfold(method='scdmk',nwann=16, #selected_basis=basis,
                    anchors={(0.5, 0,0.5):list(range(2,18))},
                    use_proj=True, mu=0, sigma=100, weight_func='Gauss', kmesh=(3,3,5), post_func=symmetrize_LWF, write_hr_nc='VO2_othermodes.nc',
        write_hr_txt='VO2_othermodes.txt',)
#downfolder.ewf=symmetrize_LWF(downfolder.ewf)
#downfolder.ewf=symmetrize_LWF(downfolder.ewf)
#print(downfolder.ewf.Rlist[0])
#print(downfolder.ewf.HwannR[0])

write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)

downfolder2=PhonopyDownfolder(phonopy_yaml=fname, mode='ifc')
downfolder2.downfold(method='scdmk',nwann=2, 
                    anchors={(.5,0.0,.5):(0,1)}, 
                    #selected_basis=[2,5],
                    use_proj=True, mu=-6, sigma=17, weight_func='Gauss', kmesh=(3,3,5), post_func=symmetrize_LWF, )
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
    knames=['$\\Gamma$', 'M','A', 'Z','R', 'M', '$\\Gamma$', 'Z'], show=False, savefig=None)


ax=downfolder2.plot_band_fitting(kvectors=np.array([[0. , 0. , 0. ],
           [0.5, 0.5 , 0. ],
           [0.5, 0.5, 0.5],
           [0. , 0. , 0.5 ],
           [0.5, 0.0, 0.5],
           [0.5,0.5,0],
           [0,0,0],
           [0.0,0.0,0.5]                           
           ]),
           downfolded_band_color='red',
           knames=['$\\Gamma$', 'M','A', 'Z','R', 'M', '$\\Gamma$', 'Z'], show=False, ax=ax)
 
plt.savefig('LWF_VO2_all.pdf')
plt.show()
