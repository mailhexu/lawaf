import numpy as np
from ase.io import write
from banddownfolder.scdm import PhonopyDownfolder
from banddownfolder.scdm.lwf import LWF
import matplotlib.pyplot as plt

def symmetrize_LWF(lwf):
    for iR, R in enumerate(lwf.Rlist):
        a=np.average(np.diag(lwf.HwannR[iR]))
        print(R, np.diag(lwf.HwannR[iR]))
        lwf.HwannR[iR][0,0]=a
        lwf.HwannR[iR][1,1]=a
        #lwf.HwannR[iR]=(lwf.HwannR[iR]+lwf.HwannR[iR].T.conj())/2.0
        lwf.HwannR[iR].imag=0.0
    return lwf

symmetrize_LWF=None

def scale(lwf, factor=340.0/1200.0):
    lwf.HwannR*=factor
    return lwf

fname = '/home/hexu/projects/VO2/phonopy_params.yaml'
downfolder=PhonopyDownfolder(phonopy_yaml=fname, mode='ifc')
downfolder.downfold(method='scdmk',nwann=2, selected_basis=[2,5], 
                    #anchors={(.5,0.0,.5):(1,0)},
                    use_proj=False, mu=-10, sigma=15, weight_func='Gauss', kmesh=(3,3,5), post_func=scale)
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
plt.savefig('LWF_VO2.pdf')
plt.show()
