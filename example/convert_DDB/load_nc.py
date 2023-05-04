from wannierbuilder.scdm.lwf import LWF
from wannierbuilder.plot import plot_band
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
lwf = LWF.load_nc(fname='./run_lwf.nc', prefix='', order='F')
ax = plot_band(lwf,
               kvectors=[[0.0, 0.0, 0.0], [0.5, 0, 0], [
                   0.5, 0.5, 0], [0, 0, 0], [.5, .5, .5]],
               knames='GXMGR',
               npoints=300,
               efermi=0.0,
               alpha=0.5,
               cell=np.eye(3),
               evals_to_freq=True,
               unit_factor=219474.6,
               label='Frequency (cm$^{-1}$)',
               ax=ax)
plt.show()
