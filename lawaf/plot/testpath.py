import numpy as np
from ase.dft.kpoints import bandpath
kvectors=np.array([[0, 0, 0],
                   [2.0/3, 1.0/3, 0], [0,0.5,0], [0,0,0]]),
knames=['$\Gamma$', 'K', 'M','\Gamma' ],
npoints=20
cell=[[2.46841652, 0.0, 0.0], [-1.2342082599999995, 2.137711413441179, 0.0], [0.0, 0.0, 9.99905754]]
band = bandpath(kvectors, cell, npoints)
kpts = band.kpts
x, X, _labels = band.get_linear_kpoint_axis()


print(kpts)
print(f"{x=}")
print(f"{X=}")
print(f"{_labels=}")
print(_labels)
