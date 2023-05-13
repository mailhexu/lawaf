import numpy as np


def kmesh_to_R(kmesh):
    """
    Build the commensurate R point for a kmesh.
    """
    k1, k2, k3 = kmesh
    Rlist = [(R1, R2, R3) for R1 in range(-k1 // 2 + 1, k1 // 2 + 1)
             for R2 in range(-k2 // 2 + 1, k2 // 2 + 1)
             for R3 in range(-k3 // 2 + 1, k3 // 2 + 1)]
    return np.array(Rlist)


def build_Rgrid(R):
    """
    Build R-point grid from the number
    """
    l1, l2, l3 = R
    Rlist = [(R1, R2, R3) for R1 in range(-l1 // 2 + 1, l1 // 2 + 1)
             for R2 in range(-l2 // 2 + 1, l2 // 2 + 1)
             for R3 in range(-l3 // 2 + 1, l3 // 2 + 1)]
    return np.array(Rlist)


def monkhorst_pack(size, gamma=True):
    """Construct a uniform sampling of k-space of given size. 
    Modified from ase.dft.kpoints with gamma_center option added"""
    if np.less_equal(size, 0).any():
        raise ValueError('Illegal size: %s' % list(size))
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    asize = np.array(size)
    mkpts = (kpts + 0.5) / size - 0.5
    if gamma:
        shift = 0.5 * ((asize + 1) % 2) / asize
        mkpts += shift
    return mkpts
