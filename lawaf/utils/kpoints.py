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
