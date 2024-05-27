import numpy as np
from ase.cell import Cell
from ase.dft.kpoints import bandpath


def kmesh_to_R(kmesh):
    """
    Build the commensurate R point for a kmesh.
    """
    k1, k2, k3 = kmesh
    Rlist = [
        (R1, R2, R3)
        for R1 in range(-k1 // 2 + 1, k1 // 2 + 1)
        for R2 in range(-k2 // 2 + 1, k2 // 2 + 1)
        for R3 in range(-k3 // 2 + 1, k3 // 2 + 1)
    ]
    return np.array(Rlist)


def build_Rgrid(R):
    """
    Build R-point grid from the number
    """
    l1, l2, l3 = R
    Rlist = [
        (R1, R2, R3)
        for R1 in range(-l1 // 2 + 1, l1 // 2 + 1)
        for R2 in range(-l2 // 2 + 1, l2 // 2 + 1)
        for R3 in range(-l3 // 2 + 1, l3 // 2 + 1)
    ]
    return np.array(Rlist)


def monkhorst_pack(size, gamma=True):
    """Construct a uniform sampling of k-space of given size.
    Modified from ase.dft.kpoints with gamma_center option added"""
    if np.less_equal(size, 0).any():
        raise ValueError("Illegal size: %s" % list(size))
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    asize = np.array(size)
    mkpts = (kpts + 0.5) / size - 0.5
    if gamma:
        shift = 0.5 * ((asize + 1) % 2) / asize
        mkpts += shift
    return mkpts


def group_band_path(bp, eps=1e-8, shift=0.15):
    xs, Xs, knames = bp.get_linear_kpoint_axis()
    kpts = bp.kpts

    m = xs[1:] - xs[:-1] < eps
    segments = [0] + list(np.where(m)[0] + 1) + [len(xs)]

    # split Xlist
    xlist, kptlist = [], []
    for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
        kptlist.append(kpts[start:end])
        xlist.append(xs[start:end] + i * shift)

    m = Xs[1:] - Xs[:-1] < eps

    s = np.where(m)[0] + 1

    for i in s:
        Xs[i:] += shift

    return xlist, kptlist, Xs, knames


def autopath(knames=None, kvectors=None, npoints=100, supercell_matrix=None, cell=None):
    kptlist = kvectors
    if knames is None and kvectors is None:
        # fully automatic k-path
        bp = Cell(cell).bandpath(npoints=npoints)
        spk = bp.special_points
        xlist, kptlist, Xs, knames = group_band_path(bp)
    elif knames is not None and kvectors is None:
        # user specified kpath by name
        bp = Cell(cell).bandpath(knames, npoints=npoints)
        spk = bp.special_points
        kpts = bp.kpts
        xlist, kptlist, Xs, knames = group_band_path(bp)
    else:
        # user spcified kpath and kvector.
        kpts, x, Xs = bandpath(kvectors, cell, npoints)
        spk = dict(zip(knames, kvectors))
        xlist = [x]
        kptlist = [kpts]

    if supercell_matrix is not None:
        kptlist = [np.dot(k, supercell_matrix) for k in kptlist]
    for name, k in spk.items():
        if name == "G":
            name = "Gamma"
    return knames, kptlist, xlist, Xs
