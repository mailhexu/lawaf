import numpy as np

from lawaf.mathutils.ws_distance import ws_translate_dist, apply_ws_distance
from lawaf.mathutils.kR_convert import R_to_onek

NDEGX = 8


def cubic_cell(a=3.905):
    return np.eye(3) * a


def mesh_grid(n, offset=0):
    """n^3 grid, components in range(offset, offset + n)."""
    return np.array([[i, j, k] for i in range(offset, offset + n)
                     for j in range(offset, offset + n)
                     for k in range(offset, offset + n)], dtype=int)


def random_hermitian_hk(kmesh, nwann, seed=0):
    rng = np.random.default_rng(seed)
    kpts = np.array([[i / kmesh[0], j / kmesh[1], k / kmesh[2]]
                     for i in range(kmesh[0])
                     for j in range(kmesh[1])
                     for k in range(kmesh[2])])
    H = rng.standard_normal((len(kpts), nwann, nwann)) + \
        1j * rng.standard_normal((len(kpts), nwann, nwann))
    H = H + H.conj().transpose(0, 2, 1)
    return kpts, H


def dft_k_to_R(kpts, Hk, Rlist):
    kweights = np.ones(len(kpts)) / len(kpts)
    HR = np.zeros((len(Rlist),) + Hk.shape[1:], dtype=complex)
    for iR, R in enumerate(Rlist):
        for ik, k in enumerate(kpts):
            HR[iR] += Hk[ik] * np.exp(-2j * np.pi * np.dot(k, R)) * kweights[ik]
    return HR


def test_identity_near_origin_odd_mesh():
    """3x3x3 mesh, centers at origin: minimal image is R itself."""
    Rlist = mesh_grid(3, -1)
    centers = np.zeros((2, 3))
    shifts, ndeg = ws_translate_dist(Rlist, centers, cubic_cell(), (3, 3, 3))
    assert ndeg.shape == (27, 2, 2)
    assert np.all(ndeg == 1)
    assert np.all(shifts[:, :, :, 0, :] == 0)


def test_even_mesh_inside_ws_cell():
    """2x2x2 mesh with coincident centers: images fold into {-1,0,1}^3."""
    Rlist = mesh_grid(2)
    centers = np.zeros((1, 3))
    shifts, ndeg = ws_translate_dist(Rlist, centers, cubic_cell(), (2, 2, 2))
    for iR in range(8):
        for d in range(ndeg[iR, 0, 0]):
            img = Rlist[iR] + shifts[iR, 0, 0, d, :]
            assert np.all(np.abs(img) <= 1)


def test_exact_tie_degenerate_pair():
    """R=(1,0,0), coincident centers, 2x2x2: +1 and -1 equidistant -> ndeg=2."""
    Rlist = np.array([[1, 0, 0]])
    centers = np.zeros((1, 3))
    shifts, ndeg = ws_translate_dist(Rlist, centers, cubic_cell(), (2, 2, 2))
    assert ndeg[0, 0, 0] == 2
    imgs = sorted(tuple(Rlist[0] + shifts[0, 0, 0, d, :]) for d in range(2))
    assert imgs == [(-1, 0, 0), (1, 0, 0)]


def test_skewed_cell_metric_decides():
    """Monoclinic cell with b skewed along +a, even grid along b: for the
    pair with r_j - r_i = +0.1a the image at -b is closer in the Cartesian
    metric, so R=(0,1,0) folds to (0,-1,0) with shift n*mp_grid=(0,-2,0).
    Componentwise folding would have kept (0,1,0)."""
    cell = np.array([[10.0, 0.0, 0.0], [1.5, 1.0, 0.0], [0.0, 0.0, 10.0]])
    Rlist = np.array([[0, 1, 0]])
    centers = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    shifts, ndeg = ws_translate_dist(Rlist, centers, cell, (3, 2, 3))
    assert ndeg[0, 0, 1] == 1
    assert tuple(shifts[0, 0, 1, 0, :]) == (0, -2, 0)
    assert tuple(Rlist[0] + shifts[0, 0, 1, 0, :]) == (0, -1, 0)


def test_corner_eightfold_degenerate():
    """Cube corner R=(1,1,1), coincident centers, 2x2x2: all 8 images tie."""
    Rlist = np.array([[1, 1, 1]])
    centers = np.zeros((1, 3))
    shifts, ndeg = ws_translate_dist(Rlist, centers, cubic_cell(), (2, 2, 2))
    assert ndeg[0, 0, 0] == 8
    assert np.all(ndeg <= NDEGX)


def test_apply_ws_distance_w90_reference():
    """Materialized plain-sum == explicit W90 per-image phase average."""
    kmesh = (2, 2, 2)
    nwann = 3
    kpts, Hk = random_hermitian_hk(kmesh, nwann)
    Rlist = mesh_grid(2)
    HR = dft_k_to_R(kpts, Hk, Rlist)
    centers = np.random.default_rng(1).standard_normal((nwann, 3)) * 0.05
    cell = cubic_cell()
    HR_ws, Rlist_ws, Rdeg = apply_ws_distance(HR, Rlist, centers, cell, kmesh)
    assert len(np.unique(Rlist_ws, axis=0)) == len(Rlist_ws)
    assert np.all(Rdeg == 1)

    shifts, ndeg = ws_translate_dist(Rlist, centers, cell, kmesh)
    for k in (np.array([0.13, 0.27, 0.41]), np.array([0.0, 0.5, 0.0])):
        ref = np.zeros((nwann, nwann), dtype=complex)
        for iR, R in enumerate(Rlist):
            for i in range(nwann):
                for j in range(nwann):
                    for d in range(ndeg[iR, i, j]):
                        Reff = R + shifts[iR, i, j, d, :]
                        ref[i, j] += HR[iR, i, j] * np.exp(
                            2j * np.pi * np.dot(k, Reff)) / ndeg[iR, i, j]
        got = R_to_onek(k, Rlist_ws, HR_ws)
        assert np.allclose(got, ref, atol=1e-12)


def test_apply_ws_distance_exact_at_mesh_points():
    kmesh = (2, 2, 2)
    nwann = 2
    kpts, Hk = random_hermitian_hk(kmesh, nwann, seed=3)
    Rlist = mesh_grid(2)
    HR = dft_k_to_R(kpts, Hk, Rlist)
    centers = np.zeros((nwann, 3))
    HR_ws, Rlist_ws, _ = apply_ws_distance(HR, Rlist, centers, cubic_cell(), kmesh)
    for ik, k in enumerate(kpts):
        got = R_to_onek(k, Rlist_ws, HR_ws)
        assert np.allclose(got, Hk[ik], atol=1e-12)


def test_fold_plus_ws_equals_weighted_grid_sum():
    """lawaf extended grid + Rdeg weights -> fold -> WS materialization
    reconstructs H(k) at every mesh k-point (Story-5 acceptance)."""
    from lawaf.mathutils.kR_convert import k_to_R
    from lawaf.mathutils.ws_distance import (apply_ws_distance,
                                             fold_R_to_mesh)
    from lawaf.utils.kpoints import build_Rgrid

    kmesh = (2, 2, 2)
    kpts, Hk = random_hermitian_hk(kmesh, nwann=3, seed=42)
    Rlist, Rdeg = build_Rgrid(kmesh, degeneracy=True)
    HR = k_to_R(kpts, Rlist, Hk, kweights=np.ones(len(kpts)) / len(kpts))
    # legacy convention: full coefficients at every extended-grid R
    cell = cubic_cell()
    centers = np.zeros((3, 3))
    tensors_f, Rlist_f = fold_R_to_mesh(Rlist, [HR], kmesh, Rdeg=Rdeg)
    HR_ws, Rl_ws, _ = apply_ws_distance(tensors_f[0], Rlist_f, centers,
                                        cell, kmesh)
    for ik, k in enumerate(kpts):
        got = R_to_onek(k, Rl_ws, HR_ws)
        assert np.allclose(got, Hk[ik], atol=1e-12)
