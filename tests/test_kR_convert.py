import numpy as np

from lawaf.mathutils.kR_convert import HR_to_k, R_to_k, R_to_onek, k_to_R
from lawaf.utils.kpoints import build_Rgrid, monkhorst_pack


def random_hermitian_hk(nkpt, nwann, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((nkpt, nwann, nwann)) + \
        1j * rng.standard_normal((nkpt, nwann, nwann))
    return H + H.conj().transpose(0, 2, 1)


def check_roundtrip(kmesh, grid_kwargs):
    kpts = monkhorst_pack(kmesh)
    Hk = random_hermitian_hk(len(kpts), 4, seed=sum(kmesh))
    Rlist, Rdeg = build_Rgrid(kmesh, **grid_kwargs)
    HR = k_to_R(kpts, Rlist, Hk)
    # full-value storage: aliased images carry identical full coefficients
    Hk2 = R_to_k(kpts, Rlist, HR, Rdeg=Rdeg)
    assert np.allclose(Hk, Hk2, atol=1e-12)
    for ik, k in enumerate(kpts):
        assert np.allclose(R_to_onek(k, Rlist, HR, Rdeg=Rdeg), Hk[ik], atol=1e-12)
    # HR_to_k matches R_to_k
    assert np.allclose(HR_to_k(HR, Rlist, kpts, Rdeg=Rdeg), Hk2, atol=1e-12)


def test_roundtrip_222_plain():
    check_roundtrip([2, 2, 2], {})


def test_roundtrip_222_wigner_seitz():
    check_roundtrip([2, 2, 2], {"wigner_seitz": True})


def test_roundtrip_333_plain():
    check_roundtrip([3, 3, 3], {})


def test_roundtrip_333_wigner_seitz():
    check_roundtrip([3, 3, 3], {"wigner_seitz": True})


def test_roundtrip_224_wigner_seitz():
    check_roundtrip([2, 2, 4], {"wigner_seitz": True})


def test_roundtrip_default_rdeg_is_ones():
    """Omitting Rdeg must equal Rdeg=1 (plain grids)."""
    kpts = monkhorst_pack([3, 3, 3])
    Hk = random_hermitian_hk(len(kpts), 3, seed=7)
    Rlist, Rdeg = build_Rgrid([3, 3, 3])
    HR = k_to_R(kpts, Rlist, Hk)
    assert np.all(Rdeg == 1)
    assert np.allclose(R_to_k(kpts, Rlist, HR), R_to_k(kpts, Rlist, HR, Rdeg=Rdeg))


def test_interpolation_invariant_folded_vs_full():
    """The old convention (Rdeg folded into values + plain sum) and the new
    one (full values + weighted sum) give identical interpolation at every
    k point, mesh and off-mesh."""
    kmesh = [2, 2, 2]
    kpts = monkhorst_pack(kmesh)
    Hk = random_hermitian_hk(len(kpts), 3, seed=11)
    Rlist, Rdeg = build_Rgrid(kmesh, wigner_seitz=True)
    HR_full = k_to_R(kpts, Rlist, Hk)  # full values
    HR_folded = HR_full * Rdeg[:, None, None]  # legacy convention
    for k in (np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.5]),
              np.array([0.13, 0.27, 0.41])):
        new = R_to_onek(k, Rlist, HR_full, Rdeg=Rdeg)
        old = R_to_onek(k, Rlist, HR_folded)  # plain, unweighted
        assert np.allclose(new, old, atol=1e-12)


def test_wigner_seitz_grid_shape():
    Rlist, Rdeg = build_Rgrid([2, 2, 2], wigner_seitz=True)
    assert Rlist.shape == (27, 3)
    assert sorted(map(tuple, Rlist)) == sorted(
        (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
    # corner vectors carry 1/8 weight, face centers 1/2, origin 1
    d = dict(zip(map(tuple, Rlist), Rdeg))
    assert d[(0, 0, 0)] == 1
    assert d[(1, 0, 0)] == 0.5
    assert d[(1, 1, 0)] == 0.25
    assert d[(1, 1, 1)] == 0.125
    # odd mesh: identity weights
    Rlist3, Rdeg3 = build_Rgrid([3, 3, 3], wigner_seitz=True)
    assert len(Rlist3) == 27 and np.all(Rdeg3 == 1)
    # degeneracy alias still works
    Rlist_a, Rdeg_a = build_Rgrid([2, 2, 2], degeneracy=True)
    assert np.array_equal(Rlist_a, Rlist) and np.array_equal(Rdeg_a, Rdeg)
