"""Tests for MMN computation and .mmn writing (lawaf.io.w90).

Format ground truth: wannier90 v3.1.0 overlap.F90:133-198 —
comment line; ``num_bands num_kpts nntot``; per (k, neighbour) block:
``ik jk G1 G2 G3`` (1-based k indices, G = lattice image triple), then
the matrix read as ``do n; do m: Re Im`` i.e. n outer, m inner,
M(m, n) = <u_{m,ik} | u_{n,jk}>. Every block is matched by wannier90
against its own kmesh_get neighbour list by (jk, G).
"""

from pathlib import Path

import numpy as np
import pytest

from lawaf.io.w90 import compute_Mmn, kmesh_nnlist, write_mmn


def random_unitaries(nk, nb, seed=0):
    rng = np.random.default_rng(seed)
    out = np.empty((nk, nb, nb), dtype=complex)
    for ik in range(nk):
        q, _ = np.linalg.qr(
            rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        )
        out[ik] = q
    return out


def cubic_recip(a=3.9):
    return 2 * np.pi / a * np.eye(3)


def test_mmn_unitarity():
    """M(k,b) = U_k^dag U_jk between complete orthonormal bases is unitary:
    M^dag M = I for every (k, b)."""
    from lawaf.utils.kpoints import monkhorst_pack
    nk, nb = 8, 4
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    psi = random_unitaries(nk, nb, seed=7)
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    mmn = compute_Mmn(psi, nnlist, nncell)
    assert mmn.shape == (nk, nnlist.shape[1], nb, nb)
    for ik in range(nk):
        for ib in range(nnlist.shape[1]):
            m = mmn[ik, ib]
            np.testing.assert_allclose(
                m.conj().T @ m, np.eye(nb), atol=1e-12
            )

def test_mmn_conjugation_direction():
    """M(m,n) = <u_m,k | u_n,jk>: rows indexed by states at k, columns by
    states at jk. Construct U_jk = U_k V for a known COMPLEX unitary V and
    check (a complex V also catches a missing conjugation directly)."""
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    # k=0's first neighbour in the w90 list is jk=1 (same cell, G=0)
    ik, ib, jk = 0, 0, nnlist[0, 0]
    assert tuple(nncell[0, 0]) == (0, 0, 0)
    rng = np.random.default_rng(3)
    v, _ = np.linalg.qr(
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    )
    psi = random_unitaries(len(kpts), 4, seed=5)
    psi[jk] = psi[ik] @ v  # jk basis is ik basis rotated by V
    mmn = compute_Mmn(psi, nnlist, nncell)
    np.testing.assert_allclose(mmn[ik, ib], v, atol=1e-12)


def test_mmn_rectangular_psi():
    """psi layout (nkpt, nbasis, nband) with nbasis != nband — the
    Wannierizer.get_psi_k layout: M = psi_k^dag psi_jk is nband x nband."""
    from lawaf.utils.kpoints import monkhorst_pack
    rng = np.random.default_rng(21)
    nbasis, nband, nk = 7, 4, 8
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    psi = np.empty((nk, nbasis, nband), dtype=complex)
    for ik in range(nk):
        q, _ = np.linalg.qr(
            rng.standard_normal((nbasis, nband))
            + 1j * rng.standard_normal((nbasis, nband))
        )
        psi[ik] = q
    mmn = compute_Mmn(psi, nnlist, nncell)
    assert mmn.shape == (nk, nnlist.shape[1], nband, nband)
    np.testing.assert_allclose(mmn[0, 0], psi[0].conj().T @ psi[1], atol=1e-12)

def test_mmn_g_invariance():
    """Periodic gauge: the VALUE must not depend on the lattice-image
    label G — only the .mmn headers carry it."""
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    psi = random_unitaries(8, 3, seed=23)
    mmn1 = compute_Mmn(psi, nnlist, nncell)
    nncell2 = nncell + 1  # bogus but value-irrelevant labels
    mmn2 = compute_Mmn(psi, nnlist, nncell2)
    np.testing.assert_array_equal(mmn1, mmn2)


def test_mmn_real_psi_complex_output():
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, cubic_recip())
    rng = np.random.default_rng(31)
    psi = rng.standard_normal((8, 4, 4))  # real input
    mmn = compute_Mmn(psi, nnlist, nncell)
    assert np.iscomplexobj(mmn)

def _parse_mmn(path):
    lines = Path(path).read_text().splitlines()
    nb, nk, nntot = (int(x) for x in lines[1].split())
    idx = 2
    blocks = {}
    for _ in range(nk * nntot):
        ik, jk, g1, g2, g3 = (int(x) for x in lines[idx].split())
        idx += 1
        m = np.empty((nb, nb), dtype=complex)
        for n in range(nb):
            for mi in range(nb):
                re, im = (float(x) for x in lines[idx].split())
                m[mi, n] = re + 1j * im
                idx += 1
        blocks[(ik - 1, jk - 1, g1, g2, g3)] = m
    assert idx == len(lines)
    return nb, nk, nntot, blocks


def test_mmn_roundtrip(tmp_path):
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    psi = random_unitaries(8, 3, seed=11)
    mmn = compute_Mmn(psi, nnlist, nncell)
    f = tmp_path / "seed.mmn"
    write_mmn(mmn, nnlist, nncell, f)
    nb, nk, nntot, blocks = _parse_mmn(f)
    assert (nb, nk, nntot) == (3, 8, nnlist.shape[1])
    for ik in range(nk):
        for ib in range(nntot):
            jk = nnlist[ik, ib]
            g = tuple(nncell[ik, ib])
            np.testing.assert_allclose(
                blocks[(ik, int(jk), *g)], mmn[ik, ib], atol=1e-12
            )


def test_mmn_roundtrip_wannier90io(tmp_path):
    w90io = pytest.importorskip("wannier90io")
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    psi = random_unitaries(8, 3, seed=13)
    mmn = compute_Mmn(psi, nnlist, nncell)
    f = tmp_path / "seed.mmn"
    write_mmn(mmn, nnlist, nncell, f)
    with open(f) as fh:
        mmn_back, nnk_back = w90io.read_mmn(fh)
    np.testing.assert_allclose(mmn_back, mmn, atol=1e-12)
    np.testing.assert_array_equal(nnk_back[:, :, 1], nnlist)  # jk, 0-based


def test_mmn_header_block_order(tmp_path):
    """First block must be ik=1 with its first w90 neighbour jk/G triple."""
    from lawaf.utils.kpoints import monkhorst_pack
    kpts = monkhorst_pack([2, 2, 2])
    recip = cubic_recip()
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    psi = random_unitaries(8, 2, seed=17)
    mmn = compute_Mmn(psi, nnlist, nncell)
    f = tmp_path / "seed.mmn"
    write_mmn(mmn, nnlist, nncell, f)
    lines = f.read_text().splitlines()
    assert lines[1].split() == ["2", "8", str(nnlist.shape[1])]
    first = lines[2].split()
    assert first[0] == "1"  # ik = 1 (1-based)
    assert int(first[1]) == nnlist[0, 0] + 1
    assert [int(x) for x in first[2:5]] == list(nncell[0, 0])
