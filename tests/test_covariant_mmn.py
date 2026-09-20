"""Epic-13 phonon MMN provider and GL-covariant link diagnostics."""

import numpy as np
import pytest
from scipy.linalg import expm

from lawaf.io.w90 import compute_Mmn, kmesh_nnlist, read_mmn, write_mmn
from lawaf.wannierization.covariant_mmn import (
    covariant_mmn_links,
    electron_covariant_mmn,
    frame_projector,
    mmn_form_spread,
    normalized_line_links,
    phonon_mmn,
)


def _unitary_frames(nk=6, nb=8, nw=3, seed=20260920):
    rng = np.random.default_rng(seed)
    U = np.empty((nk, nb, nw), dtype=complex)
    for k in range(nk):
        q, _ = np.linalg.qr(
            rng.standard_normal((nb, nw)) + 1j * rng.standard_normal((nb, nw))
        )
        U[k] = q
    nnlist = np.array([[(k + 1) % nk, (k - 1) % nk] for k in range(nk)])
    M = np.empty((nk, 2, nw, nw), dtype=complex)
    for k in range(nk):
        for b in range(2):
            M[k, b] = U[k].conj().T @ U[nnlist[k, b]]
    return U, M, nnlist, rng


def _fullrank_frames(rng, nk, nw):
    return np.array([
        np.eye(nw) + 0.2 * (
            rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
        )
        for _ in range(nk)
    ])


def test_gl_covariance_projector_and_line_modulus():
    U, M, nnlist, rng = _unitary_frames()
    nk, _, nw = U.shape
    G = _fullrank_frames(rng, nk, nw)
    R = _fullrank_frames(rng, nk, nw)
    Q, B, L = covariant_mmn_links(M, nnlist, G)
    Qp, Bp, Lp = covariant_mmn_links(M, nnlist, G @ R)
    for k in range(nk):
        for b in range(nnlist.shape[1]):
            j = nnlist[k, b]
            assert np.abs(Bp[k, b] - R[k].conj().T @ B[k, b] @ R[j]).max() < 1e-12
            assert np.abs(Lp[k, b] - np.linalg.solve(R[k], L[k, b] @ R[j])).max() < 1e-12
    assert np.abs(frame_projector(U, G) - frame_projector(U, G @ R)).max() < 1e-12
    # Per-column rescaling preserves normalized-link moduli.
    D = np.array([
        np.diag(rng.uniform(0.4, 2.0, nw) * np.exp(1j * rng.uniform(0, 2 * np.pi, nw)))
        for _ in range(nk)
    ])
    Qd, Bd, _ = covariant_mmn_links(M, nnlist, G @ D)
    z = normalized_line_links(B, Q, nnlist)
    zd = normalized_line_links(Bd, Qd, nnlist)
    assert np.abs(np.abs(z) - np.abs(zd)).max() < 1e-12


def test_provider_and_guards():
    U, M, nnlist, _ = _unitary_frames(nk=4, nb=5, nw=2)
    G = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    with pytest.raises(ValueError, match="singular"):
        covariant_mmn_links(M, nnlist, G * 0.0)
    with pytest.raises(NotImplementedError, match="cross-k"):
        electron_covariant_mmn()


def test_phonon_provider_matches_direct_overlap():
    # Gamma-centred 2x2x2 mesh, orthonormal displacement eigenvectors.
    kpts = np.array([[i, j, k] for i in (0.0, 0.5) for j in (0.0, 0.5) for k in (0.0, 0.5)])
    rng = np.random.default_rng(4)
    evecs = np.empty((len(kpts), 6, 3), dtype=complex)
    for ik in range(len(kpts)):
        q, _ = np.linalg.qr(rng.standard_normal((6, 3)) + 1j * rng.standard_normal((6, 3)))
        evecs[ik] = q
    data = phonon_mmn(evecs, kpts, np.eye(3))
    assert data.mmn.shape[:2] == data.nnlist.shape
    for ik in range(len(kpts)):
        for ib, jk in enumerate(data.nnlist[ik]):
            assert np.abs(data.mmn[ik, ib] - evecs[ik].conj().T @ evecs[jk]).max() < 1e-12


def test_package_exports():
    import lawaf

    assert lawaf.phonon_mmn is phonon_mmn
    assert lawaf.covariant_mmn_links is covariant_mmn_links


def test_mmn_roundtrip_provider(tmp_path):
    """write_mmn -> read_mmn round trip is the explicit electron provider
    entry: read-back links drive covariant_mmn_links identically."""
    rng = np.random.default_rng(9)
    kpts = np.array(
        [[i, j, k] for i in (0.0, 0.5) for j in (0.0, 0.5) for k in (0.0, 0.5)]
    )
    cell = 4.0 * np.eye(3)
    recip = 2 * np.pi * np.linalg.inv(cell).T
    psi = np.empty((8, 4, 4), dtype=complex)
    for ik in range(8):
        q, _ = np.linalg.qr(
            rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        )
        psi[ik] = q
    nnlist, nncell, _, _ = kmesh_nnlist(kpts, recip)
    M = compute_Mmn(psi, nnlist, nncell)
    path = tmp_path / "provider.mmn"
    write_mmn(M, nnlist, nncell, str(path))
    M2, nn2, nc2 = read_mmn(str(path))
    assert np.array_equal(nn2, nnlist) and np.array_equal(nc2, nncell)
    assert np.abs(M - M2).max() < 1e-11  # 12-decimal text precision
    # covariant machinery works on the read-back provider data
    G = np.array([
        np.eye(4) + 0.15 * (
            rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        )
        for _ in range(8)
    ])
    Q, B, L = covariant_mmn_links(M2, nn2, G)
    Qr, Br, Lr = covariant_mmn_links(M, nnlist, G)
    assert np.abs(B - Br).max() < 1e-11
    assert np.abs(L - Lr).max() < 1e-11


def test_mmn_form_spread_converges_to_quantum_metric():
    """On a smooth periodic frame U(k) = expm(i sum_a f_a(k) H_a), the
    link functional Omega_I converges with O(b^2) to the exact
    k-averaged line metric sum_a [||d_a u_n||^2 - (Im<u_n|d_a u_n>)^2]
    (Berry-connection squared subtracted). Rough projected gauges
    (random eigenvector phases) do not satisfy this, so certification
    uses the analytic frame; see research/2026-09-21-mmn-functional.md.
    """
    from lawaf.io.w90 import compute_Mmn, kmesh_nnlist

    rng = np.random.default_rng(31)
    nw = 3

    def herm():
        a = rng.standard_normal((nw, nw)) + 1j * rng.standard_normal((nw, nw))
        return (a + a.conj().T) / 2

    Hs = [herm() for _ in range(3)]
    eps = 0.1

    def U_of(ks):
        out = np.empty((len(ks), nw, nw), dtype=complex)
        for ik, k in enumerate(np.asarray(ks)):
            out[ik] = expm(
                1j * eps * (np.cos(2 * np.pi * k[0]) * Hs[0]
                            + np.sin(2 * np.pi * k[1]) * Hs[1]
                            + np.cos(2 * np.pi * k[2]) * Hs[2])
            )
        return out

    def mesh(N):
        return np.array([[i, j, l] for i in range(N) for j in range(N)
                         for l in range(N)], dtype=float) / N

    cell = 4.0 * np.eye(3)
    recip = 2 * np.pi * np.linalg.inv(cell).T

    def omega_at(N):
        ks = mesh(N)
        U = U_of(ks)
        nnlist, nncell, bvecs, wb = kmesh_nnlist(ks, recip)
        M = compute_Mmn(U, nnlist, nncell)
        return mmn_form_spread(M, nnlist, bvecs, wb)["omega_I"]

    N = 20
    ks = mesh(N)
    dk = 1e-6
    tot = np.zeros(nw)
    for k in ks:
        for alpha in range(3):
            kp = k.copy()
            kp[alpha] += dk
            km = k.copy()
            km[alpha] -= dk
            dup = (U_of(kp[None, :])[0] - U_of(km[None, :])[0]) / (2 * dk)
            u = U_of(k[None, :])[0]
            berry2 = np.imag(np.einsum("an,an->n", u.conj(), dup)) ** 2
            tot += np.einsum("an,an->n", dup.conj(), dup).real - berry2
    tgt = tot / N**3 * (4.0 / (2 * np.pi)) ** 2

    o8 = omega_at(8)
    o16 = omega_at(16)
    e8 = np.abs(o8 - tgt).max()
    e16 = np.abs(o16 - tgt).max()
    assert e16 < e8 / 3.0, (e8, e16)
    assert e16 < 0.05 * np.abs(tgt).max(), (e16, tgt)
    assert np.allclose(o16, tgt, rtol=0.05)
