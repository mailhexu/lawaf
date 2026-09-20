"""Epic-13 phonon MMN provider and GL-covariant link diagnostics."""

import numpy as np
import pytest

from lawaf.wannierization.covariant_mmn import (
    covariant_mmn_links,
    electron_covariant_mmn,
    frame_projector,
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
