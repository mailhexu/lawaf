"""GL-covariant MMN links for non-orthogonal Wannier frames.

This is the safe Epic-13 pilot, not a full non-orthogonal
Marzari--Vanderbilt optimizer. Given an orthonormal reference frame
``U(k)`` and a full-rank right GL frame ``G(k)``, the module builds

``Q(k) = G(k)^dag G(k)``,
``B(k,b) = G(k)^dag M(k,b) G(k+b)``, and
``L(k,b) = Q(k)^-1 B(k,b)``.

Under ``G(k) -> G(k) R(k)``, raw links B transform by congruence and
dual links L by left/right similarity; the projector
``P(k)=C(k) Q(k)^-1 C(k)^dag`` and closed Wilson-loop spectra are
therefore frame-invariant. The normalized diagonal link is a certified
per-line diagnostic under column rescaling, but is NOT a full
non-orthogonal MV spread decomposition; no optimizer is supplied here.

Phonon exactness: mass-weighted displacement eigenvectors share a
fixed orthonormal coordinate basis, so M(k,b)=psi(k)^dag psi(k+b) is
exact. Electron LCAO/W90 consumer models currently lack a cross-k
metric/MMN provider and must not synthesize it from same-k SR or atom
centres.
"""

from dataclasses import dataclass

import numpy as np

from lawaf.io.w90 import compute_Mmn, kmesh_nnlist

__all__ = [
    "CovariantMMN",
    "phonon_mmn",
    "covariant_mmn_links",
    "frame_projector",
    "normalized_line_links",
    "electron_covariant_mmn",
]


@dataclass(frozen=True)
class CovariantMMN:
    """Phonon MMN provider result, in lawaf/wannier90 conventions."""

    mmn: np.ndarray
    nnlist: np.ndarray
    nncell: np.ndarray
    bvecs: np.ndarray
    wb: np.ndarray


def phonon_mmn(evecs, kpts, cell, kmesh_tol=1e-6):
    """Build exact MMN links from phonon displacement eigenvectors.

    ``evecs[k]`` must have columns of mass-weighted phonon eigenvectors
    in the common orthonormal displacement coordinate basis. The result
    is exact for this discrete phonon Hilbert space.
    """
    evecs = np.asarray(evecs, dtype=complex)
    kpts = np.asarray(kpts, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if evecs.ndim != 3 or evecs.shape[0] != len(kpts):
        raise ValueError(
            "evecs must have shape (nk, nbasis, nband) with nk=len(kpts), "
            f"got {evecs.shape} and {len(kpts)}"
        )
    if cell.shape != (3, 3):
        raise ValueError(f"cell must be (3, 3), got {cell.shape}")
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    nnlist, nncell, bvecs, wb = kmesh_nnlist(
        kpts, reciprocal, kmesh_tol=kmesh_tol
    )
    return CovariantMMN(
        mmn=compute_Mmn(evecs, nnlist, nncell),
        nnlist=nnlist,
        nncell=nncell,
        bvecs=bvecs,
        wb=wb,
    )


def _validate_frame(mmn, nnlist, G):
    mmn = np.asarray(mmn, dtype=complex)
    nnlist = np.asarray(nnlist, dtype=int)
    G = np.asarray(G, dtype=complex)
    if mmn.ndim != 4:
        raise ValueError(f"mmn must be (nk, nn, nw, nw), got {mmn.shape}")
    nk, nn, nw, nw2 = mmn.shape
    if nw != nw2 or nnlist.shape != (nk, nn):
        raise ValueError("mmn/nnlist dimensions are inconsistent")
    if G.shape != (nk, nw, nw):
        raise ValueError(f"G must be {(nk, nw, nw)}, got {G.shape}")
    if np.any(nnlist < 0) or np.any(nnlist >= nk):
        raise ValueError("nnlist contains out-of-range k-point indices")
    return mmn, nnlist, G


def covariant_mmn_links(mmn, nnlist, G, conditioning_floor=1e-10):
    """Return Gram Q, raw links B, and dual GL-covariant links L.

    Raises when any $Q(k)=G(k)^dag G(k)$ violates the conditioning
    floor; a singular non-orthogonal frame is not a valid basis.
    """
    mmn, nnlist, G = _validate_frame(mmn, nnlist, G)
    nk, nn, nw, _ = mmn.shape
    Q = np.einsum("kma,kmb->kab", G.conj(), G)
    mineig = min(np.linalg.eigvalsh(Qk).min() for Qk in Q)
    if mineig < conditioning_floor:
        raise ValueError(
            f"non-orthogonal frame is singular: min lambda(Q)={mineig:.3e} "
            f"< {conditioning_floor:.1e}"
        )
    B = np.empty_like(mmn)
    L = np.empty_like(mmn)
    for ik in range(nk):
        for ib in range(nn):
            jk = nnlist[ik, ib]
            B[ik, ib] = G[ik].conj().T @ mmn[ik, ib] @ G[jk]
            L[ik, ib] = np.linalg.solve(Q[ik], B[ik, ib])
    return Q, B, L


def frame_projector(U, G, conditioning_floor=1e-10):
    """Return P(k)=C(k)Q(k)^-1C(k)^dag for C=UG.

    This is exactly invariant under any right GL reframe of G.
    """
    U = np.asarray(U, dtype=complex)
    G = np.asarray(G, dtype=complex)
    if U.ndim != 3 or G.ndim != 3 or U.shape[0] != G.shape[0]:
        raise ValueError("U/G must be (nk, nbasis, nw) / (nk, nw, nw)")
    if U.shape[2] != G.shape[1] or G.shape[1] != G.shape[2]:
        raise ValueError("U/G Wannier dimensions are inconsistent")
    Q = np.einsum("kma,kmb->kab", G.conj(), G)
    P = np.empty((U.shape[0], U.shape[1], U.shape[1]), dtype=complex)
    for ik in range(U.shape[0]):
        if np.linalg.eigvalsh(Q[ik]).min() < conditioning_floor:
            raise ValueError(f"singular frame at k={ik}")
        C = U[ik] @ G[ik]
        P[ik] = C @ np.linalg.solve(Q[ik], C.conj().T)
    return P


def normalized_line_links(B, Q, nnlist):
    """Return B_nn/sqrt(Q_nn(k)Q_nn(k+b)) for every line/link.

    The modulus is invariant under independent complex rescaling of
    each frame column. General GL mixing changes it by design, so use it
    as a line-localization diagnostic, not a gauge-invariant MV term.
    """
    B = np.asarray(B, dtype=complex)
    Q = np.asarray(Q, dtype=complex)
    nnlist = np.asarray(nnlist, dtype=int)
    nk, nn, nw, _ = B.shape
    if Q.shape != (nk, nw, nw) or nnlist.shape != (nk, nn):
        raise ValueError("B/Q/nnlist dimensions are inconsistent")
    z = np.empty((nk, nn, nw), dtype=complex)
    diagQ = np.diagonal(Q, axis1=1, axis2=2).real
    for ik in range(nk):
        for ib in range(nn):
            jk = nnlist[ik, ib]
            denom = np.sqrt(diagQ[ik] * diagQ[jk])
            if np.any(denom <= 0):
                raise ValueError("non-positive normalized line norm")
            z[ik, ib] = np.diag(B[ik, ib]) / denom
    return z


def electron_covariant_mmn(*_args, **_kwargs):
    """Explicit provider boundary for electron consumers.

    A correct cross-k LCAO link requires a covariant
    S(k,k+b)/dipole/MMN provider with orbital Bloch phases. Same-k SR,
    HR, and atomic centres do not determine it.
    """
    raise NotImplementedError(
        "electron covariant MMN requires a cross-k S(k,k+b), MMN, or "
        "Berry-link provider; current Siesta/Wannier90-HR consumers only "
        "retain same-k H/S and cannot synthesize it safely. Explicit "
        "provider path: read a genuine wannier90 .mmn with "
        "lawaf.io.w90.read_mmn and pass (mmn, nnlist) directly to "
        "covariant_mmn_links"
    )
