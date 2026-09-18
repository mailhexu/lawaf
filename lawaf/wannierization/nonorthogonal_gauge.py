"""Maximally localized NON-ORTHOGONAL Wannier functions via a constant
GL gauge transform.

Given the orthonormal pipeline output (any method: scdmk / projected /
mlwf, with or without disentanglement), a constant full-rank matrix
``G`` applied to the gauge, ``A'(k) = U(k) G``, produces non-orthogonal
Wannier functions whose overlap is onsite-only (``S^w = G^dag G
delta_R0``) and whose interpolated bands are unchanged at every k
(the pencil ``(G^dag H G, G^dag G)`` is a congruence).

This module chooses ``G`` to MAXIMALLY LOCALIZE the non-orthogonal
set: without the orthonormality constraint the gauge group is
GL(nwann) rather than U(nwann), and the normalized per-orbital spread

.. math::

    \\omega_n = \\sum_\\alpha \\left[
        \\frac{\\langle w_n|r_\\alpha^2|w_n\\rangle}{\\langle w_n|w_n\\rangle}
        - \\left(
            \\frac{\\langle w_n|r_\\alpha|w_n\\rangle}{\\langle w_n|w_n\\rangle}
          \\right)^2
    \\right]

can only decrease relative to the orthonormal (MV) gauge, which is a
stationary point of the unitary-restricted problem only.

Because the parent gauge is orthonormal, every moment is a fixed
Hermitian quadratic form in the columns ``g_n`` of ``G``:
``<w'_n|r_alpha|w'_n> = g_n^dag x_alpha g_n`` and
``<w'_n|r_alpha^2|w'_n> = g_n^dag y_alpha g_n`` with moment matrices
built once from ``wannR`` and the basis positions (diagonal position
approximation, the same convention as the LWF centres in
``lawaf.interfaces.phonopy.lwf.get_wannier_centers``; the norm is
exact, ``<w'_n|w'_n> = ||g_n||^2``). The Wirtinger gradient

.. math::

    \\frac{\\partial\\omega_n}{\\partial g_n^*} = \\sum_\\alpha
        \\frac{(y_\\alpha - t_{\\alpha n}) g_n
              - 2 m_{\\alpha n} (x_\\alpha - m_{\\alpha n}) g_n}
             {\\langle w_n|w_n\\rangle}

(with ``m``, ``t`` the normalized first/second moments) was verified
symbolically and against finite differences; see
``tests/test_nonorthogonal_gauge.py``.

The functional is 0-homogeneous in each column (rescaling a Wannier
function changes nothing), so the optimum carries a redundant
per-column scale; the returned ``G`` has unit-norm columns
(``<w'_n|w'_n> = 1``) and generally non-zero cross overlaps
``(G^dag G)_{mn}``.

Reference: specs/research/2026-09-18-nonorthogonal-mlwf.md (formulation
(b) restricted to constant G; the two-step baseline of formulation (a)).
"""

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "position_moment_matrices",
    "nonorthogonal_spread",
    "nonorthogonal_spread_gradient",
    "optimize_nonorthogonal_gauge",
    "apply_gauge_transform",
]


def position_moment_matrices(wannR, Rlist, Rdeg, positions):
    """Hermitian first/second position-moment matrices of the
    orthonormal Wannier basis.

    :param wannR: (nR, nbasis, nwann) coefficients of the ORTHONORMAL
        gauge in a localized parent basis;
    :param Rlist: (nR, 3) integer lattice vectors;
    :param Rdeg: (nR,) image degeneracies (W90 convention; same use as
        ``get_wannier_centers``);
    :param positions: (nbasis, 3) Cartesian positions of the parent
        basis functions;
    :return: ``(x, y)`` — lists of 3 Hermitian (nwann, nwann) matrices,
        ``x[alpha]_{mn} = <w_m|r_alpha|w_n>``, ``y[alpha]_{mn} =
        <w_m|r_alpha^2|w_n>`` in the diagonal position approximation.
    """
    wannR = np.asarray(wannR)
    Rlist = np.asarray(Rlist, dtype=float)
    positions = np.asarray(positions, dtype=float)
    Rdeg = np.ones(len(Rlist)) if Rdeg is None else np.asarray(Rdeg, dtype=float)
    nR, nbasis, nwann = wannR.shape
    x = [np.zeros((nwann, nwann), dtype=complex) for _ in range(3)]
    y = [np.zeros((nwann, nwann), dtype=complex) for _ in range(3)]
    # column weights: weight_{m,n}(iR) = c*_m(iR) c_n(iR) * Rdeg[iR]
    for iR in range(nR):
        c = wannR[iR]  # (nbasis, nwann)
        w = (c.conj()[:, :, None] * c[:, None, :]) * Rdeg[iR]  # (nb, nw, nw)
        pos = Rlist[iR][None, :] + positions  # (nbasis, 3)
        for alpha in range(3):
            pa = pos[:, alpha][:, None, None]  # (nbasis, 1, 1)
            x[alpha] += (w * pa).sum(axis=0)
            y[alpha] += (w * pa**2).sum(axis=0)
    for alpha in range(3):
        # exact by construction; enforce round-off Hermiticity
        x[alpha] = 0.5 * (x[alpha] + x[alpha].conj().T)
        y[alpha] = 0.5 * (y[alpha] + y[alpha].conj().T)
    return x, y


def nonorthogonal_spread(G, x, y, per_orbital=False):
    """Total normalized spread Omega(G) = sum_n omega_n (and optionally
    the per-orbital values) of the GL-combined Wannier functions."""
    G = np.asarray(G, dtype=complex)
    nw = G.shape[1]
    norms2 = np.einsum("an,an->n", G.conj(), G).real
    omega = np.zeros(nw)
    for alpha in range(3):
        m = np.einsum("an,ab,bn->n", G.conj(), x[alpha], G).real / norms2
        t = np.einsum("an,ab,bn->n", G.conj(), y[alpha], G).real / norms2
        omega += t - m**2
    return omega if per_orbital else omega.sum()


def nonorthogonal_spread_gradient(G, x, y):
    """Wirtinger gradient dOmega/dG* (nwann, nwann columns g_n)."""
    G = np.asarray(G, dtype=complex)
    norms2 = np.einsum("an,an->n", G.conj(), G).real
    grad = np.zeros_like(G)
    for alpha in range(3):
        xg = x[alpha] @ G
        yg = y[alpha] @ G
        m = np.einsum("an,an->n", G.conj(), xg).real / norms2
        t = np.einsum("an,an->n", G.conj(), yg).real / norms2
        grad += ((yg - t[None, :] * G) - 2.0 * m[None, :] * (xg - m[None, :] * G)) / norms2[None, :]
    return grad


def _normalize_columns(G):
    norms = np.linalg.norm(G, axis=0)
    if np.any(norms < 1e-14):
        raise ValueError("gauge column collapsed to zero norm")
    return G / norms[None, :]


def optimize_nonorthogonal_gauge(
    wannR, Rlist, Rdeg, positions, G0=None, maxiter=500, tol=1e-12,
    barrier_weight=1e-3, verbose=False,
):
    """Choose the constant GL factor G that minimizes the normalized
    per-orbital spread of the non-orthogonal Wannier functions,
    subject to a conditioning barrier (see below) that keeps G
    invertible: without it the spread infimum over GL is degenerate
    (columns collapse onto the single best-localized function).

    :param wannR: (nR, nbasis, nwann) coefficients of the ORTHONORMAL
        pipeline result (any method; the spread can only improve on it);
    :param Rlist: (nR, 3) integer lattice vectors of ``wannR``;
    :param Rdeg: (nR,) image degeneracies, or None for all-ones;
    :param positions: (nbasis, 3) Cartesian positions of the parent
        basis functions;
    :param G0: initial guess (default: identity, the orthonormal gauge);
    :return: ``(G, result)`` — G with unit-norm columns (so
        ``<w'_n|w'_n> = 1`` and ``SwannR = G^dag G`` at R=0 only), and
        the scipy OptimizeResult.

    Notes:
    - the optimum is determined up to per-column phases and a final
      unitary freedom of equal-spread sets; only the spread value and
      the resulting model are meaningful;
    - a symmetry-adapted orthonormal gauge generally loses its irrep
      labels under a generic optimal G (reducible directions mix);
      pin such structure by restricting G0/block structure yourself;
    - ``barrier_weight`` scales the ``-mu log det(G^dag G)`` GL
      conditioning barrier (``mu = barrier_weight * max(spread(G0), 1)``);
      it is free at the orthonormal start (det = 1) and forbids
      singular gauges; raise it if the optimized set is too
      ill-conditioned, lower it to chase smaller spreads;
    - the tiny ``reg`` term below lifts the exactly-flat per-column
      rescaling directions (0-homogeneity) without biasing the spread.
    """
    x, y = position_moment_matrices(wannR, Rlist, Rdeg, positions)
    nwann = wannR.shape[2]
    G0 = np.eye(nwann, dtype=complex) if G0 is None else np.asarray(G0, dtype=complex)
    if G0.shape != (nwann, nwann):
        raise ValueError(f"G0 must be {(nwann, nwann)}, got {G0.shape}")

    def pack(G):
        return np.concatenate([G.real.ravel(), G.imag.ravel()])

    def unpack(p):
        return (p[: nwann * nwann].reshape(nwann, nwann)
                + 1j * p[nwann * nwann:].reshape(nwann, nwann))

    reg = 1e-10

    # Conditioning barrier: the unconstrained infimum of the spread over
    # GL is DEGENERATE -- columns can collapse onto the single
    # best-localized function (rank 1, spread -> its value, model
    # useless). -mu log det(G^dag G) is the proper GL barrier: it
    # forbids singular G, costs nothing at the orthonormal start
    # (det = 1), and its Wirtinger gradient is -mu G (G^dag G)^{-1}.
    mu = barrier_weight * max(
        nonorthogonal_spread(_normalize_columns(G0), x, y), 1.0
    )

    def fun(p):
        G = unpack(p)
        val = nonorthogonal_spread(G, x, y)
        S = G.conj().T @ G
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e30
        val += reg * float(np.sum((np.einsum("an,an->n", G.conj(), G).real - 1.0) ** 2))
        val -= mu * float(logdet)
        return float(val)

    def jac(p):
        G = unpack(p)
        g = nonorthogonal_spread_gradient(G, x, y)
        S = G.conj().T @ G
        g = g - mu * G @ np.linalg.inv(S)
        col = 2.0 * reg * (G * (np.einsum("an,an->n", G.conj(), G).real - 1.0)[None, :])
        c = g + col
        # f real with c = df/dG*: df = 2 Re[c dG]; parametrizing
        # dG = dRe + i dIm gives df/dRe = 2 Re c, df/dIm = +2 Im c
        return np.concatenate([(2 * c.real).ravel(), (2 * c.imag).ravel()])

    def run(start):
        return minimize(
            fun, pack(start), jac=jac, method="L-BFGS-B",
            options={"maxiter": maxiter, "gtol": tol, "ftol": tol},
        )

    # The orthonormal (MV) gauge is a STATIONARY point of the GL
    # functional too (zero gradient along unitary directions by MV
    # optimality, and sometimes along all of them, e.g. symmetric
    # two-site models), so a plain start at G0 can stall at the
    # orthonormal spread. Deterministic multi-start: G0 plus small
    # seeded tilts; keep the best optimum.
    starts = [_normalize_columns(G0)]
    rng = np.random.default_rng(20260918)
    for _ in range(3):
        tilt = np.eye(nwann) + 0.05 * (
            rng.standard_normal((nwann, nwann))
            + 1j * rng.standard_normal((nwann, nwann))
        )
        starts.append(_normalize_columns(G0 @ tilt))
    best = None
    for start in starts:
        r = run(start)
        if best is None or r.fun < best.fun:
            best = r
    res = best

    G = _normalize_columns(unpack(res.x))
    if verbose:
        omega0 = nonorthogonal_spread(np.eye(nwann), x, y, per_orbital=True)
        omega1 = nonorthogonal_spread(G, x, y, per_orbital=True)
        print(f"non-orthogonal gauge: spread {omega0.sum():.6f} -> "
              f"{omega1.sum():.6f} (per orbital "
              f"{np.array2string(omega0, precision=4)} -> "
              f"{np.array2string(omega1, precision=4)}), "
              f"{res.message}")
    return G, res


def apply_gauge_transform(lwf, G):
    """Apply a constant GL factor to a finished (orthonormal) Wannier
    model, returning the non-orthogonal result.

    Works duck-typed on the phonon ``LWF`` and on HamiltonIO's
    ``LawafHamiltonian`` (EWF): transforms ``HwannR/HR_total -> G^dag H
    G``, ``wannR -> wannR G``, and installs the onsite-only overlap
    ``SwannR = G^dag G`` at R=0. ``NACLWF`` is refused (the NAC split
    is defined for orthonormal LWFs). The input object is not modified.
    """
    G = np.asarray(G, dtype=complex)
    if getattr(lwf, "nac", False):
        raise NotImplementedError(
            "apply_gauge_transform: NACLWF (NAC split) is defined for "
            "orthonormal LWFs only"
        )
    if getattr(lwf, "SwannR", None) is not None:
        raise ValueError(
            "apply_gauge_transform expects an orthonormal model "
            "(SwannR is None)"
        )
    nwann = lwf.wannR.shape[2]
    if G.shape != (nwann, nwann):
        raise ValueError(f"G must be {(nwann, nwann)}, got {G.shape}")
    Rlist = np.asarray(lwf.Rlist)
    i0 = np.where(np.all(Rlist == 0, axis=1))[0]
    if len(i0) != 1:
        raise ValueError(f"need exactly one R=0 entry in Rlist, found {len(i0)}")
    i0 = int(i0[0])
    S = np.zeros_like(lwf.wannR[:1, :nwann, :nwann]).repeat(len(Rlist), axis=0)
    S[i0] = G.conj().T @ G
    if hasattr(lwf, "HR_total"):  # phonon LWF
        from lawaf.interfaces.phonopy.lwf import LWF

        return LWF(
            factor=lwf.factor,
            Rlist=lwf.Rlist,
            Rdeg=lwf.Rdeg,
            wannR=lwf.wannR @ G,
            HR_total=G.conj().T @ lwf.HR_total @ G,
            SwannR=S,
            kpts=lwf.kpts,
            kweights=lwf.kweights,
            atoms=lwf.atoms,
        )
    from HamiltonIO.lawaf import LawafHamiltonian

    return LawafHamiltonian(
        wannR=lwf.wannR @ G,
        HwannR=G.conj().T @ lwf.HwannR @ G,
        SwannR=S,
        Rlist=lwf.Rlist,
        Rdeg=lwf.Rdeg,
        atoms=lwf.atoms,
        wann_names=getattr(lwf, "wann_names", None),
        kpts=getattr(lwf, "kpts", None),
        kweights=getattr(lwf, "kweights", None),
        is_orthogonal=False,
    )
