"""Star-covariant constrained gauge for projected LWFs (story-018, FR-004/015, ADR-003).

Given a downfolder/builder holding a projected gauge ``Amn(q)`` (the
``(nband, nwann)`` Bloch-to-Wannier matrices) and a
:class:`~lawaf.anharmonic.compatibility.RepresentationDeclaration` naming a
Wyckoff orbit and the site irrep its Wannier triplet carries, this module
re-imposes the little-group gauge constraint at every irreducible q of the
downfold mesh, propagates the gauge over each star (never re-optimizing star
arms), ties time-reversal partners, and reports dual spreads + residuals.

Constraint algebra (all identities sympy-verified in
``docs/derivations/story018_gauge_sympy.py``)
---------------------------------------------------------------------------
Let ``M_h = psi(q)^dag S_g(h, q) psi(q)`` be the little-group representation
in the retained window (coefficient space; ``S_g`` from
:mod:`lawaf.anharmonic.representation`, whose transport law and group
composition are story-014-validated).  The gauge constraint at q is the
intertwining condition

    ``M_h U(q) = U(q) D_W(h)``   for every h in the little group of q,

equivalently ``U(q) = M_h U(q) D_W(h)^dag``.  Two exact constructions act on
it:

1. **Reynolds (group-averaging) projection** ``P(U) = (1/|LG|) sum_h M_h^dag
   U D_W(h)^dag``.  Because ``M`` and ``D_W`` are homomorphisms with the same
   composition ordering, ``M_{h0} P(U) D_{W}(h0)^dag = P(U)``: the image is
   the constraint manifold, P is idempotent, and manifold members are fixed
   points (manifold preservation).  P alone does not preserve column
   orthonormality; following it by the polar factor (the orthogonal-
   Procrustes/QR retraction: for ``Z = W S Y^dag``, ``W Y^dag`` is the
   closest orthonormal-column matrix to Z in Frobenius norm) gives the
   story's reimposition map ``U -> polar(P(U))``.  Every constrained gauge
   is a fixed point of this map; the map is used as the polish step.

2. **Wigner-channel projection** (production path, single shot + exact).
   For a seed coefficient column ``u`` (a column of the unconstrained update)
   the matrix-unit channels

       ``w_a = (d / |LG|) sum_h conj(tau(h)[a, b0]) M_h u``   (b0 = 0)

   transform as ``M_g W = W tau(g)`` with ``W = [w_0 ... w_{d-1}]``: the
   projector algebra ``M_g P_{ab} = sum_c tau(g)_{ca} P_{cb}`` is verified
   symbolically.  By Schur's lemma the Gram matrix ``G = W^dag W`` of the
   channels is proportional to the identity whenever the seed's span carries
   the declared irrep; the Loewdin/QR-style normalization ``W G^{-1/2}``
   (Cholesky) then yields a gauge that satisfies the constraint to machine
   precision.  ``lambda_min(G)`` doubles as the near-degeneracy monitor: a
   seed with no overlap on the declared channel (window does not carry the
   declaration at q) drives it below ``gauge_tolerances["conditioning"]``
   and raises.

The Wannier-space representation ``D_W = tau`` is the declared site irrep as
an explicit matrix rep, constructed deterministically from the site character
table (see :func:`site_irrep_matrices`): the left regular representation of
the site group is block-diagonalized by joint class-sum eigenspaces, the
declared isotypic component is identified by its characters, a generic
element of the commutant is diagonalized to split the multiplicity space, and
the irrep matrices are read off as the restriction of each group image to one
multiplicity block.  Group closure is verified numerically at construction
and the extraction identities are verified exactly (sympy, D3) in the
derivation script.

Star propagation
    For a star arm ``q' = g.q0`` the gauge is never re-optimized:
    ``Amn(q') = X_g Amn(q0) D_W(g)^dag`` with the story-014 transport matrix
    ``X_g = psi(q')^dag S_g(q0) psi(q0)``.  Time-reversal partners (real
    force constants: ``psi(-q) = conj(psi(q))`` up to the eigenvector gauge)
    are tied, never optimized: ``Amn(-q) = Y Amn(q)^*`` with
    ``Y = psi(-q)^dag psi(q)^*``.  At self-reciprocal points (``-q = q mod
    G``) the tie is vacuous and the little-group constraint alone fixes the
    gauge.

Entry points
    :func:`constrained_localize` runs the full pipeline on a downfolder (or
    bare builder) and returns the rebuilt :class:`~lawaf.interfaces.phonopy.lwf.LWF`;
    :func:`constrain_builder_amn` is the in-place core used by the
    ``ProjectedWannierizer`` parameter hook (``WannierParams.
    symmetry_adapted_gauge``).
"""

from __future__ import annotations

import numpy as np

from .compatibility import (
    RepresentationDeclaration,
    _GroupAlgebra,
    character_table,
    little_group,
)
from .representation import (
    WindowBands,
    _window_qkey,
    _window_qname,
    build_space_group_action,
)

__all__ = [
    "constrained_localize",
    "constrain_builder_amn",
    "site_irrep_matrices",
    "constrain_amn_one_q",
    "assert_projective_closure",
    "check_window_irreps",
    "DEFAULT_GAUGE_TOLERANCES",
]

#: default tolerances (overridable via ``WannierParams.gauge_tolerances``)
DEFAULT_GAUGE_TOLERANCES = {
    "constraint_residual": 1e-10,   # reported/final eps(q) must not exceed
    "symmetrize_tol": 1e-14,        # polish fixed-point step size
    "polish_iter": 5,               # Reynolds+polar polish sweeps
    "conditioning": 1e-8,           # min channel Gram eigenvalue / svd(D_W)
    "content_tol": 0.1,             # window-multiplicity integer tolerance
    "kpt_tol": 1e-6,                # mesh-point matching
}


# ----------------------------------------------------------------------
# declared site irrep as explicit matrices
# ----------------------------------------------------------------------
def _split_isotypic(class_sums, basis):
    """Joint eigenspace refinement of commuting class sums, keeping bases."""
    sym = basis.T @ class_sums[0] @ basis
    w, U = np.linalg.eigh(sym)
    blocks = []
    for val in np.unique(np.round(w, 7)):
        sel = np.abs(w - val) <= 1e-7
        blocks.append(basis @ U[:, sel])
    if len(class_sums) == 1:
        return blocks
    out = []
    for blk in blocks:
        out.extend(_split_isotypic(class_sums[1:], blk))
    return out


def site_irrep_matrices(sga, site_ops, name: str) -> dict:
    """Explicit matrix representation ``tau`` of a declared site irrep.

    Deterministic construction from the site character table: block-diagonal
    the left regular representation of ``site_ops`` by joint class-sum
    eigenspaces, select the isotypic component whose characters match the
    declared irrep ``name``, split its multiplicity space by diagonalizing a
    generic commutant element, and restrict each ``tau``-block of the regular
    rep to one multiplicity carrier.  The extraction identities (homomorphism
    closure, Schur isotropy) are verified exactly in
    ``docs/derivations/story018_gauge_sympy.py``; closure is re-verified
    numerically here.

    Returns ``{op_index: (d, d) complex array}`` (unitary, closed under
    composition).
    """
    site_ops = [int(g) for g in site_ops]
    ct = character_table(sga, site_ops)
    if name not in ct.names:
        raise ValueError(
            f"unknown site irrep {name!r}; available: {', '.join(ct.names)}"
        )
    j = ct.names.index(name)
    d = ct.dims[j]

    alg = _GroupAlgebra(sga, site_ops)
    ops = list(ct.ops)
    n = len(ops)
    idx = {g: i for i, g in enumerate(ops)}
    # left regular representation (same convention as character_table)
    reg = {}
    for g in ops:
        P = np.zeros((n, n))
        for jj, h in enumerate(ops):
            P[idx[alg.compose(g, h)], jj] = 1.0
        reg[g] = P
    class_sums = [sum(reg[g] for g in cl) for cl in ct.classes]

    comp = None
    for cand in _split_isotypic(class_sums, np.eye(n)):
        lam = np.array(
            [np.trace(cand.T @ M @ cand).real / cand.shape[1]
             for M in class_sums]
        )
        dd = np.sqrt(n / np.sum(np.abs(lam) ** 2 / ct.class_sizes))
        if np.allclose(lam * dd / ct.class_sizes, ct.chars[j], atol=1e-6):
            comp = cand
            break
    if comp is None:  # pragma: no cover - character table is complete
        raise ValueError(f"isotypic component of {name!r} not found")

    m = comp.shape[1] // d
    Dg = {g: comp.T @ reg[g] @ comp for g in ops}
    md = d * m
    # commutant: X D_g = D_g X  ->  (D_g^T (x) I - I (x) D_g) vec(X) = 0
    rows = np.zeros((len(ops) * md * md, md * md))
    r = 0
    eye = np.eye(md)
    for g in ops:
        rows[r:r + md * md] = np.kron(Dg[g].T, eye) - np.kron(eye, Dg[g])
        r += md * md
    _, s, vt = np.linalg.svd(rows)
    smax = s[0] if s.size else 0.0
    tol = max(rows.shape) * np.finfo(float).eps * (smax if smax > 0 else 1.0)
    rank = int((s > tol).sum())
    null = vt[rank:].T
    if null.shape[1] < m * m - 1:  # pragma: no cover - algebra guarantees m^2
        raise ValueError("commutant dimension inconsistent with irrep data")

    # generic Hermitian commutant element; its eigenspaces are the
    # multiplicity carriers (d-dimensional each)
    rng = np.random.default_rng(20260829)
    coeffs = rng.normal(size=null.shape[1])
    X0 = (null * coeffs).sum(axis=1).reshape(md, md)
    evals, evecs = np.linalg.eigh(0.5 * (X0 + X0.conj().T))
    order = np.argsort(evals)
    groups, cur = [], [order[0]]
    scale = max(1.0, float(np.abs(evals).max()))
    for k in order[1:]:
        if abs(evals[k] - evals[cur[-1]]) < 1e-7 * scale:
            cur.append(k)
        else:
            groups.append(cur)
            cur = [k]
    groups.append(cur)
    carrier = None
    for grp in groups:
        if len(grp) == d:
            carrier = evecs[:, grp]
            break
    if carrier is None:  # pragma: no cover - generic element splits fully
        raise ValueError("no d-dimensional multiplicity carrier found")

    tau = {g: carrier.conj().T @ Dg[g] @ carrier for g in ops}

    # numerical closure verification (exact sympy version in the derivation)
    worst = 0.0
    for g in ops:
        for h in ops:
            worst = max(
                worst,
                np.abs(tau[g] @ tau[h] - tau[alg.compose(g, h)]).max(),
            )
    if worst > 1e-10:
        raise ValueError(
            f"extracted {name!r} matrices fail group closure "
            f"(max residual {worst:.2e})"
        )
    return tau


def assert_projective_closure(representation, compose, *, tol=1e-10):
    """Validate ``D(g)D(h)=omega(g,h)D(gh)`` and return ``omega``.

    Zone-boundary Bloch frames may realize a little group only up to scalar
    phases.  The constraint only pairs representations with the *same*
    factor system, so those phases cancel in ``M U D_W^dag``; this checker
    deliberately accepts fractional characters instead of imposing ordinary
    group closure.
    """
    rep = {int(g): np.asarray(matrix, dtype=complex) for g, matrix in representation.items()}
    if not rep:
        raise ValueError("projective closure needs at least one operation")
    dim = next(iter(rep.values())).shape[0]
    if any(matrix.shape != (dim, dim) for matrix in rep.values()):
        raise ValueError("projective representation matrices must be square and same-sized")
    factors = {}
    for g, dg in rep.items():
        for h, dh in rep.items():
            gh = int(compose(g, h))
            if gh not in rep:
                raise ValueError(f"projective product ({g}, {h})={gh} is absent")
            target = rep[gh]
            product = dg @ dh
            factor = np.vdot(target, product) / np.vdot(target, target)
            residual = float(np.abs(product - factor * target).max())
            if residual > tol or abs(abs(factor) - 1.0) > tol:
                raise ValueError(
                    f"projective closure failed for ({g}, {h}): "
                    f"residual {residual:.2e}, factor {factor:.6g}"
                )
            factors[(g, h)] = complex(factor)
    return factors


def check_window_irreps(declaration, window_bands):
    """Check optional per-anchor declarations against validated window data."""
    decl = _resolve_declaration(declaration)
    if decl.window_irreps is None:
        return
    if not isinstance(window_bands, WindowBands):
        raise TypeError("window_irreps requires validated WindowBands")
    for qpoint, declared in decl.window_irreps.items():
        qkey = _window_qkey(qpoint)
        if qkey not in window_bands.legality:
            raise ValueError(
                f"window_irreps declares {_window_qname(qkey)} but no validated "
                "window exists there"
            )
        actual = sorted(
            irrep
            for block in window_bands.legality[qkey]
            for irrep in block.irreps
        )
        expected = sorted(str(irrep) for irrep in declared)
        if expected != actual:
            raise ValueError(
                f"window_irreps mismatch at {_window_qname(qkey)}: "
                f"declared {expected}, actual {actual}"
            )


# ----------------------------------------------------------------------
# per-q numerical core (phonon-free; unit-tested directly)
# ----------------------------------------------------------------------
def constrain_amn_one_q(M, DW, U0, *, copies=1, tolerances=None, tie=None):
    """Constrain one ``Amn`` at a single q given explicit window reps.

    :param M: ``{h: (nband, nband)}`` little-group window representation
        ``M_h = psi^dag S_h psi`` (closed op set).
    :param DW: ``{h: (nwann, nwann)}`` declared Wannier-space rep; must be a
        rep of the same op set (multi-copy declarations pass the blockwise
        direct sum).
    :param U0: ``(nband, nwann)`` unconstrained gauge (orthonormal columns).
    :param copies: nwann // dim(tau) block count.
    :returns: ``(U, info)`` with the constrained gauge and a diagnostics dict
        (``eps``, ``gram_smin``, ``content_multiplicity``, ``seed_column``,
        ``dw_smin``).

    Raises :class:`ValueError` (naming the band window through ``info`` /
    message) when the window does not carry the declared content or the
    constraint is near-degenerate.
    """
    tol = dict(DEFAULT_GAUGE_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    M = {int(h): np.asarray(Mh) for h, Mh in M.items()}
    DW = {int(h): np.asarray(Dh) for h, Dh in DW.items()}
    ops = sorted(M)
    if sorted(DW) != ops:
        raise ValueError("M and DW must be representations of the same ops")
    U0 = np.asarray(U0)
    nband, nwann = U0.shape

    # window content: multiplicity of tr(DW) in tr(M)
    chi_m = np.array([np.trace(M[h]) for h in ops])
    chi_w = np.array([np.trace(DW[h]) for h in ops])
    n_ops = len(ops)
    n_tau = float((chi_m * chi_w.conj()).sum().real / n_ops)
    n_tau_int = int(round(n_tau))
    dw_smin = min(
        float(np.linalg.svd(DW[h], compute_uv=False)[-1]) for h in ops
    )
    if abs(n_tau - n_tau_int) > tol["content_tol"] or n_tau_int < 1:
        raise ValueError(
            "window does not carry the declared representation: multiplicity "
            f"{n_tau:.4f} (bands nband={nband}, nwann={nwann}, "
            f"smallest svd(D_W)={dw_smin:.2e}); the gauge constraint is "
            "unsolvable at this q"
        )
    if dw_smin < tol["conditioning"]:
        raise ValueError(
            f"near-degenerate D_W: smallest singular value {dw_smin:.2e} "
            f"below conditioning tolerance {tol['conditioning']:.1e} "
            f"(nband={nband}, nwann={nwann})"
        )

    dim = nwann // copies
    if nwann != dim * copies:
        raise ValueError(
            f"nwann={nwann} is not dim(tau)={dim} x copies={copies}"
        )

    # Seed candidates: every unconstrained column plus deterministic
    # generic combinations.  For REDUCIBLE tau|LG (e.g. T1u|D4h = A2u+Eu)
    # a single fixed column need not be cyclic for the matrix-unit
    # algebra, so the channel Gram may be singular; a generic combination
    # of the seed columns is (the channels of a generic seed span the
    # declared isotypic component).  The candidate with the largest
    # lambda_min(G) wins; if the best is still below the conditioning
    # tolerance the window is near-degenerate -> fail fast.
    nseed = U0.shape[1]
    rng = np.random.default_rng(20260829)
    combos = [np.ones(nseed) / np.sqrt(nseed)]
    if nseed >= 2:
        v = rng.normal(size=nseed)
        combos.append(v / np.linalg.norm(v))
    candidates = [U0[:, j] for j in range(nseed)]
    candidates += [U0 @ c for c in combos]
    # Fast path: a seed already (numerically) on the constraint manifold.
    # The Reynolds+polar polish below has manifold members as exact fixed
    # points, so no channel extraction is needed.  This also sidesteps a
    # structural degeneracy of the channel construction: for a MANIFESTLY
    # reducible DW (e.g. the Cartesian signed-permutation T1u restricted
    # to a D4h little group = A2u + Eu in block form) some matrix-unit
    # elements DW[h][a, 0] vanish for all h, the corresponding channels
    # are identically zero for EVERY seed, and the channel Gram is
    # singular by structure -- even though the constraint manifold is
    # perfectly regular.  (The declaration-extracted dense convention
    # mixes the blocks and never hits this; the explicit Cartesian
    # convention does at the M points of a cubic mesh.)
    eps_seed = max(
        float(np.abs(U0 - M[h] @ U0 @ DW[h].conj().T).max()) for h in ops
    )
    # With ``tie`` (Y = psi(q)^dag psi(q)^* at a self-reciprocal q, REAL
    # DW) the joint constraint {U : M_h U = U D_W(h)} AND {U = Y U^*} is
    # solved EXACTLY in one shot: T and the Reynolds projector commute
    # (M_h Y = Y M_h^* holds because S_h(q) is real at 2q = 0 mod 1, and
    # D_W real), so J = (P(U0) + T P(U0))/2 is the exact projection onto
    # the joint space; J is an intertwiner and Y-real, its Gram G = J^dag J
    # is REAL (G = G^* follows from J = Y J^*) and commutes with D_W
    # (D_W^dag G D_W = J^dag M^dag M J = G), hence U = J G^{-1/2} is the
    # orthonormal-column joint fixed point -- the same Loewdin argument as
    # the Wigner-channel construction.  lambda_min(G) doubles as the
    # near-degeneracy monitor.
    if tie is not None:
        P = sum(
            M[h] @ U0 @ DW[h].conj().T for h in ops
        ) / n_ops
        J = 0.5 * (P + tie @ P.conj())
        G = J.conj().T @ J
        lam, Vg = np.linalg.eigh(G.real)
        gram_min = float(lam.real.min())
        if gram_min < tol["conditioning"]:
            raise ValueError(
                "joint constraint channel degenerate: the tie-projected "
                f"seed has lambda_min(G)={gram_min:.1e} below "
                f"{tol['conditioning']:.1e} (nband={nband}, nwann={nwann})"
            )
        Gm = (Vg * (1.0 / np.sqrt(lam.real))) @ Vg.T
        U = J @ Gm
        seed_col = -2
    elif (
        eps_seed <= tol["constraint_residual"]
        and np.linalg.eigvalsh(
            0.5 * (U0.conj().T @ U0 + (U0.conj().T @ U0).conj().T)
        ).real.min()
        >= tol["conditioning"]
    ):
        # Fast path: a seed already (numerically) on the constraint
        # manifold AND well conditioned -- the Reynolds+polar polish has
        # manifold members as exact fixed points, so no channel
        # extraction is needed.  A rank-deficient manifold seed (e.g.
        # duplicated columns) must NOT take this exit: it would report a
        # zero residual while carrying a degenerate gauge.
        U = U0.copy()
        gram_min = float("nan")
        seed_col = -1
    else:
        best = None
        for ci, _u in enumerate(candidates):
            blocks = []
            gram_min = np.inf
            gram_ok = True
            for c in range(copies):
                # one seed per block (identical seeds would duplicate columns)
                uc = candidates[min(ci + c, len(candidates) - 1)]
                cols = []
                for a in range(dim):
                    acc = np.zeros(nband, dtype=complex)
                    for h in ops:
                        acc += np.conj(DW[h][a, 0]) * (M[h] @ uc)
                    cols.append(acc * (dim / n_ops))
                W = np.stack(cols, axis=1)
                G = W.conj().T @ W
                G = 0.5 * (G + G.conj().T)
                lam, V = np.linalg.eigh(G)
                lam_min = float(lam.real.min())
                gram_min = min(gram_min, lam_min)
                if lam_min < tol["conditioning"]:
                    gram_ok = False
                    break
                # Symmetric G^-1/2 commutes with the represented action;
                # Cholesky does not preserve this equivariance.
                Gm = (V * (1.0 / np.sqrt(lam.real))) @ V.conj().T
                blocks.append(W @ Gm)
            if gram_ok:
                cand = (np.concatenate(blocks, axis=1), gram_min, ci)
                if best is None or cand[1] > best[1]:
                    best = cand
        if best is None:
            # A reducible D_W may have a structural zero in every
            # matrix-unit channel rooted at column zero (E+A is the minimal
            # case), even though its full Reynolds projection is perfectly
            # regular.  The normalized G^-1/2 construction below is the
            # same story-018 Reynolds/polar fixed point, applied directly
            # to all Wannier columns rather than to one irrep channel.
            J = sum(M[h] @ U0 @ DW[h].conj().T for h in ops) / n_ops
            G = J.conj().T @ J
            G = 0.5 * (G + G.conj().T)
            lam, V = np.linalg.eigh(G)
            gram_min = float(lam.real.min())
            if gram_min >= tol["conditioning"]:
                Gm = (V * (1.0 / np.sqrt(lam.real))) @ V.conj().T
                best = (J @ Gm, gram_min, -3)
            else:
                raise ValueError(
                    "constraint channel degenerate: no unconstrained seed "
                    "column or combination has overlap with the declared "
                    "irrep channel "
                    f"(lambda_min < {tol['conditioning']:.1e}); the window is "
                    f"near-degenerate for this declaration (nband={nband}, "
                    f"nwann={nwann})"
                )
        Wn, gram_min, seed_col = best
        U = Wn
        # Reynolds projection onto the constraint manifold (sympy-verified:
        # P(U) = (1/N) sum_h M_h U D_W(h)^dag is the projector onto
        # {U : M_h U = U D_W(h)}), followed by the orthogonal-Procrustes/
        # polar retraction (restores orthonormal columns; constrained
        # gauges are fixed points of the composed map).
        for _ in range(int(tol["polish_iter"])):
            Z = sum(M[h] @ U @ DW[h].conj().T for h in ops) / n_ops
            vs, _ss, vt = np.linalg.svd(Z, full_matrices=False)
            U_new = vs @ vt
            if np.abs(U_new - U).max() < tol["symmetrize_tol"]:
                U = U_new
                break
            U = U_new

    eps = max(
        float(np.abs(U - M[h] @ U @ DW[h].conj().T).max()) for h in ops
    )
    if tie is not None:
        eps = max(eps, float(np.abs(U - tie @ U.conj()).max()))
    ortho = float(np.abs(U.conj().T @ U - np.eye(nwann)).max())

    info = {
        "eps": eps,
        "gram_smin": gram_min,
        "content_multiplicity": n_tau_int,
        "seed_column": seed_col,
        "dw_smin": dw_smin,
        "ortho": ortho,
    }
    if eps > tol["constraint_residual"]:
        raise ValueError(
            f"constraint residual eps={eps:.2e} exceeds tolerance "
            f"{tol['constraint_residual']:.1e} (nband={nband}, "
            f"nwann={nwann}) - window/declaration incompatible at this q"
        )
    return U, info



# ----------------------------------------------------------------------
# builder pipeline
# ----------------------------------------------------------------------
def _resolve_declaration(declaration):
    if isinstance(declaration, RepresentationDeclaration):
        return declaration
    if isinstance(declaration, dict):
        return RepresentationDeclaration(**declaration)
    raise ValueError(
        "declaration must be a RepresentationDeclaration or its dict form"
    )

def _pure_point_matrices(sga):
    """Defect-free (pure point-operation) representation matrices.

    ``S'_g[3 sigma + i, 3 kappa + j] = R_g[i, j]`` -- the Cartesian
    rotation acting on displacement components, scattering atom kappa to
    its image sigma(kappa), with NO Bloch defect phases.  For a
    symmorphic crystal these are a genuine representation of the space
    group at every q (verified on the fixture to 1e-15 for all ops).
    """
    n = sga.n_atoms
    out = {}
    for g in range(sga.n_ops):
        R = np.asarray(sga.cart_rotations[g], dtype=float)
        sig = np.asarray(sga.atom_maps[g], int)
        M = np.zeros((3 * n, 3 * n))
        for a in range(n):
            M[3 * sig[a]:3 * sig[a] + 3, 3 * a:3 * a + 3] = R
        out[int(g)] = M
    return out



def _frame_phase(k, frac_r):
    """Diagonal basis rephasing Phi(k) = diag_a exp(-2 pi i k . r_a)."""
    ph = np.exp(
        -2j
        * np.pi
        * (np.asarray(frac_r, dtype=float) @ np.asarray(k, dtype=float))
    )
    return np.diag(np.repeat(ph, 3))


def _resolve_frac_positions(builder, downfolder, frac_positions):
    """Fractional atom positions in sga atom order, or None.

    Accepts an explicit (natom, 3) array; falls back to
    ``downfolder.atoms`` (ase Atoms).
    """
    if frac_positions is not None:
        r = np.mod(
            np.asarray(frac_positions, dtype=float).reshape(-1, 3), 1.0
        )
        return r
    atoms = getattr(downfolder, "atoms", None)
    if atoms is not None:
        return np.mod(atoms.get_scaled_positions(), 1.0)
    r = getattr(builder, "frac_positions", None)
    if r is not None:
        return np.mod(np.asarray(r, dtype=float).reshape(-1, 3), 1.0)
    return None


def _resolve_sga(downfolder, builder, sga):
    if sga is not None:
        return sga
    # builder-side hints (settable by any driver; the projectedWF hook
    # cannot see the owning downfolder, so drivers that want the hook to
    # route inside downfold() attach the sga to the builder):
    hint = getattr(builder, "gauge_sga", None)
    if hint is not None:
        return hint
    hint = getattr(builder, "atoms", None)
    if hint is not None:
        return build_space_group_action(hint)
    model = getattr(downfolder, "model", None)
    phonon = getattr(model, "phonon", None)
    if phonon is not None:
        return build_space_group_action(phonon)
    atoms = getattr(downfolder, "atoms", None)
    if atoms is not None:
        return build_space_group_action(atoms)
    raise ValueError(
        "cannot build a SpaceGroupAction from this object; pass sga= "
        "(or set builder.gauge_sga / use a downfolder with .model.phonon "
        "or .atoms)"
    )


def _anchor_site_ops(sga, declaration):
    wyck = str(declaration.wyckoff).lstrip("0123456789")
    wyckoffs = list(sga.symmetry_dataset.wyckoffs)
    kappa0 = next(
        (k for k, w in enumerate(wyckoffs) if w == wyck), None
    )
    if kappa0 is None:
        raise ValueError(
            f"wyckoff {declaration.wyckoff!r} not among atoms "
            f"{wyckoffs}"
        )
    site_ops = [
        g for g in range(sga.n_ops) if sga.atom_maps[g, kappa0] == kappa0
    ]
    return kappa0, site_ops


def _build_dw(sga, site_ops, declaration, nwann):
    """Declared D_W over site ops: per-irrep blocks, ``copies`` direct sum."""
    blocks = []
    dim_total = 0
    for name in declaration.site_irreps:
        tau = site_irrep_matrices(sga, site_ops, name)
        blocks.append(tau)
        dim_total += next(iter(tau.values())).shape[0]
    dim0 = next(iter(blocks[0].values())).shape[0]
    copies = nwann // dim0
    if nwann != dim0 * copies:
        raise ValueError(
            f"nwann={nwann} incompatible with declared site irrep dim "
            f"{dim0} (from {declaration.site_irreps})"
        )
    # one declared irrep name expanded to `copies` identical blocks
    base = blocks[0]
    dw = {}
    for g, tau_g in base.items():
        D = np.zeros((nwann, nwann), dtype=complex)
        for c in range(copies):
            D[c * dim0:(c + 1) * dim0, c * dim0:(c + 1) * dim0] = tau_g
        dw[int(g)] = D
    return dw, dim0, copies


def _qkey(q, tol=None):
    return tuple(np.round(_window_qkey(q), 6) % 1.0)


def _mesh_index_map(builder, tol):
    """Map wrapped mesh keys -> builder k-point index (nearest, unique)."""
    out = {}
    for ik, k in enumerate(np.asarray(builder.kpts, dtype=float)):
        out[_qkey(k)] = ik
    return out


def _resolved_window_bands(params):
    """Return the downfolder-validated window selection, if enabled."""
    if params is None:
        return None
    window_bands = getattr(params, "_window_bands_resolved", None)
    if window_bands is None:
        return None
    if not isinstance(window_bands, WindowBands):
        raise TypeError(
            "symmetry_adapted_gauge requires validated WindowBands; "
            "construct it through the phonopy downfolder"
        )
    return window_bands


def _window_rows(builder, window_bands, qpoint):
    """Map physical retained-band ids at ``qpoint`` to builder row indices."""
    qkey = _window_qkey(qpoint)
    try:
        selected = window_bands.bands[qkey]
    except KeyError as exc:
        raise ValueError(
            f"validated window lacks anchor {_window_qname(qkey)}"
        ) from exc
    rows_by_band = {int(band): row for row, band in enumerate(builder.ibands)}
    try:
        return np.asarray([rows_by_band[int(band)] for band in selected], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"window at {_window_qname(qkey)} contains excluded band {exc.args[0]}"
        ) from exc
def _invariant_carrier_from_block(basis, M, selected, qpoint):
    """Invariant sub-carrier of an accidentally degenerate block.

    ``check_window_legality`` proved the SELECTED bands span an invariant
    subspace, but the solver may have rotated vectors inside the numerical
    degeneracy, so the row indices alone no longer name it.  Split the block
    into its isotypic components via the COMMUTANT of the little-group
    action (a generic Hermitian commutant element's eigenspaces are
    invariant and group equal-irrep copies), then choose the component
    combination of the requested dimension that overlaps the selected rows
    most.  Commutant splitting -- unlike one-column probes -- preserves
    multidimensional irreps (e.g. an E doublet inside an E+A block).
    """
    n = basis.shape[1]
    restricted = {h: basis.conj().T @ matrix @ basis for h, matrix in M.items()}
    # Commutant nullspace: M_h X - X M_h = 0 for every h, vectorized.
    rows_eq = []
    for matrix in restricted.values():
        rows_eq.append(
            np.kron(np.eye(n), matrix) - np.kron(matrix.T, np.eye(n))
        )
    hom = np.vstack(rows_eq)
    _u, singular, vh = np.linalg.svd(hom)
    null_dim = int(np.count_nonzero(singular <= 1e-9 * max(singular[0], 1.0)))
    if null_dim == 0:
        raise ValueError(
            f"degenerate block at {_window_qname(_window_qkey(qpoint))} has a "
            "trivial commutant; cannot resolve the selected invariant subspace"
        )
    combination = np.zeros(n * n, dtype=complex)
    for i in range(null_dim):
        combination += (i + 1) * vh[-1 - i].conj()
    element = combination.reshape(n, n)
    element = (element + element.conj().T) / 2
    values, vectors = np.linalg.eigh(element)
    groups = []
    start = 0
    for i in range(1, n + 1):
        if i == n or abs(values[i] - values[start]) > 1e-7 * max(
            float(np.max(np.abs(values))), 1.0
        ):
            groups.append(vectors[:, start:i])
            start = i
    k = len(selected)
    best = None
    for mask in range(1, 1 << len(groups)):
        dims = [groups[i].shape[1] for i in range(len(groups)) if mask >> i & 1]
        if sum(dims) != k:
            continue
        block = np.column_stack(
            [groups[i] for i in range(len(groups)) if mask >> i & 1]
        )
        overlap = float(np.linalg.norm((basis @ block)[selected]) ** 2)
        if best is None or overlap > best[0]:
            best = (overlap, basis @ block)
    if best is None:
        raise ValueError(
            f"selected degenerate block at {_window_qname(_window_qkey(qpoint))} "
            f"has no invariant {k}-dimensional carrier"
        )
    return best[1]


def _anchor_window_representation(sga, M, qpoint, ops, rows, _irreps, energies):
    """Resolve explicit bands inside solver-rotated accidental degeneracies."""
    energies = np.asarray(energies, dtype=float)
    rows = np.asarray(rows, dtype=int)
    scale = max(float(np.max(np.abs(energies))), 1.0)
    tol = 1e-8 * scale
    carriers = []
    consumed = set()
    for row in rows:
        if row in consumed:
            continue
        block = np.flatnonzero(np.abs(energies - energies[row]) <= tol)
        selected = np.asarray([r for r in rows if r in set(block)], dtype=int)
        consumed.update(int(r) for r in selected)
        basis = np.eye(len(energies), dtype=complex)[:, block]
        if len(selected) == len(block):
            carriers.append(basis)
            continue
        carrier = _invariant_carrier_from_block(
            basis, {h: M[h] for h in ops}, selected, qpoint
        )
        carriers.append(carrier)
    phi = np.column_stack(carriers)
    residual = max(
        float(np.abs(M[h] @ phi - phi @ (phi.conj().T @ M[h] @ phi)).max())
        for h in ops
    )
    if residual > 1e-7:
        raise ValueError(
            f"window carrier at {_window_qname(_window_qkey(qpoint))} "
            f"is not invariant (residual {residual:.3e})"
        )
    dw = {h: phi.conj().T @ M[h] @ phi for h in ops}
    assert_projective_closure(dw, _GroupAlgebra(sga, ops).compose)
    return phi, dw


def _find_ik(keymap, q, tol):
    key = _qkey(q)
    if key in keymap:
        return keymap[key]
    q = np.mod(np.asarray(q, dtype=float).reshape(3), 1.0)
    for k, ik in keymap.items():
        d = np.asarray(k) - q
        d -= np.rint(d)
        if np.linalg.norm(d) < tol:
            return ik
    raise ValueError(f"mesh point {np.round(q, 4).tolist()} not in kpts")


def _window_spreads(builder, Amn, recip_lattice=None):
    """Mmn-form spread decomposition of a gauge (crystal units), or None.

    Uses the story-006 machinery on the window-restricted overlaps
    ``M~^{k,b} = U_k^dag M^{k,b} U_{k+b}``.
    """
    if recip_lattice is None:
        return None
    try:
        from lawaf.io.w90 import compute_Mmn, kmesh_nnlist
        from lawaf.wannierization.mlwf import omega_decomposition
    except Exception:  # pragma: no cover - optional heavy import
        return None
    ndim = int(getattr(builder, "ndim", 3))
    kmesh = np.asarray(builder.kmesh[:ndim], dtype=int)
    kpts = np.asarray(builder.kpts, dtype=float)
    if ndim != 3 or len(kpts) != int(np.prod(kmesh)):
        return None
    nnlist, nncell, bvecs, wb = kmesh_nnlist(
        kpts, recip_lattice, kmesh_tol=1e-6)
    bk = kpts[nnlist] + nncell - kpts[:, None, :]
    psi = np.stack([builder.get_psi_k(ik) for ik in range(builder.nkpt)])
    mmn = compute_Mmn(psi, nnlist, nncell)
    nwann = Amn.shape[2]
    mmn_t = np.zeros((builder.nkpt, nnlist.shape[1], nwann, nwann), dtype=complex)
    for ik in range(builder.nkpt):
        for j in range(nnlist.shape[1]):
            mmn_t[ik, j] = (
                Amn[ik].conj().T @ mmn[ik, j] @ Amn[nnlist[ik, j]]
            )
    om = omega_decomposition(
        mmn_t, wb, bk, guide=omega_decomposition(mmn_t, wb, bk)["rbar"]
    )
    return {
        k: float(max(np.real(v), 0.0))
        for k, v in om.items()
        if isinstance(v, (int, float, np.floating))
        or (isinstance(v, np.ndarray) and v.ndim == 0)
    }


def constrain_builder_amn(builder, declaration, sga=None, params=None,
                          downfolder=None, dw_override=None,
                          frac_positions=None):
    """Impose the star-covariant constrained gauge on ``builder.Amn``.

    Runs the standard-path unconstrained reference per irreducible q,
    reimposes the little-group constraint (Wigner-channel projection +
    Reynolds/polar polish), propagates over star arms, ties time-reversal
    partners, and rewrites ``builder.Amn`` in place.  Idempotent per builder
    (guarded by ``builder._gauge_applied``).

    Returns the diagnostics dict (eps table, spreads, info table).
    """
    decl = _resolve_declaration(declaration)
    if decl is None:
        raise ValueError(
            "symmetry_adapted_gauge requires "
            "WannierParams.representation_declaration"
        )
    tol = dict(DEFAULT_GAUGE_TOLERANCES)
    tols = getattr(params, "gauge_tolerances", None) or {}
    tol.update(tols)

    sga = _resolve_sga(downfolder, builder, sga)
    window_bands = _resolved_window_bands(params)
    if window_bands is not None:
        check_window_irreps(decl, window_bands)
    kappa0, site_ops = _anchor_site_ops(sga, decl)
    tol_k = tol["kpt_tol"]
    keymap = _mesh_index_map(builder, tol_k)
    if params is not None:
        kmesh = tuple(int(n) for n in np.asarray(params.kmesh).reshape(3))
        kshift = getattr(params, "kshift", None)
        gamma = getattr(params, "gamma", True)
    else:
        # bare builder (campaign regauge path): derive the mesh from the
        # builder k-point list
        kpts = np.asarray(builder.kpts, dtype=float)
        kmesh = tuple(
            int(np.unique(np.round(kpts[:, a] - kpts[0, a], 5)).size)
            for a in range(3)
        )
        kshift = None
        gamma = True
    mesh_irr = sga.irreducible_qpoints(kmesh, tol=tol_k)
    if kshift is None:
        kshift = np.zeros(3)
    if (not gamma) or bool(np.any(np.abs(kshift) > 1e-8)):
        raise ValueError(
            "symmetry_adapted_gauge requires a Gamma-centered unshifted "
            "mesh (gamma=True, kshift=0)"
        )
    constraint_qs = (
        mesh_irr
        if window_bands is None
        else np.asarray(tuple(window_bands.representatives), dtype=float)
    )
    irr_keys = {_qkey(q) for q in constraint_qs}
    nwann = builder.nwann
    if dw_override is not None:
        # explicit Wannier-space representation (e.g. the Cartesian
        # signed-permutation branch law used by the campaign regauge);
        # bypasses the declaration-resolved internal-basis convention
        dw_site = {
            int(g): np.asarray(m, dtype=complex)
            for g, m in dw_override.items()
        }
        dim0 = next(iter(dw_site.values())).shape[0]
        copies = nwann // dim0 if dim0 > 0 and nwann % dim0 == 0 else 1
    else:
        dw_site, dim0, copies = _build_dw(sga, site_ops, decl, nwann)
    Amn = np.array(builder.Amn, dtype=complex, copy=True)
    Amn_unconstrained = Amn.copy()
    # Constraint family: the window-restricted space-group action
    # M(h, q) = psi(q)^dag sga.matrix(h, q) psi(q) in the lawaf Bloch
    # gauge (the same convention as check_window_legality's decomposition
    # and star transport below), defect cocycle included.
    eps_table = {}
    info_table = {}
    constrained_keys = []
    window_frames = {}

    for q in constraint_qs:
        lg = little_group(sga, q, tol=tol_k)
        ik = _find_ik(keymap, q, tol_k)
        psi = builder.get_psi_k(ik)
        M = {h: psi.conj().T @ (sga.matrix(h, q) @ psi) for h in lg}
        if window_bands is None:
            dw = {h: dw_site[h] for h in lg}
            q_copies = copies
        else:
            rows = _window_rows(builder, window_bands, q)
            if len(rows) != nwann:
                raise ValueError(
                    f"window at {_window_qname(_window_qkey(q))} selects "
                    f"{len(rows)} bands; expected nwann={nwann}"
                )
            irreps = [
                name
                for block in window_bands.legality[_window_qkey(q)]
                for name in block.irreps
            ]
            phi, dw = _anchor_window_representation(
                sga, M, q, lg, rows, irreps, builder.get_eval_k(ik)
            )
            q_copies = 1
        U0 = Amn[ik]
        if not np.any(U0):
            U0 = builder.get_Amn_one_k(ik)
            Amn_unconstrained[ik] = U0
        if window_bands is not None:
            # ``phi`` is an exact, full-rank intertwiner in the builder's
            # frame; the original selected-band indices were used above to
            # validate its dimensionality.
            U0 = phi
        # self-reciprocal points (2q = 0 mod 1): time reversal maps q to
        # itself, so realness (U = Y U^*, Y = psi^dag psi^*) can be part
        # of the constraint -- but ONLY when the branch-space rep DW is
        # REAL (T maps manifold(DW) to manifold(DW^*); the tie keeps the
        # iterate on manifold(DW) iff DW^* = DW).  The declaration-
        # extracted dense convention is complex in general: there the tie
        # stays off and the behavior is the story-018 one.  Without the
        # tie, star coverage leaves the gauge complex and wannR picks up
        # O(1e-3) imaginary noise.
        two_q = 2.0 * np.asarray(q, dtype=float)
        self_reciprocal = bool(
            np.all(np.abs(two_q - np.rint(two_q)) <= tol_k)
        )
        dw_real = all(
            float(np.abs(np.imag(np.asarray(dw[h], dtype=complex))).max())
            <= 1e-12
            for h in lg
        )
        # ... and ONLY when the window rep is tie-compatible
        # (Y M_h^* = M_h Y, e.g. real window eigenvectors): otherwise the
        # tie map does not commute with the Reynolds projector and the
        # joint {intertwiner AND tie-invariant} manifold is empty-ish.
        tie = None
        if self_reciprocal and dw_real:
            Y = psi.conj().T @ psi.conj()
            if all(
                float(np.abs(Y @ M[h].conj() - M[h] @ Y).max()) <= 1e-10
                for h in lg
            ):
                tie = Y
        try:
            U, info = constrain_amn_one_q(
                M, dw, U0, copies=q_copies, tolerances=tol, tie=tie
            )
        except ValueError as exc:
            # A degenerate JOINT (intertwiner AND tie-invariant) Gram is
            # a seed artifact: the tie-invariant projection of THIS seed
            # cancels (e.g. a symmetric seed whose free phase is
            # tie-odd) -- the constraint itself is well posed.  Retry
            # without the tie and record the drop.
            if tie is None or "joint constraint channel" not in str(exc):
                raise
            U, info = constrain_amn_one_q(
                M, dw, U0, copies=q_copies, tolerances=tol, tie=None
            )
            info["tie_dropped"] = True
        Amn[ik] = U
        eps_table[_qkey(q)] = info["eps"]
        info_table[_qkey(q)] = info
        constrained_keys.append(_qkey(q))
        if window_bands is not None:
            window_frames[_qkey(q)] = phi

    # star propagation (never re-optimized) + time-reversal ties.
    # Priority per ADR-003/story ordering: a point whose time-reversal
    # partner is a DIRECTLY constrained point takes its gauge from the tie
    # U(-q) = Y U(q)^dag* (never re-optimized, and NOT from the inversion
    # star arm, which would pick a different commutant phase); all other
    # star members are propagated.
    covered = set(constrained_keys)
    tr_keys = {_qkey(-np.asarray(k)) for k in constrained_keys}
    tr_pairs = []
    for q0 in constraint_qs:
        k0 = _qkey(q0)
        ik0 = _find_ik(keymap, q0, tol_k)
        for g, qi in sga.star(q0, tol=tol_k):
            kk = _qkey(qi)
            if kk in irr_keys or kk in covered or kk in tr_keys:
                # irreducible points are constrained directly; points with
                # a directly constrained TR partner take the tie gauge
                # (priority over the inversion star arm); anything already
                # propagated stays untouched (first transport wins).
                continue
            ik = _find_ik(keymap, qi, tol_k)
            X = (
                builder.get_psi_k(ik).conj().T
                @ sga.matrix(g, q0)
                @ builder.get_psi_k(ik0)
            )
            if window_bands is None:
                dw_transport = dw_site[g]
            else:
                # Transport the chosen carrier itself.  This keeps the
                # frame on every arm in the same copy of the isotypic
                # component, avoiding an arbitrary multiplicity rotation.
                phi_dst = X @ window_frames[k0]
                dw_transport = np.eye(nwann, dtype=complex)
                window_frames[kk] = phi_dst
            Amn[ik] = X @ Amn[ik0] @ dw_transport.conj().T
            covered.add(kk)
    for kk in [k for k in keymap if k not in covered]:
        q = np.asarray(kk)
        qtr = np.mod(-q, 1.0)
        src = None
        for c in covered:
            d = np.asarray(c) - qtr
            d -= np.rint(d)
            if np.linalg.norm(d) < tol_k:
                src = c
                break
        if src is None:
            raise ValueError(
                f"mesh point {list(np.round(q, 4))} is neither in a star of "
                "a constrained irreducible point nor a time-reversal "
                "partner of one"
            )
        ik = _find_ik(keymap, q, tol_k)
        ik0 = _find_ik(keymap, src, tol_k)
        Y = (
            builder.get_psi_k(ik).conj().T
            @ builder.get_psi_k(ik0).conj()
        )
        Amn[ik] = Y @ Amn[ik0].conj()
        # the tied partner must still satisfy the constraint (real tau
        # guarantees the tie is an intertwiner); report its residual
        psi_t = builder.get_psi_k(ik)
        qt = np.asarray(kk)
        lg_t = little_group(sga, qt, tol=tol_k)
        if window_bands is None:
            dw_t = {h: dw_site[h] for h in lg_t}
        else:
            rows_t = _window_rows(builder, window_bands, qt)
            dw_t = _anchor_window_representation(sga, psi_t, qt, lg_t, rows_t)
        eps_table[kk] = max(
            float(
                np.abs(
                    (
                        psi_t.conj().T
                        @ sga.matrix(h, qt)
                        @ psi_t
                    )
                    @ Amn[ik]
                    - Amn[ik] @ dw_t[h]
                ).max()
            )
            for h in lg_t
        )
        info_table.setdefault(kk, {"tied_from": list(src)})
        tr_pairs.append([list(kk), list(src)])
        covered.add(kk)

    builder.Amn = Amn
    builder._amn_unconstrained = Amn_unconstrained
    builder._gauge_applied = True

    cell = getattr(sga, "_aux", {}).get("cell")
    recip = (
        2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
        if cell is not None else None
    )
    spreads = {
        "constrained": _window_spreads(builder, Amn, recip),
        "unconstrained": _window_spreads(builder, Amn_unconstrained, recip),
    }
    diagnostics = {
        "eps": eps_table,
        "per_q": info_table,
        "constrained_qs": constrained_keys,
        "tr_pairs": tr_pairs,
        "spreads": spreads,
        "anchor_site": int(kappa0),
        "declaration": {
            "wyckoff": decl.wyckoff,
            "site_irreps": list(decl.site_irreps),
            "window_irreps": (
                None
                if decl.window_irreps is None
                else {
                    _qkey(q): list(irreps)
                    for q, irreps in decl.window_irreps.items()
                }
            ),
        },
    }
    builder.gauge_diagnostics = diagnostics
    return diagnostics


def constrained_localize(
    downfolder_result_or_builder,
    declaration,
    sga=None,
    params=None,
    Rlist=None,
    Rdeg=None,
):
    """Star-covariant constrained localization (story-018 entry point).

    Accepts a downfolder (:class:`~lawaf.interfaces.downfolder.Lawaf`
    subclass instance; its builder, mesh and atoms are reused) or a bare
    projected builder (in which case ``sga`` must be passable/derivable and
    ``Rlist`` given for the R-space build).  Imposes the constrained gauge on
    the builder (:func:`constrain_builder_amn`), rebuilds wannR/HR, and
    returns the phonopy :class:`LWF` (downfolder path) or the wannierizer
    ``LWF`` (bare builder), with ``gauge_diagnostics`` attached.
    """
    obj = downfolder_result_or_builder
    builder = getattr(obj, "builder", obj)
    if params is None:
        params = builder.params
    atoms = getattr(obj, "atoms", None)
    frac_positions = (
        np.mod(atoms.get_scaled_positions(), 1.0) if atoms is not None else None
    )
    diagnostics = constrain_builder_amn(
        builder,
        declaration,
        sga=sga,
        params=params,
        downfolder=obj,
        frac_positions=frac_positions,
    )
    from lawaf.mathutils.kR_convert import k_to_R
    wannk, Hwannk, _Swannk = builder.get_wannk_and_Hk()

    if Rlist is None:
        Rlist = getattr(obj, "Rlist", None)
    if Rlist is None:
        raise ValueError(
            "constrained_localize needs Rlist (downfolder attribute or "
            "explicit argument)"
        )
    if Rdeg is None:
        Rdeg = getattr(obj, "Rdeg", None)
        if Rdeg is None:
            Rdeg = np.ones(len(Rlist))

    wannR = k_to_R(builder.kpts, Rlist, wannk, kweights=builder.kweights)
    HwannR = k_to_R(builder.kpts, Rlist, Hwannk, kweights=builder.kweights)

    atoms = getattr(obj, "atoms", None)
    lwf = None
    if atoms is not None and getattr(obj, "builder", None) is obj.builder:
        # phonopy-style result object, mirroring PhonopyDownfolder.downfold
        try:
            from lawaf.interfaces.phonopy.lwf import LWF as PhonopyLWF
            from lawaf.interfaces.phonopy.phonon_downfolder import (
                get_wannier_centers,
            )

            factor = getattr(obj, "factor", None)
            centers = get_wannier_centers(
                wannR,
                Rlist,
                atoms.get_scaled_positions(),
                Rdeg=Rdeg,
            )
            lwf = PhonopyLWF(
                factor=factor,
                Rlist=Rlist,
                Rdeg=Rdeg,
                wannR=wannR,
                HR_total=HwannR,
                kpts=builder.kpts,
                kweights=builder.kweights,
                wann_centers=centers,
                atoms=atoms,
            )
        except ImportError:
            lwf = None
    if lwf is None:
        lwf = builder.k_to_R(Rlist=Rlist, Rdeg=Rdeg)
        lwf.wannR = wannR
        lwf.HwannR = HwannR

    for attr, value in (
        ("gauge_diagnostics", diagnostics),
        ("spreads", (diagnostics["spreads"] or {}).get("constrained")),
    ):
        try:
            setattr(lwf, attr, value)
        except AttributeError:
            pass  # frozen result class
    return lwf
