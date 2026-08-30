"""Residual-baseline joint fit of the anharmonic effective model (story 021,
FR-010/018, ADR-009).

The training target is the residual of the labels with respect to the
harmonic LWF baseline,

    E_total(Q, eps) = E_harm(Q) + sum_t c_t B_t(Q, eps),

so the design rows are

    energy:  B_i c = E_lab - E_harm(Q_i)
    force:   J_i c = -g_Q - grad E_harm(Q_i)          (one row per Q coord)
    stress:  K_i c = sigma_lab - (1/V) dE_harm/deps   (one row per Voigt comp)

with the sign chain of ``lawaf.anharmonic.sampling``: g_Q = M^T F (ASE forces)
and dE/dQ = -g_Q, so the force residual reads
``dE_harm/dQ + dE_anh/dQ + g_Q = 0``.

Conventions
-----------
Harmonic baseline (E_harm)
    The phonopy-path LWF stores real-space blocks ``HR_total`` (nR, nwann,
    nwann) which are consumed ONLY through the k-space interpolation

        H(k) = sum_R Rdeg[R] * HR_total[R] * exp(2 pi i k.R)

    (``lawaf.mathutils.kR_convert.R_to_onek``; ``LWF.solve_k`` diagonalizes
    H(k) and ``evals_to_freqs`` reads its eigenvalues as omega^2).  The
    consistent real-space quadratic form of the (mass-weighted) LWF
    amplitudes Q -- flat over supercell cells, c = icell*nlwf + iwann -- is
    the Born-von Karman sum with the supercell-folded kernel

        K(D)   = sum_{R in Rlist, R = D mod supercell lattice} Rdeg[R] HR[R]
        E_harm = 1/2 sum_{c,c'} Q_c^T K(l_c' - l_c) Q_c'

    Folding by residues modulo the supercell lattice (rows of the integer
    sc_matrix) is EXACTLY the commensurate-q DFT fold
    ``(1/N) sum_q H(q) exp(-2 pi i q.D)`` by q-point orthogonality; R
    vectors absent from Rlist count as zero (standard real-space
    truncation).  A single-cell baseline (no supercell) folds every R onto
    that cell: K(0) = sum_R Rdeg[R] HR[R] = H(Gamma).
    Q amplitudes are the mass-weighted LWF amplitudes (the phonopy path maps
    them to Cartesian displacements through wann_disps = wannR/sqrt(mass),
    see ``build_lwf_lattice_mapping_matrix``); wann_masses therefore do NOT
    enter E_harm -- they convert amplitude^2 to physical mass only.
    Hermiticity: a quadratic form only sees the symmetric part of its
    matrix.  The baseline stores ``Hmat = (Kmat + Kmat^T)/2`` and uses
    E = 1/2 Q^T Hmat Q, grad = Hmat Q -- exact for ANY block content and
    equal to the physical model for Hermitian HR blocks; for non-real folds
    a warning is emitted and the real symmetric part is kept.  The identity
    d/dQ (1/2 Q^T H Q) = (H + H^T)/2 Q for non-symmetric H (and the
    vanishing of the antisymmetric contribution) is sympy-verified in
    tests/test_anharmonic_fit.py and
    docs/derivations/story021_fit_identities_sympy.py.

Anharmonic coordinates
    The fit works in the DATASET (supercell cell-major) Q order.  Basis
    coordinate labels (branch, R) are mapped onto supercell cells by folding
    R modulo the supercell lattice (``basis_cell_permutation``); the mapping
    must be a bijection (each cluster R a distinct cell of this supercell).
    Without a harmonic baseline the dataset Q is interpreted directly in
    ``basis.coord_labels`` order.

Stress convention
    ASE/dataset stress sigma_v = (1/V) dE/deps_v with the Voigt order
    (xx, yy, zz, yz, xz, xy) and each tensor component stored ONCE (no
    factor 2, matching ``sampling.voigt_to_matrix``).  The harmonic baseline
    carries no strain dependence, so stress rows constrain the anharmonic
    strain-sector coefficients only.

Weights
    Per-block RMS normalization by default: every block (energy, force,
    stress) is scaled by 1/rms(b_block) so all blocks start with unit RMS
    residual and contribute equally.  User weights (dict block->factor or a
    per-row array) multiply the block scale.  All scalings are positive row
    scalings; they leave exactly-consistent solutions unchanged (sympy
    assert in the test file).

NaN policy
    Rows with non-finite labels are dropped, never imputed.  dataset Q must
    be finite (build consistent-width frames first).
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "harmonic_baseline",
    "HarmonicBaseline",
    "basis_cell_permutation",
    "build_design",
    "fit",
    "AnharmonicCoefficients",
    "CoefficientTerm",
    "LocalityRow",
    "LocalityReport",
    "CVReport",
]

VOIGT_ORDER = ("xx", "yy", "zz", "yz", "xz", "xy")
_BLOCKS = ("energy", "force", "stress")


# ---------------------------------------------------------------------------
# harmonic baseline
# ---------------------------------------------------------------------------
def _fold_class(D, S, Sinv) -> Tuple[int, int, int]:
    """Residue of the integer primitive translation ``D`` modulo the
    supercell lattice (rows of ``S``): the unique integer vector
    ``D - m @ S`` whose supercell-reduced coordinates lie in [0, 1)^3."""
    D = np.asarray(D, dtype=int)
    y = Sinv @ D.astype(float)
    m = np.floor(y + 1e-8).astype(int)
    return tuple(int(v) for v in (D - m @ S))


class HarmonicBaseline:
    """Supercell-folded harmonic quadratic form of an LWF model.

    Attributes
    ----------
    Hmat:
        Symmetric (nQ, nQ) kernel, ``E = 1/2 Q^T Hmat Q``; see the module
        docstring for the folding convention and Hermiticity handling.
    """

    def __init__(self, lwf, scmaker=None):
        HR = getattr(lwf, "HR_total", None)
        if HR is None:
            HR = getattr(lwf, "HwannR", None)
        if HR is None:
            raise TypeError(
                "harmonic_baseline needs an LWF object exposing HR_total (or "
                "HwannR for the plain lawaf.lwf.lwf.LWF); got "
                f"{type(lwf).__name__}"
            )
        HR = np.asarray(HR)
        if HR.ndim != 3 or HR.shape[1] != HR.shape[2]:
            raise ValueError(f"HR blocks must be (nR, nwann, nwann), got {HR.shape}")
        self.HR = HR
        self.Rlist = np.asarray(lwf.Rlist, dtype=int)
        if self.Rlist.ndim != 2 or self.Rlist.shape[0] != HR.shape[0]:
            raise ValueError("Rlist must be (nR, 3) and match the HR blocks")
        Rdeg = getattr(lwf, "Rdeg", None)
        self.Rdeg = (
            np.ones(len(self.Rlist), dtype=float)
            if Rdeg is None
            else np.asarray(Rdeg, dtype=float)
        )
        self.nwann = int(HR.shape[1])
        if scmaker is None:
            from lawaf.utils.supercell import SupercellMaker

            scmaker = SupercellMaker(np.eye(3, dtype=int))
        self.sc_matrix = np.asarray(scmaker.sc_matrix, dtype=int)
        self.sc_vec = np.asarray(scmaker.sc_vec, dtype=int)
        nlwf = getattr(scmaker, "nlwf", None)
        if nlwf is not None and int(nlwf) != self.nwann:
            raise ValueError(
                f"supercell nlwf={nlwf} differs from the HR block size {self.nwann}"
            )
        self.ncell = len(self.sc_vec)
        self.nQ = self.ncell * self.nwann

        S, Sinv = self.sc_matrix, np.linalg.inv(self.sc_matrix.astype(float))
        blocks: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._R_class: List[Tuple[int, int, int]] = []
        for iR in range(len(self.Rlist)):
            key = _fold_class(self.Rlist[iR], S, Sinv)
            self._R_class.append(key)
            blk = blocks.get(key)
            if blk is None:
                blk = np.zeros((self.nwann, self.nwann), dtype=complex)
                blocks[key] = blk
            blk += self.Rdeg[iR] * HR[iR]

        # block [(c, i), (c', j)] = K(l_c' - l_c)[i, j]  (Born-von Karman)
        Kmat = np.zeros((self.nQ, self.nQ), dtype=complex)
        n = self.nwann
        for c, l_c in enumerate(self.sc_vec):
            for cp, l_cp in enumerate(self.sc_vec):
                key = _fold_class(l_cp - l_c, S, Sinv)
                blk = blocks.get(key)
                if blk is not None:
                    Kmat[c * n : (c + 1) * n, cp * n : (cp + 1) * n] = blk

        # Hermiticity handling: keep the symmetric part (the antisymmetric
        # part cannot contribute to 1/2 Q^T K Q; sympy-verified identity).
        Ksym = 0.5 * (Kmat + Kmat.conj().T)
        scale = max(1.0, float(np.abs(Ksym).max()))
        if float(np.abs(Ksym.imag).max()) > 1e-10 * scale:
            warnings.warn(
                "folded harmonic kernel has a non-real part (non-Hermitian HR "
                "blocks); keeping the real symmetric part"
            )
        self.Hmat = np.ascontiguousarray(Ksym.real)

    # -- evaluation ---------------------------------------------------------
    def _as_Q(self, Q) -> np.ndarray:
        Q = np.asarray(Q, dtype=float)
        single = Q.ndim == 1
        if single:
            Q = Q[None, :]
        if Q.ndim != 2 or Q.shape[1] != self.nQ:
            raise ValueError(f"Q must have nQ={self.nQ} amplitudes, got {Q.shape}")
        return Q, single

    def energy(self, Q) -> np.ndarray:
        """E_harm for one frame (nQ,) or a batch (nframes, nQ)."""
        Q, single = self._as_Q(Q)
        E = 0.5 * np.einsum("fi,ij,fj->f", Q, self.Hmat, Q)
        return float(E[0]) if single else E

    def gradient(self, Q) -> np.ndarray:
        """dE_harm/dQ; sympy-verified identity grad = (H + H^T)/2 Q = Hmat Q."""
        Q, single = self._as_Q(Q)
        G = Q @ self.Hmat.T
        return G[0] if single else G

    def energy_and_gradient(self, Q):
        Q, single = self._as_Q(Q)
        E = 0.5 * np.einsum("fi,ij,fj->f", Q, self.Hmat, Q)
        G = Q @ self.Hmat.T
        return (float(E[0]), G[0]) if single else (E, G)

    def hessian(self) -> np.ndarray:
        """Constant Hessian (the symmetric folded kernel)."""
        return self.Hmat

    def __repr__(self):
        return (
            f"HarmonicBaseline(nwann={self.nwann}, ncell={self.ncell}, "
            f"nR={len(self.Rlist)})"
        )


def harmonic_baseline(lwf, scmaker=None) -> HarmonicBaseline:
    """Build the :class:`HarmonicBaseline` of an LWF model.

    ``lwf``: any object exposing ``HR_total`` (phonopy-path LWF / NACLWF) or
    ``HwannR`` (plain ``lawaf.lwf.lwf.LWF``), plus ``Rlist`` and optional
    ``Rdeg``.  ``scmaker``: a ``SupercellMaker`` (or anything exposing
    ``sc_matrix``/``sc_vec``/``nlwf``) or an integer supercell matrix;
    ``None`` means the Gamma cell of a single primitive cell.
    """
    if not hasattr(scmaker, "sc_vec"):
        from lawaf.utils.supercell import SupercellMaker

        scmaker = SupercellMaker(scmaker if scmaker is not None else np.eye(3, dtype=int))
    return HarmonicBaseline(lwf, scmaker)


def _supercell_source(harmonic, mapping):
    """Object exposing ``sc_matrix``/``sc_vec`` for the pool->cell map.

    The harmonic baseline when present, else the mapping's ``scmaker``
    (a ``MyLWFSC``); ``None`` for raw-matrix mappings (identity pool
    evaluation, previous behavior).
    """
    if harmonic is not None:
        return harmonic
    scm = getattr(mapping, "scmaker", None)
    if scm is not None and hasattr(scm, "sc_matrix") and hasattr(scm, "sc_vec"):
        return scm
    return None


# ---------------------------------------------------------------------------
# basis <-> dataset coordinate reconciliation
# ---------------------------------------------------------------------------
def basis_cell_permutation(basis, harmonic) -> np.ndarray:
    """Map basis pool coordinates onto dataset (cell-major) Q coordinates.

    Returns ``pc`` with ``Q_pool[i] = Q_cell[pc[i]]`` for pool coordinate i =
    (branch b, primitive translation R): R folds modulo the supercell lattice
    onto supercell cell ``fc(R)`` and maps to cell coordinate
    ``fc(R) * nlwf + b``.  The fold must be a bijection: every R lands on a
    cell of the supercell and no two R's coincide (aliased cluster factors
    need the placement-summed design, which is out of scope here).
    """
    sc_matrix = np.asarray(harmonic.sc_matrix, dtype=int)
    sc_vec = np.asarray(harmonic.sc_vec, dtype=int)
    nlwf = int(basis.nlwf)
    if len(basis.coord_labels) != len(sc_vec) * nlwf:
        raise ValueError(
            f"basis pool has {len(basis.coord_labels)} coordinates but the "
            f"supercell provides {len(sc_vec) * nlwf} (ncell={len(sc_vec)}, "
            f"nlwf={nlwf}); build the basis on the supercell R set"
        )
    Sinv = np.linalg.inv(sc_matrix.astype(float))
    cell_index = {tuple(int(v) for v in vec): c for c, vec in enumerate(sc_vec)}
    pc = np.empty(len(basis.coord_labels), dtype=int)
    used = set()
    for i, (b, R) in enumerate(basis.coord_labels):
        key = _fold_class(R, sc_matrix, Sinv)
        c = cell_index.get(key)
        if c is None:
            raise ValueError(
                f"basis factor R={R} folds to {key}, not a supercell cell; "
                "the cluster Rlist must be a residue system of the supercell"
            )
        coord = c * nlwf + int(b)
        if coord in used:
            raise ValueError(
                f"basis factors alias onto supercell coordinate {coord}: the "
                "placement-summed design for aliased clusters is not supported"
            )
        used.add(coord)
        pc[i] = coord
    if sorted(pc.tolist()) != list(range(len(pc))):
        raise ValueError("pool->cell permutation is not a bijection")
    return pc


# ---------------------------------------------------------------------------
# exact polynomial gradients of the invariant basis (no finite differences)
# ---------------------------------------------------------------------------
class _BasisGrad:
    """Exact value/gradient evaluator for an :class:`InvariantBasis`.

    Every basis column is an exact rational combination of monomials
    ``prod Q_coord * prod strain_voigt``; derivatives are closed-form
    exponent products with exact integer exponents.  The exponent rules are
    sympy-derived/verified (tests/test_anharmonic_fit.py,
    docs/derivations/story021_fit_identities_sympy.py) and only checked
    numerically against them.
    """

    def __init__(self, basis):
        self.basis = basis
        self.ncoord = len(basis.coord_labels)
        self.nterms = len(basis.terms)
        index = {lab: i for i, lab in enumerate(basis.coord_labels)}
        self.needs_strain = any(
            t.sector in ("strain", "coupled") for t in basis.terms
        )
        monos = []  # (term_index, q_exponents, strain_exponents, coeff)
        for ti, t in enumerate(basis.terms):
            for (qk, sk), coeff in t.coeffs.items():
                eq = np.zeros(self.ncoord, dtype=np.int64)
                for f in qk:
                    eq[index[f]] += 1
                es = np.zeros(6, dtype=np.int64)
                for v in sk:
                    es[int(v)] += 1
                monos.append((ti, eq, es, float(coeff)))
        self._monos = monos

    def _strain_arg(self, Q, strain):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            Q = Q[None, :]
        E = None
        if self.needs_strain:
            if strain is None:
                raise ValueError("basis contains strain terms; pass strain")
            E = np.asarray(strain, dtype=float)
            if E.ndim == 1:
                E = E[None, :]
            if E.shape != (Q.shape[0], 6):
                raise ValueError(f"strain must be (nframes, 6), got {E.shape}")
        return Q, E

    def grad_q(self, Q, strain=None) -> np.ndarray:
        """d(column)/dQ: (nframes, ncoord, nterms)."""
        Q, E = self._strain_arg(Q, strain)
        n = Q.shape[0]
        out = np.zeros((n, self.ncoord, self.nterms))
        for ti, eq, es, coeff in self._monos:
            nz = np.nonzero(eq)[0]
            for j in nz:
                v = np.ones(n)
                for k in nz:
                    v = v * Q[:, k] ** (eq[k] - (1 if k == j else 0))
                if E is not None:
                    for k in np.nonzero(es)[0]:
                        v = v * E[:, k] ** es[k]
                out[:, j, ti] += coeff * eq[j] * v
        return out

    def grad_eps(self, Q, strain=None) -> np.ndarray:
        """d(column)/d(strain Voigt): (nframes, 6, nterms)."""
        Q, E = self._strain_arg(Q, strain)
        if E is None:
            return np.zeros((Q.shape[0], 6, self.nterms))
        n = Q.shape[0]
        out = np.zeros((n, 6, self.nterms))
        for ti, eq, es, coeff in self._monos:
            nz_s = np.nonzero(es)[0]
            for j in nz_s:
                v = np.ones(n)
                for k in nz_s:
                    v = v * E[:, k] ** (es[k] - (1 if k == j else 0))
                for k in np.nonzero(eq)[0]:
                    v = v * Q[:, k] ** eq[k]
                out[:, j, ti] += coeff * es[j] * v
        return out


# ---------------------------------------------------------------------------
# design matrix
# ---------------------------------------------------------------------------
def _resolve_mapping(mapping, nQ) -> np.ndarray:
    M = getattr(mapping, "mapping_mat", mapping)
    if hasattr(M, "toarray"):
        M = M.toarray()
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[1] != nQ or M.shape[0] % 3 != 0:
        raise ValueError(
            f"mapping must be (3*natom_sc, nQ={nQ}), got {M.shape}"
        )
    return M


def build_design(basis, dataset, harmonic=None, mapping=None):
    """Residual design rows (energy / projected-force / stress).

    Returns ``(A, b, row_meta)`` with one row per usable label:
    ``(A x - b)`` is the residual; ``row_meta`` holds
    ``(frame, block, index)`` per row (index = Q coordinate, Voigt component
    or -1 for energy).  NaN-labeled rows are dropped, never imputed.  Force
    rows require ``mapping`` (a ``MyLWFSC``, its ``mapping_mat``, or a dense
    (3 natom_sc, nQ) matrix) to project the Cartesian forces, g_Q = M^T F.
    """
    ds = dataset
    pg = _BasisGrad(basis)
    nT = pg.nterms
    nQ = ds.Q.shape[1]
    pc = None
    supercell = _supercell_source(harmonic, mapping)
    if harmonic is not None:
        if harmonic.nQ != nQ:
            raise ValueError(
                f"harmonic baseline nQ={harmonic.nQ} differs from dataset Q width {nQ}"
            )
    if supercell is not None:
        # evaluate the basis at POOL coordinates always: Q_pool[i] = Q_cell[pc[i]]
        pc = basis_cell_permutation(basis, supercell)
    elif pg.ncoord != nQ:
        raise ValueError(
            f"dataset Q width {nQ} differs from the basis coordinate count "
            f"{pg.ncoord}; pass the harmonic baseline or a MyLWFSC mapping "
            "so the pool->cell permutation can be built"
        )

    Q = np.asarray(ds.Q, dtype=float)
    if not np.isfinite(Q).all():
        raise ValueError(
            "dataset.Q contains NaN (build consistent-width frames); NaN "
            "labels are dropped but amplitudes must be finite"
        )
    strain_needed = pg.needs_strain
    strain_ok = np.isfinite(ds.strain_voigt).all(axis=1)
    if strain_needed and not strain_ok.all():
        if not strain_ok.any():
            raise ValueError("basis has strain terms but no frame has a finite strain")
        warnings.warn(
            f"{int((~strain_ok).sum())} frames have NaN strain_voigt and are "
            "excluded from the fit (the basis contains strain terms)"
        )

    finite_E = np.isfinite(ds.energies) & (strain_ok if strain_needed else True)
    F_flat = np.asarray(ds.forces, dtype=float).reshape(ds.nframes, -1)
    finite_F = np.isfinite(F_flat).all(axis=1) & (strain_ok if strain_needed else True)
    Sv = ds.stress_voigt
    finite_S = np.isfinite(Sv).all(axis=1) & (strain_ok if strain_needed else True)
    need = finite_E | finite_F | finite_S
    idx = np.flatnonzero(need)
    if idx.size == 0:
        raise ValueError("no usable labeled frames")
    if (finite_F & need).any() and mapping is None:
        raise ValueError(
            "force labels present but no mapping given; pass mapping=MyLWFSC "
            "(or the (3 natom_sc, nQ) matrix) to project forces"
        )

    # anharmonic columns, evaluated in pool coordinates
    Q_pool = Q[idx][:, pc] if pc is not None else Q[idx]
    strain_idx = ds.strain_voigt[idx] if strain_needed else None
    B = np.asarray(basis.evaluate(Q_pool, strain_idx), dtype=float)  # (k, nT)
    gQ_cell = pg.grad_q(Q_pool, strain_idx)
    if pc is not None:
        inv_pc = np.argsort(pc)
        gQ_cell = gQ_cell[:, inv_pc, :]  # scatter pool -> cell coordinates
    gE_cell = pg.grad_eps(Q_pool, strain_idx)

    # baseline residual sides
    if harmonic is not None:
        E_harm = np.array([harmonic.energy(Q[i]) for i in idx])
        g_harm = Q[idx] @ harmonic.Hmat.T
    else:
        E_harm = np.zeros(len(idx))
        g_harm = np.zeros((len(idx), nQ))

    b_E = np.asarray(ds.energies, dtype=float)[idx] - E_harm
    g_lab = np.einsum("kc,fk->fc", _resolve_mapping(mapping, nQ), F_flat[idx])
    b_F = -g_lab - g_harm
    # stress rows carry ASE units (eV/A^3 of the LABELED cell): the energy
    # labels are totals of that cell, so sigma = dE/deps / V_cell with the
    # cell volume of the dataset frames (NO primitive-cell division: the
    # fitted E is the labeled cell's total energy)
    vol = np.abs(np.linalg.det(np.asarray(ds.cell, dtype=float)[idx]))
    b_S = Sv[idx]  # the harmonic baseline carries no strain dependence

    A_rows, b_rows, meta = [], [], []
    for k, i in enumerate(idx):
        if finite_E[i]:
            A_rows.append(B[k])
            b_rows.append(b_E[k])
            meta.append((int(i), "energy", -1))
        if finite_F[i]:
            for c in range(nQ):
                A_rows.append(gQ_cell[k, c, :])
                b_rows.append(b_F[k, c])
                meta.append((int(i), "force", c))
        if finite_S[i]:
            for v in range(6):
                A_rows.append(gE_cell[k, v, :] / vol[k])
                b_rows.append(b_S[k, v])
                meta.append((int(i), "stress", v))
    A = np.asarray(A_rows, dtype=float).reshape(len(meta), nT)
    b = np.asarray(b_rows, dtype=float)
    return A, b, meta


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def _row_weights(b, row_meta, user_weights):
    """Per-block RMS normalization (1/rms of the block residual side), times
    optional user weights (dict block->factor or per-row array)."""
    w = np.ones(len(b))
    scales: Dict[str, float] = {}
    for blk in _BLOCKS:
        sel = [i for i, m in enumerate(row_meta) if m[1] == blk]
        if not sel:
            continue
        rms = float(np.sqrt(np.mean(b[sel] ** 2)))
        s = 1.0 / rms if rms > 0 else 1.0
        scales[blk] = s
        for i in sel:
            w[i] = s
    if user_weights is not None:
        if isinstance(user_weights, dict):
            for blk, fac in user_weights.items():
                if blk not in _BLOCKS:
                    raise ValueError(f"unknown block {blk!r}")
                sel = [i for i, m in enumerate(row_meta) if m[1] == blk]
                for i in sel:
                    w[i] *= float(fac)
        else:
            uw = np.asarray(user_weights, dtype=float)
            if uw.shape != (len(b),):
                raise ValueError("per-row weights must have one entry per design row")
            w = w * uw
    return w, scales


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------
def _ridge_solve(A, b, w, alpha):
    """Weighted ridge via dense normal equations.

    The design matrix is always materialized dense by ``build_design``, so
    the exact normal-equation solve is preferred up to a large budget
    (``n*p <= 4e8``); scipy lsmr was tried as a fallback for larger
    systems but with a small damp it converges too slowly on tall designs
    (under-converged fits: observed CV force cosine 0.71 vs 0.89 exact).
    """
    n, p = A.shape
    if n * p > 4e8:  # very large: approximate sparse path, no stderr
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsmr

        Aw = csr_matrix(A * w[:, None])
        x = lsmr(Aw, b * w, damp=float(alpha))[0]
        return x, None
    Aw = A * w[:, None]
    bw = b * w
    G = Aw.T @ Aw
    if alpha > 0:
        G = G + float(alpha) * np.eye(p)
    rhs = Aw.T @ bw
    try:
        x = np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(G, rhs, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(Aw)) if alpha == 0 else p
    dof = max(n - rank, 1)
    r = Aw @ x - bw
    sigma2 = float(r @ r) / dof
    if p <= 3000:
        Ginv = np.linalg.pinv(G)
        if alpha > 0:
            middle = G - float(alpha) * np.eye(p)
            cov = sigma2 * (Ginv @ middle @ Ginv)
        else:
            cov = sigma2 * Ginv
    else:
        # large p: inverse via one triangular-free solve (pinv's SVD is
        # needlessly expensive at this size); identical for full-rank G
        Ginv = np.linalg.solve(G, np.eye(p))
        if alpha > 0:
            middle = G - float(alpha) * np.eye(p)
            cov = sigma2 * (Ginv @ middle @ Ginv)
        else:
            cov = sigma2 * Ginv
    stderr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return x, stderr


def _screened_greedy(A, b, w, n_coeff=None, seed=0, n_screen=None):
    """Screened greedy forward selection (pymultibinit semantics).

    1. deterministic row subsample (linspace, ``_screening_dataset`` style);
    2. rank all columns by the single-column residual-norm gain
       ``|b|^2 - (a.b)^2/|a|^2`` on the subsample (zero columns rejected);
    3. greedy forward selection on the full rows over the surviving pool in
       ascending candidate order, ``score < best - tol`` else exact-tie
       lower-index wins, singular trials (lstsq rank deficiency) rejected.
    Runs to ``n_coeff`` columns, or (default) until the SSR stops improving.
    """
    n, p = A.shape
    target = p if n_coeff is None else min(int(n_coeff), p)
    nsub = int(min(n, max(10 * target, 64)))
    sub = np.linspace(0, n - 1, nsub, dtype=int)
    Aw_s = A[sub] * w[sub, None]
    bs = b[sub] * w[sub]
    denom = np.einsum("ij,ij->j", Aw_s, Aw_s)
    rhs = Aw_s.T @ bs
    scores = np.full(p, np.inf)
    ok = denom > 0
    scores[ok] = (bs @ bs) - rhs[ok] ** 2 / denom[ok]
    order = np.argsort(scores, kind="stable")
    # pymultibinit: pool = max(10 * ncoeff, 1000), capped at p (screening
    # only bites for large bases)
    pool_size = min(p, max(10 * target, 1000)) if n_screen is None else int(n_screen)
    pool = [int(j) for j in order[np.isfinite(scores[order])][:pool_size]]

    Aw = A * w[:, None]
    bw = b * w
    # Gram/Schur-complement forward selection: identical candidate scores,
    # tie-breaking, singular rejection and early-stop semantics as a
    # per-candidate lstsq scan (min |A_{S+c} x - b|^2 with the new column's
    # Schur gain (z_c - G_{cS} x_S)^2 / (G_cc - G_{cS} G_SS^{-1} G_{Sc});
    # a (near-)zero Schur denominator is exactly lstsq rank deficiency),
    # at O(pool * target^2) total cost instead of O(pool * target * n * k^2).
    Gpp = np.einsum("ij,ij->j", Aw, Aw)  # diag of the full Gram
    G = Aw.T @ Aw
    z = Aw.T @ bw
    pool_arr = np.asarray(pool, dtype=int)
    G_pool = G[np.ix_(pool_arr, pool_arr)]
    z_pool = z[pool_arr]
    d_pool = Gpp[pool_arr]
    rel_tol = 1e-10

    selected: List[int] = []
    sel_pos: List[int] = []  # positions within pool_arr
    chosen = set()
    prev_ssr = None
    skipped_singular = 0
    while len(selected) < target:
        if sel_pos:
            S = np.asarray(sel_pos, dtype=int)
            G_SS = G_pool[np.ix_(S, S)]
            x_S = np.linalg.lstsq(G_SS, z_pool[S], rcond=None)[0]
            cross = G_pool[:, S]  # (pool, |S|)
            P = np.linalg.pinv(G_SS)
            denom = d_pool - np.einsum("is,is->i", cross @ P, cross)
            num = z_pool - cross @ x_S
        else:
            denom = d_pool.copy()
            num = z_pool.copy()
        gain = np.full(len(pool_arr), -np.inf)
        ok = denom > rel_tol * np.maximum(d_pool, 1e-300)
        gain[ok] = num[ok] ** 2 / denom[ok]
        for pos in np.flatnonzero(~ok):
            if int(pool_arr[pos]) not in chosen:
                skipped_singular += 1
        gain[[p_ for p_, c in enumerate(pool_arr) if int(c) in chosen]] = -np.inf
        if not np.isfinite(gain).any() or gain.max() <= 1e-12:
            break
        gmax = gain.max()
        tied = np.flatnonzero(gain >= gmax - 1e-12)
        best_pos = min(tied, key=lambda p_: int(pool_arr[p_]))
        selected.append(int(pool_arr[best_pos]))
        sel_pos.append(int(best_pos))
        chosen.add(int(pool_arr[best_pos]))
        # (early stop is the gain test above: max gain <= 1e-12 means no
        # candidate improves the SSR, matching the original prev-SSR check)
    if selected:
        S = np.asarray(sel_pos, dtype=int)
        x_sel = np.linalg.lstsq(
            G_pool[np.ix_(S, S)], z_pool[S], rcond=None
        )[0]
    else:
        x_sel = np.zeros(0)
    return selected, x_sel, skipped_singular


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@dataclass
class CoefficientTerm:
    """One selected basis column with its fitted coefficient."""

    index: int
    seed: object  # ClusterKey of basis.terms[index]
    order: int
    sector: str
    coeff: float
    stderr: Optional[float]


@dataclass
class LocalityRow:
    term_index: int
    sector: str
    order: int
    radius_lattice: float  # max pairwise |dR| over the seed factors
    radius_cart: Optional[float]  # same in Angstrom (needs the primitive cell)
    abs_coeff: float
    mass_fraction: float


@dataclass
class LocalityReport:
    """Coefficient mass versus cluster radius (distance between R factors)."""

    rows: List[LocalityRow]
    r90_lattice: Optional[float] = None
    r90_cart: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"{'term':>5} {'sector':>8} {'order':>5} {'radius':>8} "
            f"{'r/Å':>8} {'|c|':>10} {'mass%':>7}"
        ]
        for r in sorted(self.rows, key=lambda r: (r.radius_lattice, r.term_index)):
            cart = "" if r.radius_cart is None else f"{r.radius_cart:8.3f}"
            lines.append(
                f"{r.term_index:>5} {r.sector:>8} {r.order:>5} "
                f"{r.radius_lattice:>8.3f} {cart:>8} {r.abs_coeff:>10.4g} "
                f"{100 * r.mass_fraction:>6.2f}%"
            )
        r90 = f"r90(lattice)={self.r90_lattice}"
        if self.r90_cart is not None:
            r90 += f" r90(Angstrom)={self.r90_cart:.3f}"
        lines.append(r90)
        return "\n".join(lines)


@dataclass
class CVReport:
    """Cross-validation metrics on the dataset 'cv' split."""

    n_train_rows: int
    n_cv_frames: int
    energy_mae: Optional[float] = None
    force_rmse: Optional[float] = None
    force_cosine: Optional[float] = None
    stress_rmse: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def _locality_report(basis, coefficients, selected, prim_cell=None) -> LocalityReport:
    rows = []
    masses = []
    radii = []
    for ti in selected:
        t = basis.terms[ti]
        Rs = [np.asarray(R, dtype=float) for (_, R) in t.seed.factors]
        radius = 0.0
        for i in range(len(Rs)):
            for j in range(i + 1, len(Rs)):
                radius = max(radius, float(np.linalg.norm(Rs[i] - Rs[j])))
        cart = None
        if prim_cell is not None and len(Rs) > 1:
            cart = 0.0
            for i in range(len(Rs)):
                for j in range(i + 1, len(Rs)):
                    d = Rs[i] - Rs[j]
                    cart = max(cart, float(np.linalg.norm(prim_cell.T @ d)))
        mass = abs(float(coefficients[ti]))
        rows.append(
            LocalityRow(ti, t.sector, t.order, radius, cart, mass, 0.0)
        )
        masses.append(mass)
        radii.append(radius)
    total = sum(masses)
    if total > 0:
        for r in rows:
            r.mass_fraction = r.abs_coeff / total
    order = sorted(range(len(rows)), key=lambda k: (radii[k], rows[k].term_index))
    cum = 0.0
    r90_lattice = r90_cart = None
    for k in order:
        cum += rows[k].mass_fraction
        if cum >= 0.9:
            r90_lattice = rows[k].radius_lattice
            r90_cart = rows[k].radius_cart
            break
    return LocalityReport(rows=rows, r90_lattice=r90_lattice, r90_cart=r90_cart)


def _cv_metrics(basis, coefficients, harmonic, dataset, pc, mapping) -> CVReport:
    ds = dataset
    split = np.asarray([str(s) for s in ds.split])
    cv_frames = np.flatnonzero(split == "cv")
    train_rows = int(np.sum(split == "train"))
    report = CVReport(n_train_rows=train_rows, n_cv_frames=int(len(cv_frames)))
    if len(cv_frames) == 0:
        report.notes.append("no cv frames in the dataset")
        return report
    pg = _BasisGrad(basis)
    Q = np.asarray(ds.Q, dtype=float)[cv_frames]
    nQ = Q.shape[1]
    if not np.isfinite(Q).all():
        report.notes.append("cv frames contain NaN amplitudes; metrics skipped")
        return report
    strain_needed = pg.needs_strain
    strain = np.asarray(ds.strain_voigt, dtype=float)[cv_frames]
    if strain_needed:
        ok_strain = np.isfinite(strain).all(axis=1)
        if not ok_strain.all():
            report.notes.append(
                f"{int((~ok_strain).sum())} cv frames dropped (NaN strain)"
            )
            cv_frames = cv_frames[ok_strain]
            Q, strain = Q[ok_strain], strain[ok_strain]
            if len(cv_frames) == 0:
                return report
    Q_pool = Q[:, pc] if pc is not None else Q
    E_anh = np.asarray(basis.evaluate(Q_pool, strain if strain_needed else None))
    gQ_pool = pg.grad_q(Q_pool, strain if strain_needed else None)
    gQ_cell = gQ_pool[:, np.argsort(pc), :] if pc is not None else gQ_pool
    gE_cell = pg.grad_eps(Q_pool, strain if strain_needed else None)
    if harmonic is not None:
        E_pred = E_anh @ coefficients + np.array(
            [harmonic.energy(q) for q in Q]
        )
        g_pred = -(gQ_cell @ coefficients + Q @ harmonic.Hmat.T)
    else:
        E_pred = E_anh @ coefficients
        g_pred = -(gQ_cell @ coefficients)
    # ASE units of the labeled cell (see build_design)
    vol = np.abs(np.linalg.det(np.asarray(ds.cell, dtype=float)[cv_frames]))
    sigma_pred = (gE_cell @ coefficients) / vol[:, None]

    E_lab = np.asarray(ds.energies, dtype=float)[cv_frames]
    okE = np.isfinite(E_lab)
    if okE.any():
        report.energy_mae = float(np.mean(np.abs(E_pred[okE] - E_lab[okE])))
    else:
        report.notes.append("no finite cv energies")

    F_flat = np.asarray(ds.forces, dtype=float)[cv_frames].reshape(len(cv_frames), -1)
    if np.isfinite(F_flat).any():
        if mapping is None:
            report.notes.append("cv force metrics need the mapping (not given)")
        else:
            M = _resolve_mapping(mapping, nQ)
            g_lab = np.einsum("kc,fk->fc", M, F_flat)
            okF = np.isfinite(F_flat).all(axis=1)
            if okF.any():
                d = g_pred[okF] - g_lab[okF]
                report.force_rmse = float(np.sqrt(np.mean(d**2)))
                cos = []
                for a, bb in zip(g_pred[okF], g_lab[okF]):
                    na, nb = np.linalg.norm(a), np.linalg.norm(bb)
                    if na < 1e-30 and nb < 1e-30:
                        continue
                    if na < 1e-30 or nb < 1e-30:
                        cos.append(0.0)
                    else:
                        cos.append(float(a @ bb / (na * nb)))
                if cos:
                    report.force_cosine = float(np.mean(cos))
    else:
        report.notes.append("no finite cv forces")

    S_lab = ds.stress_voigt[cv_frames]
    okS = np.isfinite(S_lab).all(axis=1)
    if okS.any():
        report.stress_rmse = float(
            np.sqrt(np.mean((sigma_pred[okS] - S_lab[okS]) ** 2))
        )
    else:
        report.notes.append("no finite cv stresses")
    return report


def _fit_fingerprint(basis, dataset, scales, user_weights, selection, alpha,
                     n_coeff, seed, A, b, selected) -> str:
    h = hashlib.sha256()
    h.update(basis.fingerprint.encode())
    h.update(np.asarray([str(s) for s in dataset.split]).tobytes())
    h.update(repr({k: round(float(v), 15) for k, v in scales.items()}).encode())
    if isinstance(user_weights, np.ndarray):
        h.update(np.ascontiguousarray(user_weights).tobytes())
    else:
        h.update(repr(user_weights).encode())
    h.update(
        json.dumps(
            [
                selection,
                float(alpha),
                None if n_coeff is None else int(n_coeff),
                int(seed),
                [int(i) for i in selected],
            ]
        ).encode()
    )
    h.update(np.ascontiguousarray(A).tobytes())
    h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
@dataclass
class AnharmonicCoefficients:
    """Fitted anharmonic model: coefficients over an :class:`InvariantBasis`.

    The evaluation methods take Q in the DATASET (cell-major) order;
    ``coord_perm`` is the pool->cell index map used internally.
    """

    basis: object
    harmonic: Optional[HarmonicBaseline]
    terms: List[CoefficientTerm]
    coefficients: np.ndarray  # (n_terms,) dense; zeros off the support
    stderrs: np.ndarray
    selected: Tuple[int, ...]
    locality: LocalityReport
    fingerprint: str
    cv: CVReport
    selection: str
    ridge_alpha: float
    weights: Dict[str, float]
    coord_perm: Optional[np.ndarray] = None

    # -- evaluation ---------------------------------------------------------
    def _pool_Q(self, Q, strain):
        Q = np.asarray(Q, dtype=float)
        single = Q.ndim == 1
        if single:
            Q = Q[None, :]
        st = None
        if strain is not None:
            st = np.asarray(strain, dtype=float)
            if st.ndim == 1:
                st = st[None, :]
        Qp = Q[:, self.coord_perm] if self.coord_perm is not None else Q
        return Qp, st, single

    def anharm_energy(self, Q, strain=None) -> np.ndarray:
        Qp, st, single = self._pool_Q(Q, strain)
        E = np.asarray(self.basis.evaluate(Qp, st), dtype=float) @ self.coefficients
        return float(E[0]) if single else E

    def energy(self, Q, strain=None):
        """Total fitted energy (harmonic baseline included when present)."""
        E = self.anharm_energy(Q, strain)
        if self.harmonic is not None:
            E = E + self.harmonic.energy(Q)
        return E

    def anharm_gradient(self, Q, strain=None) -> np.ndarray:
        Qp, st, single = self._pool_Q(Q, strain)
        g = _BasisGrad(self.basis).grad_q(Qp, st) @ self.coefficients
        if self.coord_perm is not None:
            out = np.empty_like(g)
            out[:, self.coord_perm] = g
            g = out
        return g[0] if single else g

    def gradient(self, Q, strain=None):
        """Total fitted dE/dQ (harmonic included), dataset coordinate order."""
        g = self.anharm_gradient(Q, strain)
        if self.harmonic is not None:
            g = g + self.harmonic.gradient(Q)
        return g

    def __repr__(self):
        return (
            f"AnharmonicCoefficients(n_terms={len(self.coefficients)}, "
            f"selected={len(self.selected)}, selection={self.selection!r}, "
            f"fingerprint={self.fingerprint[:12]}...)"
        )


# ---------------------------------------------------------------------------
# top-level fit
# ---------------------------------------------------------------------------
def fit(
    dataset,
    basis,
    harmonic,
    weights=None,
    selection: str = "ridge",
    ridge_alpha: float = 1e-8,
    n_coeff: Optional[int] = None,
    seed: int = 0,
    mapping=None,
    prim_cell=None,
) -> AnharmonicCoefficients:
    """Joint residual-baseline fit of the anharmonic coefficients.

    Parameters
    ----------
    dataset:
        :class:`~lawaf.anharmonic.dataset.TrainingDataset` with Q, strain and
        an active label block; 'train' rows are fitted, 'cv' rows reported.
    basis:
        :class:`~lawaf.anharmonic.basis.InvariantBasis`.
    harmonic:
        :class:`HarmonicBaseline` (or None for a pure anharmonic fit; then
        the dataset Q must already be in ``basis.coord_labels`` order).
    weights:
        None (per-block RMS normalization) / dict block->factor / per-row
        array; user weights multiply the block scale.
    selection:
        'ridge' (weighted ridge, ``ridge_alpha``) or 'screened_greedy'
        (screened forward selection to ``n_coeff`` columns or until the SSR
        stops improving; deterministic; singular trials rejected).
    mapping:
        MyLWFSC / mapping matrix, needed when force labels are present.
    prim_cell:
        Optional 3x3 primitive-cell rows for the Cartesian locality report;
        derived from the dataset cell and the supercell matrix when absent.
    """
    if selection not in ("ridge", "screened_greedy"):
        raise ValueError(f"selection must be 'ridge' or 'screened_greedy', got {selection!r}")
    A, b, row_meta = build_design(basis, dataset, harmonic=harmonic, mapping=mapping)
    w, scales = _row_weights(b, row_meta, weights)
    split = np.asarray([str(s) for s in dataset.split])
    train = np.asarray(
        [i for i, m in enumerate(row_meta) if split[m[0]] == "train"], dtype=int
    )
    if train.size == 0:
        raise ValueError("no 'train' rows in the dataset")
    At, bt, wt = A[train], b[train], w[train]

    nT = A.shape[1]
    if selection == "ridge":
        coefficients, stderrs = _ridge_solve(At, bt, wt, ridge_alpha)
        if stderrs is None:  # sparse lsmr fallback: no covariance available
            stderrs = np.full(nT, np.nan)
        selected = tuple(int(i) for i in np.flatnonzero(coefficients))
    else:
        sel, x_sel, _ = _screened_greedy(At, bt, wt, n_coeff=n_coeff, seed=seed)
        selected = tuple(int(i) for i in sel)
        coefficients = np.zeros(nT)
        coefficients[list(selected)] = x_sel
        stderrs = np.full(nT, np.nan)

    pc = None
    supercell = _supercell_source(harmonic, mapping)
    if supercell is not None:
        pc = basis_cell_permutation(basis, supercell)
    prim = prim_cell
    if prim is None and supercell is not None:
        S = np.asarray(supercell.sc_matrix, dtype=float)
        prim = np.linalg.solve(S, np.asarray(dataset.cell, dtype=float)[0])

    fingerprint = _fit_fingerprint(
        basis, dataset, scales, weights, selection, ridge_alpha, n_coeff,
        seed, A, b, selected,
    )
    locality = _locality_report(basis, coefficients, selected, prim)
    cv = _cv_metrics(basis, coefficients, harmonic, dataset, pc, mapping)
    terms = [
        CoefficientTerm(
            index=ti,
            seed=basis.terms[ti].seed,
            order=basis.terms[ti].order,
            sector=basis.terms[ti].sector,
            coeff=float(coefficients[ti]),
            stderr=None if np.isnan(stderrs[ti]) else float(stderrs[ti]),
        )
        for ti in selected
    ]
    return AnharmonicCoefficients(
        basis=basis,
        harmonic=harmonic,
        terms=terms,
        coefficients=coefficients,
        stderrs=stderrs,
        selected=selected,
        locality=locality,
        fingerprint=fingerprint,
        cv=cv,
        selection=selection,
        ridge_alpha=float(ridge_alpha),
        weights=dict(scales),
        coord_perm=pc,
    )
