"""Evaluator and ASE calculator for the anharmonic LWF effective model
(story 022, FR-011/012, ADR-004/010).

:func:`evaluate_atoms` / :class:`LWFModelCalculator` invert the sampling
pipeline of :mod:`lawaf.anharmonic.sampling`: given an ``ase.Atoms`` object
they recover the mode amplitudes ``Q`` (and the strain parameters) and
evaluate the fitted :class:`~lawaf.anharmonic.fit.AnharmonicCoefficients`.

Conventions
-----------
Q recovery
    ``u = M @ Q`` with the flat (3 natom_sc, nQ) mapping matrix ``M`` of
    ``build_lwf_lattice_mapping_matrix`` (column ``c = icell * nlwf + iwann``
    in ``scmaker.sc_vec`` order).  The least-squares inverse ``Q = M^+ u`` is
    computed with a precomputed dense pseudo-inverse for small models
    (``nQ <= 10``) and ``scipy.sparse.linalg.lsqr`` on the sparse mapping
    otherwise (``solver="auto"``; both paths selectable explicitly).

Forces
    ASE forces are ``F = -dE/du``.  Because the model only defines the energy
    on the subspace ``{M Q}``, the Cartesian force reported is the minimum-
    norm gradient of the canonical lift ``E(M^+ u)``::

        F = -M (M^T M)^{-1} dE/dQ

    which is the exact left inverse of the sampling projection
    ``g_Q = M^T F`` (so ``M^T F == -F-forces`` sign chain of sampling) and
    reduces to ``-M dE/dQ`` for orthonormal modes.  For non-orthonormal
    columns of ``M`` the plain product ``-M dE/dQ`` violates the work chain
    rule ``delta-E = -F . delta-u`` along subspace displacements; the
    identity is sympy-verified in ``tests/test_anharmonic_model.py``.

Out-of-subspace displacements
    A displacement component outside ``range(M)`` cannot be represented by
    the model.  It is detected as the least-squares residual
    ``||u - M M^+ u||`` and reported through
    :class:`LWFSubspaceResidualWarning` (carrying ``.residual_norm``); the
    energy is evaluated at the PROJECTED amplitudes.

Strain (ADR-010)
    A strained cell is recognized as ``cell = cell_ref @ (I + eps)``; the
    deformation gradient ``F = cell @ cell_ref^{-1}`` must be symmetric to
    numerical precision (a warning is emitted otherwise), and the atomic
    displacement is recovered by inverting the affine scaling exactly,
    ``u = F^{-1} x - x_ref`` (the exact inverse of ``sampling.make_atoms``;
    ``x - F x_ref`` agrees only to first order in eps).
    The model is evaluated with the reference mapping ``M(eps) = M(0)``:
    this is the small-strain regime, exact to first order in ``eps``
    (ADR-010 validity note).  For a basis WITHOUT strain terms the strain is
    ignored entirely.

Stress
    ``sigma_v = (1/vol) dE/deps_v`` in the Voigt order (xx, yy, zz, yz, xz,
    xy), each tensor component stored once - exactly the ``fit.py`` design
    convention.  The harmonic baseline carries no strain dependence, so the
    stress constrains the strain-sector coefficients only.  The volume is
    the per-primitive-cell volume of the DEFORMED configuration,
    ``vol = V0_prim * det(I + eps)`` (Jacobi: ``d det(I+eps)/d eps =
    det(I+eps) (I+eps)^{-1}``, sympy-verified); ``V0_prim`` is the reference
    primitive-cell volume ``|det(cell_ref)| / |det(sc_matrix)|``.  ASE
    stress is tension-positive (``sigma = (1/V) dE/deps``), matching
    ``atoms.get_stress()``.

Hessian
    ``d2E/dQ2 = H_harm + sum_t c_t d2B_t/dQ2`` with the exact monomial
    second-derivative exponent rule
    ``n_j (n_k - delta_jk) prod Q^(n - delta_j - delta_k)`` (sympy-verified
    against ``sympy.diff``); no finite differences anywhere.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from lawaf.anharmonic.basis import InvariantBasis
from lawaf.anharmonic.dataset import matrix_to_voigt
from lawaf.anharmonic.fit import (
    AnharmonicCoefficients,
    _BasisGrad,
)
from lawaf.anharmonic.sampling import voigt_to_matrix

__all__ = [
    "AnharmonicModel",
    "LWFModelCalculator",
    "LWFSubspaceResidualWarning",
    "evaluate_atoms",
]

#: ``solver="auto"`` uses a precomputed dense pseudo-inverse at or below this
#: many amplitudes and scipy lsqr above it (dense pinv cost is O(nQ^3)).
DENSE_PINV_MAX_NQ = 10


class LWFSubspaceResidualWarning(UserWarning):
    """The atomic displacement leaves the model subspace ``range(M)``.

    Attributes
    ----------
    residual_norm:
        ``||u - M M^+ u||_2`` - the norm of the unrepresentable component.
    """

    def __init__(self, message: str, residual_norm: float):
        super().__init__(message)
        self.residual_norm = float(residual_norm)


# ---------------------------------------------------------------------------
# exact monomial Hessian of the invariant basis
# ---------------------------------------------------------------------------
class _BasisHess:
    """Exact second Q-derivatives of an :class:`InvariantBasis`.

    For a monomial ``prod Q_l^n_l`` the exponent rule is
    ``d2/dQj dQk = n_j (n_k - delta_jk) prod Q_l^(n_l - delta_lj - delta_lk)``,
    sympy-verified against ``sympy.diff`` in
    ``tests/test_anharmonic_model.py::test_sympy_monomial_hessian_exponent_rule``.
    """

    def __init__(self, basis: InvariantBasis):
        self.basis = basis
        self.ncoord = len(basis.coord_labels)
        self.nterms = len(basis.terms)
        index = {lab: i for i, lab in enumerate(basis.coord_labels)}
        self.needs_strain = any(
            t.sector in ("strain", "coupled") for t in basis.terms
        )
        monos = []
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

    def hess_q(self, Q, strain=None) -> np.ndarray:
        """d2(column)/dQj dQk: (nframes, ncoord, ncoord, nterms)."""
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
        n = Q.shape[0]
        out = np.zeros((n, self.ncoord, self.ncoord, self.nterms))
        for ti, eq, es, coeff in self._monos:
            nz = np.nonzero(eq)[0]
            for a, j in enumerate(nz):
                for k in nz[a:]:
                    mult = eq[j] * (eq[k] - (1 if j == k else 0))
                    if mult == 0:
                        # e.g. diagonal d2/dQj2 of a linear factor: the
                        # exponent rule would evaluate 0 * Q**-1 at this
                        # point; the true second derivative is exactly 0.
                        continue
                    expo = eq.copy()
                    expo[j] -= 1
                    expo[k] -= 1
                    v = np.ones(n)
                    for l in nz:
                        v = v * Q[:, l] ** expo[l]
                    if E is not None:
                        for l in np.nonzero(es)[0]:
                            v = v * E[:, l] ** es[l]
                    block = coeff * mult * v
                    out[:, j, k, ti] += block
                    if k != j:
                        out[:, k, j, ti] += block
        return out


# ---------------------------------------------------------------------------
# mapping resolution / Q recovery
# ---------------------------------------------------------------------------
def _resolve_M(mapping) -> np.ndarray:
    M = getattr(mapping, "mapping_mat", mapping)
    if hasattr(M, "toarray"):
        M = M.toarray()
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] % 3 != 0:
        raise ValueError(
            f"mapping must be (3*natom_sc, nQ), got {np.shape(mapping)}"
        )
    return M


def _resolve_sparse_M(mapping):
    """The mapping as scipy sparse (for lsqr), or None if it was dense."""
    M = getattr(mapping, "mapping_mat", mapping)
    if hasattr(M, "tocsr"):
        return M.tocsr()
    return None


def _strain_needed(coefficients: AnharmonicCoefficients) -> bool:
    return any(
        t.sector in ("strain", "coupled") for t in coefficients.basis.terms
    )


def _pick_solver(M_sparse, nQ: int, solver: str):
    if solver == "auto":
        solver = "pinv" if nQ <= DENSE_PINV_MAX_NQ else "lsqr"
    if solver not in ("pinv", "lsqr"):
        raise ValueError(f"solver must be auto/pinv/lsqr, got {solver!r}")
    if solver == "lsqr" and M_sparse is None:
        solver = "pinv"  # dense mapping: pinv is exact and cheap at this size
    return solver


def _recover_amplitudes(atoms, M, ref_atoms, solver, M_sparse):
    """Deformation gradient, strain Voigt, displacement and least-squares Q.

    Returns ``(Q, eps_voigt, residual_norm, asym_scale)`` where
    ``asym_scale`` is the relative asymmetry of the recovered deformation
    gradient ``F`` (column convention, see below).
    """
    cell = np.asarray(atoms.get_cell().array, dtype=float)
    ref_cell = np.asarray(ref_atoms.get_cell().array, dtype=float)
    # ase.set_cell(scale_atoms=True) preserves fractional coordinates, so the
    # positions transform as column vectors x -> F x with
    # F = (inv(cell_ref) @ cell).T = (I + eps) for cell = cell_ref @ (I + eps)
    F = (np.linalg.inv(ref_cell) @ cell).T
    F_sym = 0.5 * (F + F.T)
    asym_scale = float(np.abs(F - F.T).max() / max(1.0, np.abs(F).max()))
    eps_voigt = matrix_to_voigt(F_sym - np.eye(3))
    # exact inverse of sampling.make_atoms: positions were built as
    # x = F (x_ref + M Q); un-scale FIRST, then subtract the reference
    u = (
        np.linalg.solve(F, atoms.get_positions().T).T
        - ref_atoms.get_positions()
    ).ravel()
    if solver == "pinv":
        Q = np.linalg.pinv(M) @ u
    else:
        from scipy.sparse.linalg import lsqr

        Q = lsqr(M_sparse, u, atol=1e-13, btol=1e-13)[0]
    resid = float(np.linalg.norm(u - M @ Q))
    return Q, eps_voigt, resid, asym_scale


def evaluate_atoms(
    atoms,
    model: "AnharmonicModel",
    mapping,
    ref_atoms=None,
    solver: str = "auto",
    residual_tol: float = 1e-10,
    primitive_volume: Optional[float] = None,
    warn_residual: bool = True,
) -> dict:
    """Evaluate ``model`` on an ``ase.Atoms`` frame.

    Returns a dict with ``energy`` (float), ``forces`` (natom, 3),
    ``stress`` (Voigt 6), ``Q`` (nQ,), ``strain_voigt`` (6,) and
    ``residual_norm`` (out-of-subspace displacement norm).  See the module
    docstring for the convention chain.  :class:`LWFModelCalculator` is a
    thin ASE wrapper around this function.
    """
    M = _resolve_M(mapping)
    nQ = M.shape[1]
    if nQ != model.nQ:
        raise ValueError(
            f"mapping has nQ={nQ} amplitudes but the model expects {model.nQ}"
        )
    M_sparse = _resolve_sparse_M(mapping)
    if ref_atoms is None:
        ref_atoms = getattr(mapping, "sc_atoms", None)
    if ref_atoms is None:
        raise ValueError(
            "reference atoms required: pass ref_atoms (the undisplaced "
            "supercell) or a mapping exposing .sc_atoms"
        )

    Q, eps_voigt, resid, asym_scale = _recover_amplitudes(
        atoms, M, ref_atoms, _pick_solver(M_sparse, nQ, solver), M_sparse
    )
    u_norm = max(1.0, float(np.linalg.norm(atoms.get_positions())))
    if warn_residual and resid > residual_tol * u_norm:
        warnings.warn(
            LWFSubspaceResidualWarning(
                f"displacement leaves the model subspace: ||u - M M+ u|| = "
                f"{resid:.3e}; evaluating at the projected amplitudes",
                residual_norm=resid,
            )
        )
    if _strain_needed(model.coefficients) and asym_scale > 1e-8:
        warnings.warn(
            f"cell deformation is not symmetric (max |F - F^T| = "
            f"{asym_scale:.3e}); using the symmetric part as strain"
        )

    E = model.energy(Q, eps_voigt)
    g = model.gradient(Q, eps_voigt)
    F = -(M @ np.linalg.solve(M.T @ M, g)).reshape(-1, 3)
    sigma = model.stress(Q, eps_voigt, primitive_volume=primitive_volume)
    return {
        "energy": float(E),
        "forces": F,
        "stress": np.asarray(sigma, dtype=float),
        "Q": Q,
        "strain_voigt": eps_voigt,
        "residual_norm": resid,
    }


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class AnharmonicModel:
    """Evaluator for a fitted :class:`AnharmonicCoefficients`.

    Parameters
    ----------
    coefficients:
        The fit result (harmonic baseline included when present).
    primitive_volume:
        Reference per-primitive-cell volume in A^3, required for
        :meth:`stress`.  ``LWFModelCalculator`` derives it from the
        reference supercell when not given.
    """

    def __init__(
        self,
        coefficients: AnharmonicCoefficients,
        primitive_volume: Optional[float] = None,
    ):
        self.coefficients = coefficients
        self.primitive_volume = (
            None if primitive_volume is None else float(primitive_volume)
        )
        self._grad = None
        self._hess = None

    # -- lazy exact derivative machinery ------------------------------------
    @property
    def _poly_grad(self) -> _BasisGrad:
        if self._grad is None:
            self._grad = _BasisGrad(self.coefficients.basis)
        return self._grad

    @property
    def _poly_hess(self) -> _BasisHess:
        if self._hess is None:
            self._hess = _BasisHess(self.coefficients.basis)
        return self._hess

    # -- sizes ---------------------------------------------------------------
    @property
    def basis(self) -> InvariantBasis:
        return self.coefficients.basis

    @property
    def harmonic(self):
        return self.coefficients.harmonic

    @property
    def nQ(self) -> int:
        """Dataset-order amplitude count (supercell cells x branches)."""
        if self.coefficients.harmonic is not None:
            return int(self.coefficients.harmonic.nQ)
        return len(self.coefficients.basis.coord_labels)

    # -- evaluation ----------------------------------------------------------
    def energy(self, Q, strain=None) -> float:
        """Total energy of one frame (harmonic baseline included)."""
        return self.coefficients.energy(Q, strain)

    def gradient(self, Q, strain=None) -> np.ndarray:
        """dE/dQ in the dataset (cell-major) coordinate order."""
        return self.coefficients.gradient(Q, strain)

    def energy_and_gradient(self, Q, strain=None):
        return self.energy(Q, strain), self.gradient(Q, strain)

    def stress(
        self,
        Q,
        strain=None,
        primitive_volume: Optional[float] = None,
    ) -> np.ndarray:
        """``sigma_v = dE/deps_v / vol`` with ``vol = V0 * det(I + eps)`` (the
        fit.py convention, stored-once Voigt components).  Zero for a basis
        without strain terms (the harmonic baseline carries no strain
        dependence)."""
        V0 = primitive_volume
        if V0 is None:
            V0 = self.primitive_volume
        c = self.coefficients
        if not _strain_needed(c):
            return np.zeros(6)
        if strain is None:
            raise ValueError(
                "the basis contains strain terms; stress at an unstrained "
                "frame requires strain=zeros(6)"
            )
        if V0 is None:
            raise ValueError(
                "primitive_volume is required for stress: pass it to "
                "AnharmonicModel(primitive_volume=...) or to .stress()"
            )
        st = np.asarray(strain, dtype=float)
        Q2 = np.asarray(Q, dtype=float)
        if Q2.ndim == 1:
            Q2 = Q2[None, :]
        st2 = st if st.ndim == 2 else st[None, :]
        Qp = Q2[:, c.coord_perm] if c.coord_perm is not None else Q2
        g_eps = self._poly_grad.grad_eps(Qp, st2)
        ane = (g_eps @ c.coefficients)[0]
        vol = V0 * float(np.linalg.det(np.eye(3) + voigt_to_matrix(st)))
        return np.asarray(ane, dtype=float) / vol

    def hessian(self, Q, strain=None) -> np.ndarray:
        """Exact d2E/dQ2 (harmonic kernel + exact monomial second
        derivatives), dataset coordinate order."""
        c = self.coefficients
        Q2 = np.asarray(Q, dtype=float)
        single = Q2.ndim == 1
        if single:
            Q2 = Q2[None, :]
        st2 = None
        if self._poly_hess.needs_strain:
            if strain is None:
                raise ValueError("basis contains strain terms; pass strain")
            st2 = np.asarray(strain, dtype=float)
            if st2.ndim == 1:
                st2 = st2[None, :]
        Qp = Q2[:, c.coord_perm] if c.coord_perm is not None else Q2
        H = self._poly_hess.hess_q(Qp, st2) @ c.coefficients  # (f, n, n)
        if c.coord_perm is not None:
            inv = np.argsort(c.coord_perm)
            H = H[:, inv[:, None], inv[None, :]]
        if c.harmonic is not None:
            H = H + c.harmonic.hessian()[None, :, :]
        return H[0] if single else H

    def __repr__(self):
        return (
            f"AnharmonicModel(nQ={self.nQ}, "
            f"n_terms={len(self.coefficients.basis.terms)}, "
            f"selected={len(self.coefficients.selected)}, "
            f"harmonic={self.coefficients.harmonic is not None})"
        )


# ---------------------------------------------------------------------------
# ASE calculator
# ---------------------------------------------------------------------------
class LWFModelCalculator(Calculator):
    """ASE calculator wrapping an :class:`AnharmonicModel`.

    Parameters
    ----------
    model:
        :class:`AnharmonicModel`.
    mapping:
        A ``MyLWFSC`` (its ``mapping_mat`` and ``sc_atoms`` are used), a raw
        mapping matrix (then ``ref_atoms`` is required), or anything else
        exposing ``mapping_mat``.
    ref_atoms:
        Reference (undisplaced, unstrained) supercell; defaults to
        ``mapping.sc_atoms``.
    solver:
        ``"auto"`` (dense pinv at nQ <= 10, sparse lsqr above), ``"pinv"``
        or ``"lsqr"``.

    ADR-010 validity note: the mapping ``M`` is kept at its REFERENCE value
    for strained cells; the strain enters only through the energy's strain
    parameters recovered from ``cell = cell_ref @ (I + eps)``.  This is the
    small-strain regime (exact to first order in eps).
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: AnharmonicModel,
        mapping,
        ref_atoms=None,
        solver: str = "auto",
        residual_tol: float = 1e-10,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self._mapping_obj = mapping
        self._M = _resolve_M(mapping)
        if self._M.shape[1] != model.nQ:
            raise ValueError(
                f"mapping has nQ={self._M.shape[1]} amplitudes but the model "
                f"expects {model.nQ}"
            )
        self._M_sparse = _resolve_sparse_M(mapping)
        if ref_atoms is None:
            ref_atoms = getattr(mapping, "sc_atoms", None)
        if ref_atoms is None:
            raise ValueError(
                "reference atoms required: pass ref_atoms (the undisplaced "
                "supercell) or a mapping exposing .sc_atoms"
            )
        self._ref_atoms = ref_atoms
        self._solver = _pick_solver(self._M_sparse, self._M.shape[1], solver)
        self.residual_tol = float(residual_tol)
        self._V0 = None
        if model.primitive_volume is not None:
            self._V0 = float(model.primitive_volume)

    def _reference_volume(self) -> Optional[float]:
        """Reference per-primitive-cell volume, if it can be determined."""
        if self._V0 is not None:
            return self._V0
        sc_matrix = None
        sm = getattr(self._mapping_obj, "scmaker", None)
        sc_matrix = getattr(sm, "sc_matrix", None)
        if sc_matrix is None:
            sc_matrix = getattr(self._mapping_obj, "sc_matrix", None)
        if sc_matrix is None:
            hb = self.model.coefficients.harmonic
            if hb is not None:
                sc_matrix = np.asarray(hb.sc_matrix, dtype=float)
        if sc_matrix is None:
            return None
        vol_sc = abs(
            np.linalg.det(np.asarray(self._ref_atoms.get_cell().array, dtype=float))
        )
        det_S = abs(np.linalg.det(np.asarray(sc_matrix, dtype=float)))
        return vol_sc / det_S

    def reconstruct_amplitudes(self, atoms):
        """``(Q, strain_voigt, residual_norm)`` recovered from ``atoms``."""
        Q, eps, resid, _ = _recover_amplitudes(
            atoms, self._M, self._ref_atoms, self._solver, self._M_sparse
        )
        return Q, eps, resid

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces", "stress"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        V0 = self._V0
        if V0 is None and _strain_needed(self.model.coefficients):
            V0 = self._reference_volume()
            if V0 is None:
                raise ValueError(
                    "cannot determine the reference primitive volume for the "
                    "stress: pass primitive_volume to AnharmonicModel"
                )
        out = evaluate_atoms(
            self.atoms,
            self.model,
            self._M,
            ref_atoms=self._ref_atoms,
            solver=self._solver,
            residual_tol=self.residual_tol,
            primitive_volume=V0,
        )
        self.results = {
            "energy": out["energy"],
            "forces": out["forces"],
            "stress": out["stress"],
        }
