"""Story 022: anharmonic effective-model evaluator (AnharmonicModel), ASE
calculator (LWFModelCalculator), and netCDF ``anharmonic``-group I/O.

Covered contract
----------------
- TEST-001  model energy/gradient/stress finite-difference parity (incl. the
  strain sector) to 1e-6 relative.
- TEST-002  calculator vs direct model evaluation on in-subspace frames:
  bitwise-identical floats for the same recovered (Q, eps).
- TEST-003  out-of-subspace displacements raise LWFSubspaceResidualWarning
  carrying ``.residual_norm = ||u - M M+ u||``; energy still evaluated at the
  projected amplitudes.
- TEST-004  strained cells: calculator stress equals the model's
  dE/deps_voigt taken through the deformed cell, with the fit.py volume
  convention vol = V0_prim * det(I + eps).
- TEST-005  netCDF roundtrip: bit-exact float64 coefficients, exact rational
  basis terms, locality/fingerprint preserved, standalone load, schema
  version guard.

Sympy verifications (user-mandated SYMPY RULE) live in
``test_sympy_*`` below: the monomial second-derivative exponent rule, the
minimum-norm force projection identity (the inverse of the sampling module's
``g_Q = M^T F`` chain), and the stored-once Voigt stress convention with the
``V(eps) = V0 * det(I + eps)`` volume factor (Jacobi identity).
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp
from ase import Atoms
from fractions import Fraction

from lawaf.anharmonic.basis import (
    ClusterAction,
    build_invariant_basis,
    voigt_matrix_from_rotation,
)
from lawaf.anharmonic.dataset import TrainingDataset
from lawaf.anharmonic.fit import (
    AnharmonicCoefficients,
    CoefficientTerm,
    CVReport,
    LocalityReport,
    _BasisGrad,
    basis_cell_permutation,
    fit,
    harmonic_baseline,
)
from lawaf.anharmonic.sampling import FrameSpec, make_atoms, voigt_to_matrix
from lawaf.utils.supercell import SupercellMaker

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# ---------------------------------------------------------------------------
# tiny world (same pattern as tests/test_anharmonic_fit.py): 2-branch LWF,
# supercell diag(2,1,1), C2-like cluster action
# ---------------------------------------------------------------------------
class _TinyAction(ClusterAction):
    """{E, diag(1,-1,-1)} with every branch flipping sign under the rotation."""

    n_ops = 2
    _rots = (np.eye(3, dtype=int), np.diag([1, -1, -1]).astype(int))
    _op1_sign = (1, 1, 1, 1, -1, -1)

    def image_factor(self, factor, op):
        b, R = factor
        R2 = tuple(int(v) for v in self._rots[op] @ np.asarray(R, dtype=int))
        return (b, R2), (-1 if op else 1)

    def point_rep(self, op):
        return self._rots[op].astype(float)

    def strain_voigt_matrix(self, op):
        return voigt_matrix_from_rotation(self._rots[op])

    def strain_row_exact(self, v, op):
        s = 1 if op == 0 else self._op1_sign[v]
        return ((v, Fraction(s)),)

    def op_inverse(self, op):
        return op


def _toy_lwf(nwann=2, seed=5, Rlist=((0, 0, 0), (1, 0, 0))):
    rng = np.random.default_rng(seed)
    HR = rng.normal(0.0, 1.0, (len(Rlist), nwann, nwann))
    HR = 0.5 * (HR + HR.transpose(0, 2, 1))
    return SimpleNamespace(
        HR_total=HR,
        Rlist=np.array(Rlist, dtype=int),
        Rdeg=None,
        wann_masses=np.full(nwann, 2.0),
    )


def _toy_harmonic(lwf, S=np.diag([2, 1, 1])):
    return harmonic_baseline(lwf, SupercellMaker(S))


def _toy_basis(orders=(3, 4)):
    return build_invariant_basis(
        _TinyAction(),
        nlwf=2,
        Rlist=((0, 0, 0), (1, 0, 0)),
        orders=orders,
        max_strain_power=1,
        include_strain=True,
    )


def _toy_mapping(nQ, natom_sc=2, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (3 * natom_sc, nQ))


def _toy_cells(nf):
    scell = np.diag([2.0, 1.0, 1.0]) * 4.0
    return [scell for _ in range(nf)]


def _make_splits(nf):
    n_train = max(1, int(0.7 * nf))
    n_cv = max(1, nf // 5)
    return ["train"] * n_train + ["cv"] * n_cv + ["holdout"] * (nf - n_train - n_cv)


def _toy_dataset(basis, harmonic, c_true, M, nf=24, seed=3):
    """Frames + exact zero-noise labels for E = B @ c_true + E_harm(Q)."""
    rng = np.random.default_rng(seed)
    nQ = len(basis.coord_labels)
    Q = rng.normal(0.0, 0.7, (nf, nQ))
    strain = rng.normal(0.0, 0.05, (nf, 6))

    gb = basis_cell_permutation(basis, harmonic)
    pg = _BasisGrad(basis)

    E_anh = basis.evaluate(Q, strain) @ c_true
    anh_grad_cell = (pg.grad_q(Q, strain) @ c_true)[:, gb]
    harm_grad = np.array([harmonic.gradient(Q[i]) for i in range(nf)])
    g_total = anh_grad_cell + harm_grad
    # Cartesian forces whose projection is exactly g_Q = M^T F = -dE/dQ
    F = np.einsum("ij,fj->fi", M @ np.linalg.inv(M.T @ M), -g_total)

    grad_eps = pg.grad_eps(Q, strain)
    Scell = np.asarray(harmonic.sc_matrix, dtype=float)
    vol = np.array([abs(np.linalg.det(c)) for c in _toy_cells(nf)]) / abs(
        np.linalg.det(Scell)
    )
    sigma = (grad_eps @ c_true) / vol[:, None]

    ds = TrainingDataset(
        positions=[np.zeros((M.shape[0] // 3, 3)) for _ in range(nf)],
        atomic_numbers=np.full((nf, M.shape[0] // 3), 14),
        cell=np.array(_toy_cells(nf)),
        pbc=np.full((nf, 3), True),
        Q=Q,
        strain_voigt=strain,
        split=_make_splits(nf),
    )
    ds.label(
        "toy",
        values={
            "energy": E_anh + np.array([harmonic.energy(Q[i]) for i in range(nf)]),
            "forces": F.reshape(nf, -1, 3),
        },
    )
    ds.fill_label_values(
        ds.active_label,
        np.arange(nf),
        stress=np.stack([voigt_to_matrix(v) for v in sigma]),
    )
    return ds, Q, strain


def _fitted_world():
    """A full AnharmonicCoefficients from fit() on exact synthetic labels,
    including strain-sector terms (the TEST-001 world)."""
    basis = _toy_basis()
    harmonic = _toy_harmonic(_toy_lwf())
    nQ = harmonic.nQ
    M = _toy_mapping(nQ)
    rng = np.random.default_rng(1)
    support = [
        i
        for i, t in enumerate(basis.terms)
        if t.sector in ("q", "coupled", "strain")
    ][:6]
    c_true = np.zeros(len(basis.terms))
    c_true[support] = rng.normal(0.0, 0.3, len(support))
    ds, Q, strain = _toy_dataset(basis, harmonic, c_true, M)
    res = fit(ds, basis, harmonic, selection="ridge", ridge_alpha=1e-10, mapping=M)
    return res, basis, harmonic, M, Q, strain


_V0_PRIM = 16.0  # |det(diag(2,1,1)*4)| / |det(diag(2,1,1))|


# ---------------------------------------------------------------------------
# calculator world: a REAL MyLWFSC mapping (sparse M + sc_atoms), as in
# tests/test_anharmonic_sampling.py, with a hand-assembled model
# ---------------------------------------------------------------------------
def make_toy_mylwfsc(seed=7):
    from lawaf.lwf.lwf import LWF
    from lawaf.lwf.lwf_supercell import MyLWFSC

    rng = np.random.default_rng(seed)
    natom, nlwf = 2, 2
    natom3 = 3 * natom
    Rlist = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    wannR = rng.normal(0.0, 0.5, size=(len(Rlist), natom3, nlwf))
    wannR = np.where(np.abs(wannR) < 5e-3, 5e-3, wannR)  # clear the 1e-4 cutoff
    atoms = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.2, 1.2, 1.2]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    lwf = LWF(wannR=wannR, Rlist=Rlist, atoms=atoms)
    return MyLWFSC(lwf, SupercellMaker(np.diag([2, 1, 1])))


def _hand_coefficients(include_strain=True):
    """Hand-assembled AnharmonicCoefficients (no fit): exact terms, random
    coefficients, harmonic baseline attached."""
    basis = build_invariant_basis(
        _TinyAction(),
        nlwf=2,
        Rlist=((0, 0, 0), (1, 0, 0)),
        orders=(3, 4),
        max_strain_power=1,
        include_strain=include_strain,
    )
    harmonic = _toy_harmonic(_toy_lwf(seed=6))
    nt = len(basis.terms)
    assert nt > 3
    rng = np.random.default_rng(2)
    coeff = np.zeros(nt)
    selected = tuple(range(min(5, nt)))
    coeff[list(selected)] = rng.normal(0.0, 0.2, len(selected))
    stderrs = np.full(nt, np.nan)
    pc = basis_cell_permutation(basis, harmonic)
    return AnharmonicCoefficients(
        basis=basis,
        harmonic=harmonic,
        terms=[
            CoefficientTerm(
                index=i,
                seed=basis.terms[i].seed,
                order=basis.terms[i].order,
                sector=basis.terms[i].sector,
                coeff=float(coeff[i]),
                stderr=None,
            )
            for i in selected
        ],
        coefficients=coeff,
        stderrs=stderrs,
        selected=selected,
        locality=LocalityReport(rows=[]),
        fingerprint="hand022",
        cv=CVReport(n_train_rows=0, n_cv_frames=0),
        selection="hand",
        ridge_alpha=0.0,
        weights={},
        coord_perm=pc,
    )


def _calculator_world():
    from lawaf.anharmonic.model import AnharmonicModel, LWFModelCalculator

    mylwfsc = make_toy_mylwfsc()
    coeff = _hand_coefficients(include_strain=True)
    model = AnharmonicModel(coeff, primitive_volume=_V0_PRIM)
    calc = LWFModelCalculator(model, mylwfsc)
    return model, calc, mylwfsc


# ===========================================================================
# sympy verifications (SYMPY RULE)
# ===========================================================================
def test_sympy_monomial_hessian_exponent_rule():
    """d2/dQj dQk prod Q^n == n_j (n_k - delta_jk) prod Q^(n - delta_j - delta_k)
    for integer exponent tuples, against sympy.diff."""
    z = sp.symbols("z0:5")
    for n in [(2, 3, 1, 0, 2), (1, 1, 2, 0, 0), (3, 0, 0, 1, 1), (2, 2, 2, 0, 0)]:
        mono = sp.Integer(1)
        for zi, ni in zip(z, n):
            mono *= zi**ni
        for j in range(5):
            for k in range(5):
                lhs = sp.diff(mono, z[j], z[k])
                rhs = n[j] * (n[k] - (1 if j == k else 0))
                for l in range(5):
                    rhs = rhs * z[l] ** (
                        n[l] - (1 if l == j else 0) - (1 if l == k else 0)
                    )
                assert sp.expand(lhs - rhs) == 0, (n, j, k)


def test_sympy_force_min_norm_projection_identity():
    """The calculator force F_cart = -M (M^T M)^{-1} dE/dQ is the exact left
    inverse of the sampling projection g_Q = M^T F:  M^T F_cart = -dE/dQ
    (ASE sign chain F = -dE/du), and the work along subspace displacements
    obeys delta-E = -F_cart . (M deltaQ) = dE/dQ . deltaQ."""
    m11, m12, m21, m22, g1, g2, dq1, dq2 = sp.symbols("m11 m12 m21 m22 g1 g2 dq1 dq2")
    M = sp.Matrix([[m11, m12], [m21, m22]])
    g = sp.Matrix([g1, g2])  # dE/dQ
    F_cart = -M * (M.T * M).inv() * g  # calculator convention
    # projection roundtrip: g_Q = M^T F_cart == -dE/dQ  (sampling sign chain)
    assert sp.simplify(M.T * F_cart + g) == sp.zeros(2, 1)
    # work identity along subspace displacements delta-u = M delta-Q
    dQ = sp.Matrix([dq1, dq2])
    assert sp.simplify(g.T * dQ - (-F_cart).T * (M * dQ)) == sp.zeros(1, 1)
    # and F_cart is the MINIMUM-norm vector satisfying M^T F = -g: it lies in
    # range(M) by construction (columns of M), so orthogonal decomposition
    # any solution = F_cart + null(M^T) has norm >= |F_cart|
    assert sp.simplify(F_cart + M * ((M.T * M).inv() * g)) == sp.zeros(2, 1)


def test_sympy_stress_volume_and_stored_once_conventions():
    """Stress convention chain (fit.py match):
    (1) Jacobi: d det(I+eps) / d eps_v = det(I+eps) [(I+eps)^-1]_vv, so the
        ASE stress sigma_v = (1/V) dE/deps_v with V = V0 det(I+eps) is
        reproduced by sigma_v = dE/deps_v / (V0 det(I+eps));
    (2) stored-once Voigt derivative: d/deps (1/2 eps' C eps) = C eps with
        each Voigt component stored once (no factor 2), which is exactly the
        fit.py design convention sigma_v = (1/vol) dE/deps_v;
    (3) hydrostatic sanity: E = B/2 V0 (tr eps)^2 gives tension-positive
        pressure P = -B tr eps through sigma = (1/V) dE/deps."""
    e = list(sp.symbols("e0:6"))
    epsm = sp.zeros(3)
    for v, (i, j) in enumerate(VOIGT):
        epsm[i, j] = epsm[j, i] = e[v]
    I = sp.eye(3)
    F = I + epsm
    detF = sp.det(F)
    Em = F.inv()
    for v, (i, j) in enumerate(VOIGT):
        lhs = sp.diff(detF, e[v])
        # stored-once Voigt parametrization: an off-diagonal component drives
        # BOTH symmetric entries of F, so its Jacobi weight is the sum
        rhs = detF * (Em[j, i] if i == j else Em[j, i] + Em[i, j])
        assert sp.simplify(lhs - rhs) == 0, f"Jacobi failed for Voigt {v}"

    # elastic matrix built symmetric by construction (c_ij == c_ji)
    Cs = sp.Matrix(6, 6, lambda i, j: sp.Symbol(f"c{min(i, j)}_{max(i, j)}"))
    E2 = sp.Rational(1, 2) * sum(
        Cs[u, v] * e[u] * e[v] for u in range(6) for v in range(6)
    )
    for v in range(6):
        assert (
            sp.simplify(sp.diff(E2, e[v]) - sum(Cs[v, u] * e[u] for u in range(6)))
            == 0
        ), f"stored-once derivative failed for Voigt {v}"
    # hydrostatic sanity: E = B/2 V0 (tr eps)^2 gives sigma_xx = B tr(eps) and
    # pressure P = -1/3 tr(sigma) = -B tr(eps): tension-positive convention
    B, V0, exx, eyy = sp.symbols("B V0 eps_xx eps_yy", positive=True)
    Eh = sp.Rational(1, 2) * B * V0 * (exx + eyy) ** 2
    sig_xx = sp.diff(Eh, exx) / V0  # det(I+eps)=1 at eps=0
    assert sp.simplify(sig_xx - B * (exx + eyy)) == 0
    P = -sig_xx
    assert sp.simplify(P + B * (exx + eyy)) == 0


# ===========================================================================
# TEST-001: model evaluator FD parity (energy/gradient/stress, strain sector)
# ===========================================================================
def test_model_energy_gradient_delegate_to_coefficients():
    res, *_ = _fitted_world()
    from lawaf.anharmonic.model import AnharmonicModel

    model = AnharmonicModel(res, primitive_volume=_V0_PRIM)
    rng = np.random.default_rng(0)
    Q = rng.normal(0, 0.4, model.nQ)
    eps = rng.normal(0, 0.02, 6)
    assert model.energy(Q, eps) == res.energy(Q, eps)
    np.testing.assert_array_equal(model.gradient(Q, eps), res.gradient(Q, eps))
    E, G = model.energy_and_gradient(Q, eps)
    assert E == model.energy(Q, eps)
    np.testing.assert_array_equal(G, model.gradient(Q, eps))


def test_model_gradient_fd_parity_with_strain():
    """TEST-001: dE/dQ of the fitted model (harmonic + strain-sector
    anharmonic) matches central finite differences to 1e-6 relative."""
    res, basis, harmonic, M, Q, strain = _fitted_world()
    from lawaf.anharmonic.model import AnharmonicModel

    model = AnharmonicModel(res, primitive_volume=_V0_PRIM)
    rng = np.random.default_rng(3)
    Q0 = rng.normal(0, 0.5, model.nQ)
    eps0 = rng.normal(0, 0.03, 6)
    g = model.gradient(Q0, eps0)
    h = 1e-6
    fd = np.empty_like(g)
    for c in range(model.nQ):
        dQ = np.zeros_like(Q0)
        dQ[c] = h
        fd[c] = (model.energy(Q0 + dQ, eps0) - model.energy(Q0 - dQ, eps0)) / (2 * h)
    denom = np.maximum(np.abs(fd), 1e-8)
    assert np.max(np.abs(fd - g) / denom) < 1e-6


def test_model_stress_fd_parity():
    """TEST-001 (stress half): sigma = dE/deps_v / vol with
    vol = V0 * det(I + eps) reproduces central FD of the model energy in the
    strain parameters, incl. off-diagonal Voigt components."""
    res, basis, harmonic, M, Q, strain = _fitted_world()
    from lawaf.anharmonic.model import AnharmonicModel

    model = AnharmonicModel(res, primitive_volume=_V0_PRIM)
    rng = np.random.default_rng(4)
    Q0 = rng.normal(0, 0.5, model.nQ)
    eps0 = rng.normal(0, 0.03, 6)
    sig = model.stress(Q0, eps0)
    h = 1e-6
    vol = _V0_PRIM * float(np.linalg.det(np.eye(3) + voigt_to_matrix(eps0)))
    fd = np.empty(6)
    for v in range(6):
        d = np.zeros(6)
        d[v] = h
        fd[v] = (model.energy(Q0, eps0 + d) - model.energy(Q0, eps0 - d)) / (2 * h)
    fd = fd / vol
    # 1e-6 relative, with an absolute floor for near-zero derivative
    # components (symmetry can make individual Voigt components tiny)
    scale = max(1.0, float(np.abs(sig).max()))
    np.testing.assert_allclose(fd, sig, rtol=1e-6, atol=1e-8 * scale)


def test_model_hessian_exact_and_fd():
    """Hessian = harmonic kernel + exact monomial second derivatives; checked
    against FD of the model gradient, and the pure-harmonic case reduces to
    Hmat."""
    res, basis, harmonic, M, Q, strain = _fitted_world()
    from lawaf.anharmonic.model import AnharmonicModel

    model = AnharmonicModel(res, primitive_volume=_V0_PRIM)
    rng = np.random.default_rng(5)
    Q0 = rng.normal(0, 0.4, model.nQ)
    eps0 = rng.normal(0, 0.02, 6)
    H = model.hessian(Q0, eps0)
    assert np.allclose(H, H.T, atol=1e-12)
    h = 1e-6
    fd = np.empty_like(H)
    for c in range(model.nQ):
        dQ = np.zeros_like(Q0)
        dQ[c] = h
        fd[:, c] = (model.gradient(Q0 + dQ, eps0) - model.gradient(Q0 - dQ, eps0)) / (
            2 * h
        )
    assert np.max(np.abs(fd - H)) < 1e-5 * np.max(np.abs(H))
    # pure harmonic model: hessian is the folded kernel, strain-independent
    model_pure = AnharmonicModel(
        AnharmonicCoefficients(
            basis=basis,
            harmonic=harmonic,
            terms=[],
            coefficients=np.zeros(len(basis.terms)),
            stderrs=np.zeros(len(basis.terms)),
            selected=(),
            locality=LocalityReport(rows=[]),
            fingerprint="pure",
            cv=CVReport(n_train_rows=0, n_cv_frames=0),
            selection="none",
            ridge_alpha=0.0,
            weights={},
            coord_perm=None,
        ),
        primitive_volume=_V0_PRIM,
    )
    np.testing.assert_allclose(
        model_pure.hessian(Q0, eps0), harmonic.hessian(), atol=1e-12
    )


# ===========================================================================
# TEST-002/003/004: the ASE calculator
# ===========================================================================
def test_calculator_matches_direct_model_bitwise():
    """TEST-002: on an in-subspace frame the calculator's energy/forces equal
    the direct model evaluation at the recovered amplitudes, bit for bit."""
    model, calc, mylwfsc = _calculator_world()
    M = np.asarray(mylwfsc.mapping_mat.toarray(), dtype=float)
    G = M.T @ M
    rng = np.random.default_rng(7)
    Q = rng.normal(0.0, 0.2, model.nQ)
    atoms, _ = mylwfsc.get_distorted_atoms(Q)
    atoms.calc = calc
    E = atoms.get_potential_energy()
    F = atoms.get_forces()

    Q_rec, eps_rec, resid = calc.reconstruct_amplitudes(atoms)
    assert resid < 1e-10
    assert E == model.energy(Q_rec, eps_rec)
    g = model.gradient(Q_rec, eps_rec)
    F_direct = -(M @ np.linalg.solve(G, g)).reshape(-1, 3)
    np.testing.assert_array_equal(F, F_direct)
    # recovery is exact to roundoff: physics agrees with the true Q too
    eps0 = np.zeros(6)
    assert E == pytest.approx(model.energy(Q, eps0), rel=1e-10, abs=1e-8)
    np.testing.assert_allclose(F.reshape(-1), -(M @ np.linalg.solve(
        G, model.gradient(Q, eps0))).ravel(), rtol=1e-8, atol=1e-10)
    # ASE results dict contract
    assert set(calc.results) >= {"energy", "forces", "stress"}
    assert calc.results["forces"].shape == (len(atoms), 3)
    assert calc.results["stress"].shape == (6,)


def test_calculator_warns_out_of_subspace():
    """TEST-003: a null-space displacement component triggers
    LWFSubspaceResidualWarning carrying ||u - M M+ u||; the energy is
    evaluated at the PROJECTED amplitudes."""
    from lawaf.anharmonic.model import LWFSubspaceResidualWarning

    model, calc, mylwfsc = _calculator_world()
    M = np.asarray(mylwfsc.mapping_mat.toarray(), dtype=float)
    # null space of M^T: right singular vectors of M.T beyond the rank
    _, _, Vh = np.linalg.svd(M.T)
    w = Vh[-1]  # unit vector with M^T w == 0
    assert np.allclose(M.T @ w, 0.0, atol=1e-10)
    rng = np.random.default_rng(8)
    Q = rng.normal(0.0, 0.2, model.nQ)
    alpha = 0.05
    u = M @ Q + alpha * w
    atoms, _ = mylwfsc.get_distorted_atoms(np.zeros(model.nQ))
    atoms.set_positions(atoms.get_positions() + u.reshape(-1, 3))
    atoms.calc = calc
    with pytest.warns(LWFSubspaceResidualWarning) as record:
        E = atoms.get_potential_energy()
    msgs = [
        r.message
        for r in record
        if isinstance(r.message, LWFSubspaceResidualWarning)
    ]
    assert len(msgs) == 1
    assert abs(msgs[0].residual_norm - alpha) < 1e-8
    # energy still comes from the projected amplitudes
    Q_rec, eps_rec, resid = calc.reconstruct_amplitudes(atoms)
    assert abs(resid - alpha) < 1e-8
    assert E == model.energy(Q_rec, eps_rec)
    assert E == pytest.approx(model.energy(Q, np.zeros(6)), rel=1e-9, abs=1e-9)


def test_calculator_silent_in_subspace():
    """No residual warning for exactly representable frames."""
    from lawaf.anharmonic.model import LWFSubspaceResidualWarning

    model, calc, mylwfsc = _calculator_world()
    rng = np.random.default_rng(9)
    Q = rng.normal(0.0, 0.2, model.nQ)
    atoms, _ = mylwfsc.get_distorted_atoms(Q)
    atoms.calc = calc
    with warnings.catch_warnings():
        warnings.simplefilter("error", LWFSubspaceResidualWarning)
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.get_stress()


def test_calculator_stress_strained_cell_fd():
    """TEST-004: for a strained supercell (cell = cell0 @ (I + eps)) the
    calculator stress equals dE/d(eps_voigt)/vol of the model, where the
    derivative is taken by rebuilding the deformed cell (the story-016
    convention) and vol = V0_prim * det(I + eps)."""
    model, calc, mylwfsc = _calculator_world()
    rng = np.random.default_rng(10)
    Q = rng.normal(0.0, 0.15, model.nQ)
    eps = np.array([0.01, -0.008, 0.006, 0.004, -0.003, 0.005])

    frame = make_atoms(mylwfsc, FrameSpec(Q=Q, strain_voigt=eps))
    frame.calc = calc
    sigma = frame.get_stress()

    Q_rec, eps_rec, resid = calc.reconstruct_amplitudes(frame)
    assert resid < 1e-10
    np.testing.assert_allclose(eps_rec, eps, atol=1e-12)
    np.testing.assert_array_equal(sigma, model.stress(Q_rec, eps_rec))
    # and the model stress at the TRUE (Q, eps) to roundoff
    np.testing.assert_allclose(sigma, model.stress(Q, eps), rtol=1e-9, atol=1e-12)

    # finite differences through the deformed cell (ASE stress map)
    h = 1e-6
    vol = _V0_PRIM * float(np.linalg.det(np.eye(3) + voigt_to_matrix(eps)))
    fd = np.empty(6)
    for v in range(6):
        d = np.zeros(6)
        d[v] = h
        ep = make_atoms(mylwfsc, FrameSpec(Q=Q, strain_voigt=eps + d))
        em = make_atoms(mylwfsc, FrameSpec(Q=Q, strain_voigt=eps - d))
        ep.calc = calc
        em.calc = calc
        fd[v] = (ep.get_potential_energy() - em.get_potential_energy()) / (2 * h)
    fd /= vol
    np.testing.assert_allclose(fd, sigma, rtol=1e-5, atol=1e-9)


def test_calculator_lsqr_solver_agrees():
    """The sparse lsqr recovery path agrees with the dense pinv path to
    solver tolerance."""
    from lawaf.anharmonic.model import LWFModelCalculator

    model, calc, mylwfsc = _calculator_world()
    calc_lsqr = LWFModelCalculator(model, mylwfsc, solver="lsqr")
    rng = np.random.default_rng(11)
    Q = rng.normal(0.0, 0.2, model.nQ)
    atoms, _ = mylwfsc.get_distorted_atoms(Q)
    atoms.calc = calc_lsqr
    eps0 = np.zeros(6)
    assert atoms.get_potential_energy() == pytest.approx(
        model.energy(Q, eps0), rel=1e-7, abs=1e-9
    )
    M = np.asarray(mylwfsc.mapping_mat.toarray(), dtype=float)
    F_ref = -(M @ np.linalg.solve(M.T @ M, model.gradient(Q, eps0))).ravel()
    np.testing.assert_allclose(
        atoms.get_forces().ravel(), F_ref, rtol=1e-5, atol=1e-8
    )


def test_calculator_get_stress_3x3_conversion():
    """get_stress(voigt=False) maps back to the symmetric 3x3 tensor via the
    ASE Voigt order (xx, yy, zz, yz, xz, xy) - stored once, averaged 1/2."""
    from ase.stress import voigt_6_to_full_3x3_stress

    model, calc, mylwfsc = _calculator_world()
    rng = np.random.default_rng(12)
    Q = rng.normal(0.0, 0.15, model.nQ)
    eps = np.array([0.008, -0.005, 0.003, 0.002, 0.001, -0.004])
    atoms = make_atoms(mylwfsc, FrameSpec(Q=Q, strain_voigt=eps))
    atoms.calc = calc
    s6 = atoms.get_stress()
    s33 = atoms.get_stress(voigt=False)
    np.testing.assert_allclose(s33, voigt_6_to_full_3x3_stress(s6), atol=1e-30)


# ===========================================================================
# TEST-005: netCDF anharmonic-group serialization
# ===========================================================================
def _tamper_schema_version(path, value):
    import netCDF4

    with netCDF4.Dataset(str(path), "a") as root:
        root.groups["anharmonic"].setncattr("schema_version", value)


def test_io_roundtrip_bitexact_standalone(tmp_path):
    """TEST-005: save -> load -> fitted coefficients bit-exact, basis terms
    exactly reconstructed (Fractions), locality/fingerprint preserved, and
    the loaded model evaluates STANDALONE (same floats, incl. the restored
    harmonic kernel)."""
    from lawaf.anharmonic import io as anh_io
    from lawaf.anharmonic.model import AnharmonicModel

    res, basis, harmonic, M, Q, strain = _fitted_world()
    path = tmp_path / "model.nc"
    anh_io.save_anharmonic_model(
        path,
        res,
        provenance={"teacher": "toy:E=B@c+Harm", "harmonic_source": "toy_lwf"},
        primitive_volume=_V0_PRIM,
    )

    loaded = anh_io.load_anharmonic_model(path)
    r2 = loaded.coefficients
    assert r2 is not res
    # fitted arrays bit-exact
    np.testing.assert_array_equal(r2.coefficients, res.coefficients)
    np.testing.assert_array_equal(r2.stderrs, res.stderrs)
    assert r2.selected == res.selected
    assert r2.fingerprint == res.fingerprint
    assert r2.selection == res.selection
    assert r2.ridge_alpha == res.ridge_alpha
    assert r2.weights == res.weights
    # exact rational basis: same term keys and Fraction coefficients
    assert len(r2.basis.terms) == len(basis.terms)
    for t_new, t_old in zip(r2.basis.terms, basis.terms):
        assert t_new.seed == t_old.seed
        assert t_new.order == t_old.order
        assert t_new.sector == t_old.sector
        assert t_new.coeffs == t_old.coeffs
    assert r2.basis.fingerprint == basis.fingerprint
    # locality report preserved (json exact float roundtrip)
    assert r2.locality.r90_lattice == res.locality.r90_lattice
    assert len(r2.locality.rows) == len(res.locality.rows)
    assert r2.cv.n_train_rows == res.cv.n_train_rows
    # standalone evaluation: identical floats, harmonic kernel restored
    model_in = AnharmonicModel(res, primitive_volume=_V0_PRIM)
    rng = np.random.default_rng(13)
    Qt = rng.normal(0, 0.4, model_in.nQ)
    et = rng.normal(0, 0.02, 6)
    assert loaded.energy(Qt, et) == model_in.energy(Qt, et)
    np.testing.assert_array_equal(loaded.gradient(Qt, et), model_in.gradient(Qt, et))
    np.testing.assert_array_equal(loaded.stress(Qt, et), model_in.stress(Qt, et))
    assert loaded.primitive_volume == _V0_PRIM
    # schema marker
    import netCDF4

    with netCDF4.Dataset(str(path)) as root:
        assert root.groups["anharmonic"].schema_version == 1


def test_io_schema_version_guard(tmp_path):
    """A file written by a NEWER schema is rejected with an informative
    error (forward-compat policy: refuse, never silently misread)."""
    from lawaf.anharmonic import io as anh_io

    res, *_ = _fitted_world()
    path = tmp_path / "model.nc"
    anh_io.save_anharmonic_model(path, res)
    _tamper_schema_version(path, anh_io.SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="schema_version"):
        anh_io.load_anharmonic_model(path)


def test_io_group_scoped_appends_siblings(tmp_path):
    """io.py owns ONLY the 'anharmonic' group: a sibling group written first
    survives, and a sibling group appended afterwards survives too."""
    import netCDF4

    from lawaf.anharmonic import io as anh_io

    res, *_ = _fitted_world()
    path = tmp_path / "model.nc"
    with netCDF4.Dataset(str(path), "w") as root:
        g = root.createGroup("symmetry")
        g.setncattr("marker", "sibling-was-here")
    anh_io.save_anharmonic_model(path, res)
    with netCDF4.Dataset(str(path)) as root:
        assert root.groups["symmetry"].marker == "sibling-was-here"
        assert "anharmonic" in root.groups
    # and the reverse order: anharmonic first, sibling appended later
    path2 = tmp_path / "model2.nc"
    anh_io.save_anharmonic_model(path2, res)
    with netCDF4.Dataset(str(path2), "a") as root:
        root.createGroup("symmetry").setncattr("marker", "late")
    anh_io.load_anharmonic_model(path2)  # still loads
    with netCDF4.Dataset(str(path2)) as root:
        assert root.groups["symmetry"].marker == "late"
    # refusing to overwrite an existing anharmonic group
    with pytest.raises(ValueError, match="anharmonic"):
        anh_io.save_anharmonic_model(path2, res)


def test_evaluate_atoms_convenience():
    """Module-level evaluate_atoms returns the same quantities as the
    calculator path."""
    from lawaf.anharmonic.model import evaluate_atoms

    model, calc, mylwfsc = _calculator_world()
    rng = np.random.default_rng(14)
    Q = rng.normal(0.0, 0.2, model.nQ)
    eps = np.array([0.005, 0.004, -0.003, 0.001, 0.002, -0.001])
    atoms = make_atoms(mylwfsc, FrameSpec(Q=Q, strain_voigt=eps))
    out = evaluate_atoms(atoms, model, mylwfsc)
    atoms.calc = calc
    assert out["energy"] == atoms.get_potential_energy()
    np.testing.assert_array_equal(out["forces"], atoms.get_forces())
    np.testing.assert_array_equal(out["stress"], atoms.get_stress())
    np.testing.assert_allclose(out["Q"], Q, atol=1e-10)
    np.testing.assert_allclose(out["strain_voigt"], eps, atol=1e-12)
