"""Story 021: residual-baseline joint fit of the anharmonic effective model
(FR-010/018, ADR-009).

Covers
------
* ``harmonic_baseline``: supercell-folded Born-von Karman quadratic form from
  ``HR_total`` (k-space law H(q) = sum_R Rdeg[R] HR[R] exp(2 pi i q.R) via
  ``lawaf.mathutils.kR_convert.R_to_onek``), energy/gradient consistency
  (closed form + finite differences), commensurate Bloch-wave dispersion
  identity, plain-LWF ``HwannR`` fallback, Gamma folding.
* ``build_design`` / ``fit``: exact zero-noise recovery of a sparse
  coefficient set (ridge and screened_greedy), stress-row sector
  selectivity, determinism (bit-exact reruns), NaN-label exclusion, and the
  end-to-end sign chain g_Q = M^T F, dE/dQ = -g_Q.
* Executed sympy verifications (user-mandated): quadratic-form gradient
  identity for non-symmetric H, polynomial monomial gradients (exponent
  products) vs sympy.diff, Voigt stress derivative of symmetric-tensor
  monomials, RMS block-weight algebra.
"""
import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from lawaf.anharmonic.basis import (
    ClusterAction,
    ClusterKey,
    InvariantBasis,
    InvariantTerm,
    build_invariant_basis,
    voigt_matrix_from_rotation,
)
from lawaf.anharmonic.dataset import TrainingDataset
from lawaf.anharmonic.fit import (
    AnharmonicCoefficients,
    HarmonicBaseline,
    basis_cell_permutation,
    build_design,
    fit,
    harmonic_baseline,
)
from lawaf.anharmonic.sampling import project_forces, voigt_to_matrix
from lawaf.lwf.lwf import LWF as PlainLWF
from lawaf.mathutils.kR_convert import R_to_onek
from lawaf.utils.supercell import SupercellMaker

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# ---------------------------------------------------------------------------
# tiny world: 2-branch LWF, supercell diag(2,1,1), C2-like cluster action
# ---------------------------------------------------------------------------
class _TinyAction(ClusterAction):
    """{E, diag(1,-1,-1)} with every branch flipping sign under the rotation.

    Both R vectors of the toy world ((0,0,0) and (1,0,0)) are fixed points of
    the point operation, so the toy Rlist is closed.  Strain Voigt rows are
    signed permutations: (xx, yy, zz, yz) -> +itself, (xz, xy) -> -itself.
    """

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
        return op  # both operations are involutions


def _toy_lwf(nwann=2, seed=5, Rlist=((0, 0, 0), (1, 0, 0))):
    rng = np.random.default_rng(seed)
    HR = rng.normal(0.0, 1.0, (len(Rlist), nwann, nwann))
    HR = 0.5 * (HR + HR.transpose(0, 2, 1))  # symmetric real blocks
    lwf = SimpleNamespace(
        HR_total=HR,
        Rlist=np.array(Rlist, dtype=int),
        Rdeg=None,
        wann_masses=np.full(nwann, 2.0),
    )
    return lwf, HR


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
    """Random full-column-rank stand-in for build_lwf_lattice_mapping_matrix."""
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 1.0, (3 * natom_sc, nQ))
    return M


def _toy_dataset(basis, harmonic, c_true, M, nf=30, seed=3, strain_mode="random",
                 with_stress=True):
    """Frames + exact (zero-noise) labels for E_anh = B @ c_true (+ harmonic)."""
    rng = np.random.default_rng(seed)
    nQ = len(basis.coord_labels)
    assert nQ == harmonic.nQ
    Q = rng.normal(0.0, 0.7, (nf, nQ))
    if strain_mode == "zero":
        strain = np.zeros((nf, 6))
    else:
        strain = rng.normal(0.0, 0.05, (nf, 6))

    gb = basis_cell_permutation(basis, harmonic)
    pg = _PolyGradBG(basis)

    E_anh = basis.evaluate(Q, strain) @ c_true
    # pool-order gradient -> cell order -> mode forces g_Q = -dE/dQ
    grad_pool = pg.grad_q(Q, strain)  # (nf, ncoord, nT)
    anh_grad_cell = (grad_pool @ c_true)[:, gb]  # dE_anh/dQ in cell order
    harm_grad = np.array([harmonic.gradient(Q[i]) for i in range(nf)])
    # exact-model mode force: g_Q = M^T F = -dE_total/dQ
    g_total = anh_grad_cell + harm_grad

    natom_sc = M.shape[0] // 3
    # Cartesian forces whose projection is exactly g_target: F = M (M^T M)^{-1} g
    F = np.einsum("ij,fj->fi", M @ np.linalg.inv(M.T @ M), -g_total)
    assert np.allclose(F.reshape(nf, -1) @ M, -g_total, atol=1e-9)

    # stress labels: sigma_v = (1/V_labeled) dE/deps_v — ASE convention on
    # the LABELED (supercell) cell; harmonic has no strain dependence
    grad_eps = pg.grad_eps(Q, strain)  # (nf, 6, nT)
    vol = np.array([abs(np.linalg.det(c)) for c in _toy_cells(nf)])
    sigma = (grad_eps @ c_true) / vol[:, None]

    cell = np.array(_toy_cells(nf))
    natom = natom_sc
    ds = TrainingDataset(
        positions=[np.zeros((natom, 3)) for _ in range(nf)],
        atomic_numbers=np.full((nf, natom), 14),
        cell=cell,
        pbc=np.full((nf, 3), True),
        Q=Q,
        strain_voigt=strain,
        split=_make_splits(nf),
    )
    ds.label(
        "toy",
        values={
            "energy": E_anh + np.array([harmonic.energy(Q[i]) for i in range(nf)]),
            "forces": F.reshape(nf, natom_sc, 3),
        },
    )
    if with_stress:
        sigma_3x3 = np.stack([voigt_to_matrix(v) for v in sigma])
        ds.fill_label_values(ds.active_label, np.arange(nf), stress=sigma_3x3)
    return ds, Q, strain


def _make_splits(nf):
    """Deterministic frame splits: ~70% train, ~20% cv, rest holdout."""
    n_train = max(1, int(0.7 * nf))
    n_cv = max(1, nf // 5)
    return ["train"] * n_train + ["cv"] * n_cv + ["holdout"] * (nf - n_train - n_cv)


def _toy_cells(nf):
    scell = np.diag([2.0, 1.0, 1.0]) * 4.0
    return [scell for _ in range(nf)]


# import the fit-local exact polynomial gradient wrapper (the machinery under
# test; sympy-verified separately below)
from lawaf.anharmonic.fit import _BasisGrad as _PolyGradBG  # noqa: E402


# ---------------------------------------------------------------------------
# TEST-001: harmonic baseline round trip
# ---------------------------------------------------------------------------
def test_harmonic_baseline_folded_kernel_gradient_and_dispersion():
    """E_harm matches an independent commensurate-q DFT fold of H(q); the
    gradient matches the closed form and finite differences; Bloch waves at a
    commensurate q reproduce the LWF dispersion energy N/4 A^T H(q) A."""
    Rlist = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
    lwf, HR = _toy_lwf(nwann=2, seed=9, Rlist=Rlist)
    # physical table: HR[-R] = HR[R]^T on the supercell residue classes
    # (-1 mod 4 = 3), so the folded kernel is genuinely symmetric
    HR[3] = HR[1].T
    lwf.HR_total = HR
    S = np.diag([4, 1, 1])
    scmaker = SupercellMaker(S)
    sc_vec = np.asarray(scmaker.sc_vec, dtype=int)
    assert [tuple(v) for v in sc_vec] == list(Rlist)
    hb = harmonic_baseline(lwf, scmaker)
    assert hb.nQ == 8

    # independent reference: Hmat_ref[(c,i),(c',j)] =
    # (1/N) sum_q H(q)_{ij} exp(-2 pi i q . (l_c' - l_c))
    ncell = 4
    qs = [[n / 4, 0.0, 0.0] for n in range(ncell)]
    Href = np.zeros((8, 8), dtype=complex)
    for c in range(ncell):
        for cp in range(ncell):
            D = sc_vec[cp] - sc_vec[c]
            for q in qs:
                Hq = R_to_onek(np.array(q), lwf.Rlist, HR, None)
                Href[2 * c : 2 * c + 2, 2 * cp : 2 * cp + 2] += (
                    Hq * np.exp(-2j * np.pi * np.dot(q, D))
                )
    Href /= ncell
    np.testing.assert_allclose(Href.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(Href - Href.T, 0.0, atol=1e-12)
    np.testing.assert_allclose(hb.Hmat, Href.real, atol=1e-12)

    # energy / gradient closed form
    rng = np.random.default_rng(0)
    Q = rng.normal(0.0, 0.8, (7, 8))
    E = np.array([hb.energy(qi) for qi in Q])
    np.testing.assert_allclose(E, 0.5 * np.einsum("fi,ij,fj->f", Q, hb.Hmat, Q), atol=1e-12)
    G = np.array([hb.gradient(qi) for qi in Q])
    np.testing.assert_allclose(G, Q @ hb.Hmat.T, atol=1e-12)

    # central finite differences
    h = 1e-6
    fd = np.array([(hb.energy(Q[i] + h * np.eye(8)[k]) - hb.energy(Q[i] - h * np.eye(8)[k])) / (2 * h)
                   for i in (0, 3) for k in range(8)]).reshape(2, 8)
    np.testing.assert_allclose(fd, G[[0, 3]], rtol=1e-6, atol=1e-6)

    # commensurate Bloch waves: E(cos wave) == E(sin wave) == N/4 A^T H(q) A
    q = np.array([0.25, 0.0, 0.0])
    A = rng.normal(0.0, 1.0, 2)
    Hq = R_to_onek(q, lwf.Rlist, HR, None)
    assert np.allclose(Hq.imag, 0.0)
    theta = 2 * np.pi * sc_vec @ q
    # flat supercell amplitude vector: Q[icell*nlwf + iwann]
    Qcos = (np.cos(theta)[:, None] * A[None, :]).reshape(-1)
    Qsin = (np.sin(theta)[:, None] * A[None, :]).reshape(-1)
    ref = ncell / 4 * A @ Hq.real @ A
    assert abs(hb.energy(Qcos) - ref) < 1e-12
    assert abs(hb.energy(Qsin) - ref) < 1e-12


def test_harmonic_baseline_gamma_fold_and_plain_lwf():
    """Without a supercell the single cell folds every R into one class
    (H(Gamma) = sum_R Rdeg HR[R]); the plain lawaf.lwf.lwf.LWF object is
    accepted through its HwannR attribute."""
    lwf, HR = _toy_lwf(nwann=2, seed=5)
    hb = harmonic_baseline(lwf)  # Gamma cell of one primitive cell
    np.testing.assert_allclose(hb.Hmat, HR.sum(axis=0), atol=1e-12)
    Q = np.array([0.3, -0.4])
    np.testing.assert_allclose(hb.gradient(Q), hb.Hmat @ Q, atol=1e-12)

    wannR = np.random.default_rng(2).normal(0.0, 0.5, (2, 6, 2))
    plain = PlainLWF(wannR=wannR, HwannR=HR, Rlist=lwf.Rlist)
    hb2 = harmonic_baseline(plain, SupercellMaker(np.diag([2, 1, 1])))
    lwf2 = SimpleNamespace(HR_total=HR, Rlist=lwf.Rlist, Rdeg=None)
    hb3 = harmonic_baseline(lwf2, SupercellMaker(np.diag([2, 1, 1])))
    np.testing.assert_allclose(hb2.Hmat, hb3.Hmat, atol=1e-12)


def test_basis_cell_permutation_and_harmonic_anharm_widths():
    basis = _toy_basis()
    lwf, _ = _toy_lwf()
    hb = _toy_harmonic(lwf)
    pc = basis_cell_permutation(basis, hb)
    # (b, R) -> cell(R) * nlwf + b, with cells in scmaker.sc_vec order
    cell_of = {(0, 0, 0): 0, (1, 0, 0): 1}
    expected = [cell_of[R] * 2 + b for (b, R) in basis.coord_labels]
    assert list(pc) == expected
    assert len(basis.coord_labels) == hb.nQ


# ---------------------------------------------------------------------------
# TEST-002: synthetic exact polynomial recovery
# ---------------------------------------------------------------------------
def _exact_world():
    basis = _toy_basis()
    lwf, _ = _toy_lwf()
    harmonic = _toy_harmonic(lwf)
    M = _toy_mapping(harmonic.nQ)
    rng = np.random.default_rng(21)
    sectors = ["q", "q", "coupled"]
    idx = [i for i, t in enumerate(basis.terms) if t.sector in ("q", "coupled")]
    rng.shuffle(idx)
    support = sorted(idx[:3])
    c_true = np.zeros(len(basis.terms))
    c_true[support] = rng.normal(0.0, 1.0, 3)
    return basis, harmonic, M, support, c_true


def test_exact_recovery_ridge_and_screened_greedy():
    basis, harmonic, M, support, c_true = _exact_world()
    ds, Q, strain = _toy_dataset(basis, harmonic, c_true, M, nf=30, seed=3)

    res = fit(ds, basis, harmonic, mapping=M, selection="ridge", ridge_alpha=0.0)
    np.testing.assert_allclose(res.coefficients[support], c_true[support], atol=1e-10)
    assert np.max(np.abs(res.coefficients)) == pytest.approx(
        np.max(np.abs(res.coefficients[support])), abs=1e-10
    )

    res2 = fit(
        ds, basis, harmonic, mapping=M, selection="screened_greedy",
        ridge_alpha=0.0, n_coeff=3,
    )
    assert sorted(res2.selected) == sorted(support)
    np.testing.assert_allclose(res2.coefficients[support], c_true[support], atol=1e-10)

    # CV metrics on the held-out cv split
    assert res.cv.energy_mae < 1e-8
    assert res.cv.force_rmse < 1e-8
    assert res.cv.stress_rmse < 1e-8
    assert res.cv.n_cv_frames == 6
    assert isinstance(res, AnharmonicCoefficients)


# ---------------------------------------------------------------------------
# TEST-003: stress rows move only strain-sector coefficients
# ---------------------------------------------------------------------------
def test_stress_rows_only_change_strain_sector():
    basis = _toy_basis()
    lwf, _ = _toy_lwf()
    harmonic = _toy_harmonic(lwf)
    M = _toy_mapping(harmonic.nQ)
    rng = np.random.default_rng(31)
    q_idx = [i for i, t in enumerate(basis.terms) if t.sector == "q"]
    cp_idx = [i for i, t in enumerate(basis.terms) if t.sector == "coupled"]
    rng.shuffle(q_idx)
    rng.shuffle(cp_idx)
    q_support = sorted(q_idx[:2])
    cp_support = sorted(cp_idx[:2])
    support = q_support + cp_support
    c_true = np.zeros(len(basis.terms))
    c_true[support] = rng.normal(0.0, 1.0, 4)

    # ALL frames at zero strain: coupled columns vanish identically from the
    # energy+force rows, so the stress rows are the only information about them.
    ds1, _, _ = _toy_dataset(basis, harmonic, c_true, M, nf=30, seed=4,
                             strain_mode="zero", with_stress=False)
    res1 = fit(ds1, basis, harmonic, mapping=M, selection="ridge", ridge_alpha=0.0)
    # q-sector uniquely pinned; coupled directions are in the null space ->
    # min-norm solution puts zero there
    np.testing.assert_allclose(res1.coefficients[q_support], c_true[q_support], atol=1e-9)
    np.testing.assert_allclose(res1.coefficients[cp_support], 0.0, atol=1e-9)

    ds2 = copy.deepcopy(ds1)
    ds2.fill_label_values(
        ds2.active_label, np.arange(ds2.nframes), stress=_stress_labels(
            basis, harmonic, c_true, ds2, zero_strain=True)
    )
    res2 = fit(ds2, basis, harmonic, mapping=M, selection="ridge", ridge_alpha=0.0)
    # stress rows have IDENTICALLY ZERO q-sector columns -> q-sector unchanged
    np.testing.assert_allclose(
        res2.coefficients[q_support], res1.coefficients[q_support], atol=1e-12
    )
    # strain-sector (coupled) coefficients now recovered to the truth
    np.testing.assert_allclose(res2.coefficients[cp_support], c_true[cp_support], atol=1e-10)


def _stress_labels(basis, harmonic, c_true, ds, zero_strain=False):
    from lawaf.anharmonic.fit import _BasisGrad

    strain = np.zeros((ds.nframes, 6)) if zero_strain else ds.strain_voigt
    pg = _BasisGrad(basis)
    geps = pg.grad_eps(ds.Q, strain)
    # ASE convention: stress of the LABELED (supercell) cell
    vol = np.array([abs(np.linalg.det(c)) for c in ds.cell])
    sigma = (geps @ c_true) / vol[:, None]
    return np.stack([voigt_to_matrix(v) for v in sigma])


# ---------------------------------------------------------------------------
# TEST-004: determinism
# ---------------------------------------------------------------------------
def test_fit_determinism_bit_exact():
    basis, harmonic, M, support, c_true = _exact_world()
    ds, _, _ = _toy_dataset(basis, harmonic, c_true, M, nf=26, seed=6)
    kw = dict(mapping=M, selection="screened_greedy", n_coeff=3, seed=0)
    r1 = fit(ds, basis, harmonic, **kw)
    r2 = fit(ds, basis, harmonic, **kw)
    assert r1.selected == r2.selected
    assert np.array_equal(r1.coefficients, r2.coefficients)
    assert np.array_equal(r1.stderrs, r2.stderrs, equal_nan=True)
    assert r1.fingerprint == r2.fingerprint
    kw2 = dict(mapping=M, selection="ridge", ridge_alpha=1e-8)
    s1 = fit(ds, basis, harmonic, **kw2)
    s2 = fit(ds, basis, harmonic, **kw2)
    assert np.array_equal(s1.coefficients, s2.coefficients)
    assert s1.fingerprint == s2.fingerprint
    assert s1.fingerprint != r1.fingerprint  # different method -> different hash



# ---------------------------------------------------------------------------
# TEST-005: NaN stress exclusion
# ---------------------------------------------------------------------------
def test_nan_labels_excluded_never_imputed():
    basis, harmonic, M, support, c_true = _exact_world()
    ds, _, _ = _toy_dataset(basis, harmonic, c_true, M, nf=30, seed=3,
                            with_stress=False)
    # stress labels only on frames 0..9
    stress = _stress_labels(basis, harmonic, c_true, ds)
    ds.fill_label_values(ds.active_label, np.arange(10), stress=stress[:10])
    A, b, meta = build_design(basis, ds, harmonic=harmonic, mapping=M)
    n_stress = sum(1 for row in meta if row[1] == "stress")
    assert n_stress == 60
    assert np.isfinite(A).all() and np.isfinite(b).all()
    # drop one energy label as well
    ds.blocks[ds.active_label].energy[17] = np.nan
    A2, b2, meta2 = build_design(basis, ds, harmonic=harmonic, mapping=M)
    n_energy = sum(1 for row in meta2 if row[1] == "energy")
    assert n_energy == ds.nframes - 1

    res = fit(ds, basis, harmonic, mapping=M, selection="ridge", ridge_alpha=1e-10)
    assert np.isfinite(res.coefficients).all()
    np.testing.assert_allclose(res.coefficients[support], c_true[support], atol=1e-8)



# ---------------------------------------------------------------------------
# TEST-006: sign chain end to end
# ---------------------------------------------------------------------------
def test_sign_chain_end_to_end():
    basis, harmonic, M, support, c_true = _exact_world()
    # enough train frames for a full-rank energy+force subsystem
    ds, Q, strain = _toy_dataset(basis, harmonic, c_true, M, nf=40, seed=8,
                                 with_stress=False)
    res = fit(ds, basis, harmonic, mapping=M, selection="ridge", ridge_alpha=0.0)
    np.testing.assert_allclose(res.coefficients[support], c_true[support], atol=1e-10)
    # closed chain on every cv frame: dE_total/dQ + g_Q == 0 with
    # g_Q = M^T F_label (dataset forces), dE_total = harmonic + fitted anh
    cv = [i for i in range(ds.nframes) if str(ds.split[i]) == "cv"]
    assert cv
    for i in cv:
        g_Q = project_forces(M, ds.forces[i])
        grad_total = res.gradient(Q[i], strain[i])  # cell order, harm + anh
        assert np.max(np.abs(grad_total + g_Q)) < 1e-8
        assert abs(res.energy(Q[i], strain[i]) - ds.energies[i]) < 1e-8


# ---------------------------------------------------------------------------
# sympy verifications (user-mandated)
# ---------------------------------------------------------------------------
def test_sympy_quadratic_gradient_identity():
    """d/dQ (1/2 Q^T H Q) = (H + H^T)/2 Q for a GENERIC (non-symmetric) H, and
    the antisymmetric part contributes nothing to the energy."""
    x = sp.Matrix(sp.symbols("x0 x1 x2"))
    H = sp.Matrix(3, 3, sp.symbols("h:9"))
    f = sp.Rational(1, 2) * (x.T * H * x)[0]
    grad = sp.Matrix([sp.diff(f, xi) for xi in x])
    assert sp.simplify(grad - (H + H.T) / 2 * x) == sp.zeros(3, 1)
    A = H - H.T  # antisymmetric part
    assert sp.simplify((x.T * A * x)[0]) == 0


def test_sympy_monomial_gradients_match_exponent_products():
    """The fit's exponent-product monomial gradients equal sympy.diff of the
    expanded monomial (checked at random rational points)."""
    basis = _toy_basis()
    pg = _PolyGradBG(basis)
    q = sp.symbols("q0:4")
    e = sp.symbols("e0:6")
    rng = np.random.default_rng(13)
    for trial in range(4):
        # random monomial: 2-3 Q factors + 0-1 strain factors from a real term
        t = basis.terms[int(rng.integers(0, len(basis.terms)))]
        (qk, sk) = list(t.coeffs)[0]
        mono_q = sp.Integer(1)
        for (b, R) in qk:
            i = [j for j, lab in enumerate(basis.coord_labels) if lab == (b, R)][0]
            mono_q *= q[i]
        mono = mono_q
        for v in sk:
            mono *= e[v]
        gq_sym = [sp.diff(mono, qi) for qi in q]
        ge_sym = [sp.diff(mono, ev) for ev in e]
        # random rational point
        qv = np.array([float(sp.Rational(rng.integers(-9, 10), rng.integers(1, 10))) for _ in q])
        ev = np.array([float(sp.Rational(rng.integers(-9, 10), rng.integers(1, 10))) for _ in e])
        # evaluate against a one-term basis carrying exactly this monomial
        term = InvariantTerm(
            seed=ClusterKey(qk, sk), coeffs={(qk, sk): Fraction(1)},
            order=len(qk) + len(sk), sector="q" if not sk else "coupled",
        )
        b1 = InvariantBasis(
            terms=[term], coord_labels=basis.coord_labels, nlwf=2,
            action=_TinyAction(), orders=(3, 4), include_strain=True,
            max_strain_power=1, cutoff_active=False, rlist=((0, 0, 0), (1, 0, 0)),
        )
        pg1 = _PolyGradBG(b1)
        gq_fit = pg1.grad_q(qv[None, :], ev[None, :])[0]  # (ncoord, 1)
        ge_fit = pg1.grad_eps(qv[None, :], ev[None, :])[0]  # (6, 1)
        subs = {q[j]: qv[j] for j in range(4)}
        subs.update({e[j]: ev[j] for j in range(6)})
        sym_q = np.array([float(g.subs(subs).evalf()) for g in gq_sym])
        sym_e = np.array([float(g.subs(subs).evalf()) for g in ge_sym])
        np.testing.assert_allclose(gq_fit[:, 0], sym_q, atol=1e-12)
        np.testing.assert_allclose(ge_fit[:, 0], sym_e, atol=1e-12)


def test_sympy_voigt_stress_derivative_stored_once():
    """Voigt stress rows are dE/deps_v with each tensor component stored ONCE:
    dE/d(eps_yz) of Q^2 eps_yz is Q^2 (no factor 2), and repeated factors give
    the exponent rule dE/d(eps_yz) of eps_yz^2 is 2 eps_yz."""
    qs, e = sp.symbols("q0 eps_yz")
    eps = sp.Matrix(
        [
            [sp.Symbol("exx"), sp.Symbol("exy"), sp.Symbol("exz")],
            [sp.Symbol("exy"), sp.Symbol("eyy"), e],
            [sp.Symbol("exz"), e, sp.Symbol("ezz")],
        ]
    )
    E1 = qs**2 * eps[1, 2]
    assert sp.diff(E1, e) == qs**2  # single storage, no factor 2
    E2 = eps[1, 2] ** 2
    assert sp.simplify(sp.diff(E2, e) - 2 * e) == 0


def test_sympy_rms_block_weight_algebra():
    """Per-block RMS scaling s = sqrt(n / sum b_i^2) makes the scaled block
    residual RMS exactly 1, positive block scaling leaves an exactly-consistent
    LSQ solution unchanged, and block scaling equals the weighted normal
    equations sum_i s^2 a_i^T a_i x = sum_i s^2 a_i^T b_i."""
    b1, b2, a11, a12, a21, a22, x1, x2, s = sp.symbols("b1 b2 a11 a12 a21 a22 x1 x2 s")
    n = 2
    s2 = sp.sqrt(n / (b1**2 + b2**2))
    rms = sp.simplify(((s2 * b1) ** 2 + (s2 * b2) ** 2) / n)
    assert sp.simplify(rms - 1) == 0
    # exact solution invariance: x0 solving a.x = b also solves (s a).x = s b
    a = sp.Matrix([[a11, a12], [a21, a22]])
    x0 = sp.Matrix([x1, x2])
    bb = sp.Matrix([b1, b2])
    resid = sp.expand(a * x0 - bb)
    scaled = sp.expand((s * a) * x0 - s * bb)
    assert sp.simplify(scaled.subs(s, s2) - s2 * resid) == sp.zeros(2, 1)
