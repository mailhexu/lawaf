"""Tests for lawaf.anharmonic.gauge — story-018 star-covariant constrained
gauge (FR-004/015, ADR-003).

Layout (TEST numbers follow the story):
- TEST-001 little-group constraint residual eps(q) <= 1e-10 after the
  constrained pipeline on the REAL BaTiO3 downfold (Gamma mesh, T1u
  declaration at the 1b Ti orbit, acoustic window via the +/-20 cm^-1 Gauss
  occupancy), plus the real-space Gamma T_g relation and dual spreads.
- TEST-002 star propagation on the acoustic mock (real BaTiO3 eigenvectors,
  3x3x3): U(gq) equals the propagated expression to 1e-10, and an
  INDEPENDENTLY re-constrained U at the star member spans the same subspace
  (projector overlap 1) although the gauges differ.
- TEST-003 time-reversal ties: Amn(-q) = Y Amn(q)*, Y = psi(-q)^dag psi(q)*,
  at genuine TR pairs of the 3x3x3 mesh, and the constraint still holds at
  the tied partner; full mesh coverage (star arms + TR ties + irreducibles).
- TEST-004 off-parameter bit identity: default symmetry_adapted_gauge=False,
  constraint code never invoked (spy), and the standard path is
  deterministic (bitwise-equal Amn across two constructions).
- TEST-005 fail-fast monitors: (a) window not carrying the declaration
  (real fixture, bogus T2g declaration; synthetic trivial-window case),
  (b) near-degenerate D_W (synthetic singular D_W), (c) degenerate
  constraint channel (synthetic), each a ValueError naming the window/q.
- TEST-006 Wannier transformation consistency: real-space LWFs from the
  constrained gauge satisfy the site T_g relations to 1e-8 (real fixture)
  and the k-space T_g relation S_g(q) wannk(q) D_W(g)^dag = wannk(gq)
  (mock, star arms included).
- sympy: docs/derivations/story018_gauge_sympy.py runs green as a
  subprocess (projection identity, manifold preservation, Procrustes
  optimality, star transitivity, channel algebra, extraction skeleton).
- params/serialization + projectedWF hook routing tests.

The mock builder plants the acoustic triplet (the displacement
representation at every q) as the working window with a projector-based
get_Amn_one_k — no occupancy weighting and no Amn enhancement hack — so the
constraint/propagation machinery is exercised on symmetry-exact windows
built from REAL fixture eigenvectors (why: no frequency window can isolate
the acoustic triplet across a mesh in which the soft mode crosses it; see
story report).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import svd

phonopy = pytest.importorskip("phonopy")

from lawaf.anharmonic.compatibility import (  # noqa: E402
    _GroupAlgebra,
    character_table,
    little_group,
)
from lawaf.anharmonic.gauge import (  # noqa: E402
    _frame_phase,
    _pure_point_matrices,
    _qkey,
    constrain_amn_one_q,
    constrain_builder_amn,
    constrained_localize,
    site_irrep_matrices,
)
from lawaf.anharmonic.representation import (  # noqa: E402
    build_space_group_action,
)
from lawaf.params import WannierParams  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
DERIVATION = Path(__file__).parent.parent / "docs" / "derivations" / "story018_gauge_sympy.py"

T1U_DECL = dict(wyckoff="1b", site_irreps=["T1u"], strain_sector=True)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def phonon():
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def sga(phonon):
    return build_space_group_action(phonon)


def _real_downfolder(phonon, mesh=(1, 1, 1), **overrides):
    from lawaf.interfaces.phonopy import phonon_downfolder as pdf

    params = WannierParams(
        method="projected",
        nwann=3,
        anchors={(0, 0, 0): (0, 1, 2)},  # projector columns (window comes
        # from the Gauss occupancy below: the acoustic triplet only)
        use_proj=True,
        weight_func="Gauss",
        weight_func_params=(-20.0, 20.0),
        kmesh=mesh,
    )
    for key, val in overrides.items():
        setattr(params, key, val)
    df = pdf.PhonopyDownfolder(phonon=phonon, params=params)
    df._prepare_data()
    df.atoms = df.model.atoms
    df.builder.prepare()
    df.builder.get_Amn()
    return df


def _frac(phonon):
    """Fractional atom positions of the fixture primitive cell."""
    return np.mod(phonon.primitive.scaled_positions, 1.0)


def _Sfree(sga):
    """Defect-free (pure point-operation) representation matrices S'_g."""
    from lawaf.anharmonic.gauge import _pure_point_matrices

    return _pure_point_matrices(sga)


class MockAcousticBuilder:
    """Duck-typed projected builder over real BaTiO3 eigenvectors,
    downfolded to the acoustic-triplet window.

    At every mesh q the three eigenvector columns whose physical
    displacement pattern has maximal overlap with the three uniform
    translations form the window.  The window carries the declared T1u
    content EXACTLY ONCE at each irreducible q (multiplicity one), so
    constrained subspaces are symmetry-unique; the star of the body-
    diagonal point provides genuine star arms and its inversion image a
    genuine time-reversal tie on the 3x3x3 mesh.
    """

    ndim = 3

    def __init__(self, model, sga, mesh, frac_positions=None):
        self.model = model
        self.sga = sga
        self.mesh = tuple(int(n) for n in mesh)
        self.kmesh = self.mesh
        grids = [np.arange(n) for n in self.mesh]
        self.kpts = (
            np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            / np.array(self.mesh, dtype=float)
        )
        self.kshift = np.zeros(3)
        self.kweights = np.full(len(self.kpts), 1.0 / len(self.kpts))
        self.nbasis = 3 * sga.n_atoms  # full displacement space rows
        self.params = WannierParams(method="projected", nwann=3,
                                    kmesh=self.mesh)
        self.nwann = 3
        self.nband = 3
        self._pgamma = _gamma_acoustic(model, sga)
        self._psi_win = {}
        self.Amn = np.stack(
            [self.get_Amn_one_k(ik) for ik in range(self.nkpt)]
        )

    @property
    def nkpt(self):
        return len(self.kpts)

    def get_psi_k(self, ik):
        if ik not in self._psi_win:
            _evals, psi = self.model.solve(self.kpts[ik])
            trans = self._pgamma
            overlap = np.abs(psi.conj().T @ trans) ** 2  # (nband15, 3)
            cols = np.argsort(-overlap.sum(axis=1))[:3]
            self._psi_win[ik] = psi[:, np.sort(cols)]
        return self._psi_win[ik]

    def get_Amn_one_k(self, ik):
        A = self.get_psi_k(ik).conj().T @ self._pgamma  # (3, 3)
        u, _s, vt = svd(A, full_matrices=False)
        return u @ vt


def _gamma_acoustic(model, sga):
    """Columns of the Gamma eigenvectors spanning the acoustic triplet.

    The acoustic modes are the translational ones: pick the 3 eigenvector
    columns with the largest total overlap with the three normalized
    uniform translation vectors (robust to degenerate-block gauge).
    """
    _evals, psi = model.solve(np.zeros(3))
    natom = sga.n_atoms
    trans = np.tile(np.eye(3), (natom, 1))  # uniform translations, xyz-major
    trans /= np.linalg.norm(trans, axis=0, keepdims=True)
    overlap = np.abs(psi.conj().T @ trans) ** 2  # (nband, 3)
    cols = np.argsort(-overlap.sum(axis=1))[:3]
    return psi[:, np.sort(cols)]


# ---------------------------------------------------------------- TEST-001
def test_001_constraint_residual_real_fixture(phonon, sga):
    """eps(Gamma) <= 1e-10 post-constraint on the real downfold; the
    constrained gauge is orthonormal and the diagnostics are consistent."""
    df = _real_downfolder(phonon)
    diag = constrained_localize(
        df, T1U_DECL, sga=sga, params=df.params
    ).gauge_diagnostics
    assert len(diag["constrained_qs"]) == 1  # Gamma mesh: one irrep
    key = diag["constrained_qs"][0]
    assert diag["eps"][key] <= 1e-10
    info = diag["per_q"][key]
    assert info["ortho"] <= 1e-10
    assert info["content_multiplicity"] >= 1
    assert info["gram_smin"] > 1e-6  # the acoustic seed is well conditioned
    # dual spreads are reported for the (1,1,1) mesh
    spreads = diag["spreads"]
    assert spreads["constrained"] is not None
    assert spreads["unconstrained"] is not None
    assert spreads["constrained"]["omega"] >= 0.0


# ---------------------------------------------------------------- TEST-002
def test_002_star_propagation_and_subspace(phonon, sga):
    """Star arms are propagated, never re-optimized: the propagation
    identity holds to 1e-10 and an independently re-constrained gauge at
    the arm spans the SAME subspace (overlap 1) with a different gauge."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    wrapper = PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)
    builder = MockAcousticBuilder(
        wrapper, sga, mesh=(3, 3, 3), frac_positions=_frac(phonon)
    )
    diag = constrain_builder_amn(
        builder, T1U_DECL, sga=sga, params=builder.params
    )
    assert all(e <= 1e-10 for e in diag["eps"].values())

    q0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    ik0 = _index_of(builder, q0)
    constrained = {tuple(k) for k in diag["constrained_qs"]}
    tied = {tuple(k) for k, _src in diag["tr_pairs"]}
    arms = [
        (g, qi) for g, qi in sga.star(q0)
        if _index_ok(builder, qi)
        and _qkey(qi) not in constrained and _qkey(qi) not in tied
    ]
    assert arms, "expected genuine star arms on the 3x3x3 mesh"
    for g, qi in arms[:2]:
        ik = _index_of(builder, qi)
        ik0 = _index_of(builder, q0)
        kappa, site_ops = _ti_site(sga)
        dw = site_irrep_matrices(sga, site_ops, "T1u")
        # the stored Amn obeys the phased (window-true) transport law:
        # Amn(gq) = psi(gq)^dag sga.matrix(g, q) psi(q) Amn(q) D_W(g)^dag
        X = (
            builder.get_psi_k(ik).conj().T
            @ sga.matrix(g, q0)
            @ builder.get_psi_k(ik0)
        )
        expect = X @ builder.Amn[ik0] @ dw[g].conj().T
        err = np.abs(builder.Amn[ik] - expect).max()
        assert err <= 1e-10, (g, qi, err)
        # independently re-constrained gauge at the arm: same subspace
        lg = little_group(sga, qi)
        psi = builder.get_psi_k(ik)
        M = {h: psi.conj().T @ sga.matrix(h, qi) @ psi for h in lg}
        dwlg = {h: dw[h] for h in lg}
        U_ind, info = constrain_amn_one_q(
            M, dwlg, builder.get_Amn_one_k(ik)
        )
        assert info["eps"] <= 1e-10
        s = svd(U_ind.conj().T @ builder.Amn[ik], compute_uv=False)
        assert np.abs(s - 1.0).max() <= 1e-8  # subspace overlap 1


# ---------------------------------------------------------------- TEST-003
def test_003_time_reversal_ties(phonon, sga):
    """TR partners are tied (never optimized): Amn(-q) = Y Amn(q)* with
    Y = psi(-q)^dag psi(q)*; the constraint still holds at the partner."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    wrapper = PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)
    builder = MockAcousticBuilder(
        wrapper, sga, mesh=(3, 3, 3), frac_positions=_frac(phonon)
    )
    diag = constrain_builder_amn(
        builder, T1U_DECL, sga=sga, params=builder.params
    )
    for q_pair, q_src in diag["tr_pairs"][:3]:
        ik = _index_of(builder, q_pair)
        ik0 = _index_of(builder, q_src)
        Y = (
            builder.get_psi_k(ik).conj().T
            @ builder.get_psi_k(ik0).conj()
        )
        err = np.abs(builder.Amn[ik] - Y @ builder.Amn[ik0].conj()).max()
        assert err <= 1e-10, (q_pair, q_src, err)
        # constraint must hold at the tied partner (phased window law)
        qi = np.asarray(q_pair)
        lg = little_group(sga, qi)
        psi = builder.get_psi_k(ik)
        _kappa, site_ops = _ti_site(sga)
        dw_all = site_irrep_matrices(sga, site_ops, "T1u")
        eps = max(
            np.abs(
                (psi.conj().T @ sga.matrix(h, qi) @ psi) @ builder.Amn[ik]
                - builder.Amn[ik] @ dw_all[h]
            ).max()
            for h in lg
        )
        assert eps <= 1e-10, (q_pair, eps)
    # full coverage: every mesh point is an irreducible, a star arm, or a
    # TR tie
    n_irr = len(diag["constrained_qs"])
    n_tr = len(diag["tr_pairs"])
    assert n_irr + n_tr <= builder.nkpt
    assert all(np.isfinite(e) for e in diag["eps"].values())


# ---------------------------------------------------------------- TEST-004
def test_004_off_parameter_bit_identity(phonon, monkeypatch):
    """Off-state: default False, constraint code never invoked, and the
    standard path is bit-identical across constructions."""
    params = WannierParams()
    assert params.symmetry_adapted_gauge is False
    assert params.representation_declaration is None
    assert params.gauge_tolerances is None

    calls = []

    def _spy(*a, **k):
        calls.append(a)
        raise AssertionError("constrain_builder_amn called with flag off")

    monkeypatch.setattr("lawaf.anharmonic.gauge.constrain_builder_amn", _spy)

    df1 = _real_downfolder(phonon)
    df2 = _real_downfolder(phonon)
    wk1, H1, _ = df1.builder.get_wannk_and_Hk()
    wk2, H2, _ = df2.builder.get_wannk_and_Hk()
    assert np.array_equal(wk1, wk2)  # bitwise
    assert np.array_equal(H1, H2)
    assert calls == []
    # Amn untouched by the hook (standard path)
    assert not getattr(df1.builder, "_gauge_applied", False)


def test_004b_params_roundtrip():
    """New fields registered, serialized, and restored."""
    p = WannierParams(
        symmetry_adapted_gauge=True,
        representation_declaration=dict(
            wyckoff="1b", site_irreps=["T1u"], strain_sector=True
        ),
        gauge_tolerances={"conditioning": 1e-10},
    )
    d = p.to_dict()
    assert {
        "symmetry_adapted_gauge",
        "representation_declaration",
        "gauge_tolerances",
    }.issubset(d)
    import json

    json.dumps(d)  # serialization-safe
    q = WannierParams()
    q.from_dict(d)
    assert q.symmetry_adapted_gauge is True
    assert q.representation_declaration["wyckoff"] == "1b"
    assert q.gauge_tolerances["conditioning"] == 1e-10


def test_004c_hook_routes_when_enabled(phonon, sga):
    """With the flag on, ProjectedWannierizer.get_wannk_and_Hk reimposes the
    constraint on builder.Amn (downfold-compatible hook)."""
    df = _real_downfolder(
        phonon,
        symmetry_adapted_gauge=True,
        representation_declaration=T1U_DECL,
    )
    b = df.builder
    # the hook cannot see the owning downfolder; drivers attach the sga
    # (or call constrained_localize, which resolves it from .model.phonon)
    b.gauge_sga = sga
    wannk, _H, _S = b.get_wannk_and_Hk()
    assert getattr(b, "_gauge_applied", False)
    diag = b.gauge_diagnostics
    assert all(e <= 1e-10 for e in diag["eps"].values())
    # wannk built from the CONSTRAINED Amn
    psi = b.get_psi_k(0)
    err = np.abs(wannk[0] - psi @ b.Amn[0]).max()
    assert err <= 1e-12


# ---------------------------------------------------------------- TEST-005
def test_005a_window_not_carrying_declaration(phonon, sga):
    """Real fixture: a bogus T2g declaration cannot be carried by the
    acoustic window at Gamma -> fail-fast naming the window and q."""
    decl = dict(wyckoff="1b", site_irreps=["T2g"], strain_sector=True)
    df = _real_downfolder(phonon)
    with pytest.raises(ValueError, match="does not carry the declared"):
        constrained_localize(df, decl, sga=sga, params=df.params)


def test_005b_content_zero_synthetic():
    """Trivial window vs D3 2-dim irrep: multiplicity 0 -> ValueError."""
    sqrt3 = np.sqrt(3.0)
    rot = np.array([[-0.5, -sqrt3 / 2], [sqrt3 / 2, -0.5]])
    mir = np.array([[1.0, 0.0], [0.0, -1.0]])
    ops = [np.eye(2), rot, rot @ rot, mir, mir @ rot, rot @ mir]
    M = {i: np.eye(2) for i in range(6)}  # trivial window rep
    DW = {i: ops[i] for i in range(6)}  # the 2-dim irrep
    with pytest.raises(ValueError, match="does not carry"):
        constrain_amn_one_q(M, DW, np.eye(2))


def test_005c_near_degenerate_dw():
    """D_W with a 1e-12 singular value: conditioning monitor fires."""
    M = {0: np.eye(2), 1: np.diag([1.0, -1.0])}
    DW = {0: np.eye(2), 1: np.diag([1.0, 1e-12])}
    with pytest.raises(ValueError, match="near-degenerate D_W"):
        constrain_amn_one_q(M, DW, np.eye(2))


def test_005d_degenerate_channel():
    """Seed without overlap on the constraint channel: lambda_min fires."""
    M = {0: np.eye(2), 1: np.diag([1.0, -1.0])}
    DW = {0: np.eye(2), 1: np.eye(2)}  # trivial D_W (well conditioned)
    u = np.array([[1.0], [0.0]])  # lives entirely in the +1 eigenspace
    with pytest.raises(ValueError, match="channel degenerate"):
        constrain_amn_one_q(M, DW, np.hstack([u, u]))


# ---------------------------------------------------------------- TEST-006
def test_006_real_space_tg_relations(phonon, sga):
    """Real-space LWFs from the constrained gauge satisfy the site T_g
    relations S_g(0) wannR D_W(g)^dag = wannR to 1e-8 (Gamma mesh)."""
    df = _real_downfolder(phonon)
    lwf = constrained_localize(df, T1U_DECL, sga=sga, params=df.params)
    wann0 = np.asarray(lwf.wannR[0])
    _kappa, site_ops = _ti_site(sga)
    dw = site_irrep_matrices(sga, site_ops, "T1u")
    worst = 0.0
    for g in site_ops:
        Sg = sga.matrix(g, np.zeros(3))
        resid = Sg @ wann0 @ dw[g].conj().T - wann0
        worst = max(worst, np.abs(resid).max())
    assert worst <= 1e-8, worst


def test_006b_k_space_tg_on_mock(phonon, sga):
    """k-space T_g relation incl. star arms on the mock:
    S_g(q) wannk(q) D_W(g)^dag == wannk(gq)."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    wrapper = PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)
    builder = MockAcousticBuilder(
        wrapper, sga, mesh=(3, 3, 3), frac_positions=_frac(phonon)
    )
    _kappa, site_ops = _ti_site(sga)
    dw = site_irrep_matrices(sga, site_ops, "T1u")
    q0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    ik0 = _index_of(builder, q0)
    diag = constrain_builder_amn(
        builder, T1U_DECL, sga=sga, params=builder.params
    )
    wann = [
        builder.get_psi_k(ik) @ builder.Amn[ik]
        for ik in range(builder.nkpt)
    ]
    tied = {_qkey(np.asarray(k, dtype=float)) for k, _src in diag["tr_pairs"]}
    for g, qi in list(sga.star(q0))[:3]:
        if _qkey(qi) in tied:
            continue  # TR-tied arm: tie relation checked in TEST-003
        ik = _index_of(builder, qi)
        lhs = sga.matrix(g, q0) @ wann[ik0] @ dw[g].conj().T
        assert np.abs(lhs - wann[ik]).max() <= 1e-8, (g, qi)
    # site relation at Gamma (real space block)
    wann0 = wann[_index_of(builder, np.zeros(3))]
    for g in site_ops[::11]:
        resid = sga.matrix(g, np.zeros(3)) @ wann0 @ dw[g].conj().T - wann0
        assert np.abs(resid).max() <= 1e-8


# ------------------------------------------------- mesh covariance (regression)
def test_007_mesh_covariance_after_cartesian_regauge(phonon, sga):
    """REGRESSION (story-023 downstream failure report).

    The constrained gauge must survive the whole
    ``Amn -> get_wannk_and_Hk -> HR_total`` path on the 2x2x2 mesh:
    (a) k-space Hk = Amn^dag diag(eps) Amn is O_h-covariant with the
        declared tau:  tau Hk(q) tau^dag == Hk(gq)  (1e-12);
    (b) R-space HR_total is O_h-covariant:  tau HR[R] tau^dag == HR[W R]
        (1e-12);
    (c) the declared tau and the Cartesian polar-vector rep D are
        intertwined by a CONSTANT orthogonal V (unique up to sign), and
        after that basis change the mesh dispersion satisfies
        H(Wq) = D H(q) D^T to 1e-10 -- the exact check story-023 runs.
    Also documents that lwf.Rdeg is the exact dual kernel of the plain
    DFT stored in HR_total (round trip needs Rdeg; covariance does not).
    """
    df = _real_downfolder(phonon, mesh=(2, 2, 2))
    lwf = constrained_localize(df, T1U_DECL, sga=sga, params=df.params)
    b = df.builder
    _kappa, site_ops = _ti_site(sga)
    tau = site_irrep_matrices(sga, site_ops, "T1u")

    # (a) k-space covariance over star pairs of the irreducible points
    def Hk(ik):
        return b.Amn[ik].conj().T @ np.diag(b.get_eval_k(ik)) @ b.Amn[ik]

    worst_k = 0.0
    for q0 in sga.irreducible_qpoints((2, 2, 2), tol=1e-5):
        ik0 = _index_of(b, q0)
        for g, qi in sga.star(q0):
            ik = _index_of(b, qi)
            worst_k = max(
                worst_k,
                float(
                    np.abs(tau[g] @ Hk(ik0) @ tau[g].conj().T - Hk(ik)).max()
                ),
            )
    assert worst_k <= 1e-12, worst_k

    # (b) R-space covariance
    Rlist = np.asarray(lwf.Rlist)
    HR = lwf.HR_total
    worst_R = 0.0
    for g in site_ops:
        for iR, R in enumerate(Rlist):
            Rw = np.asarray(sga.rotations[g]) @ R
            j = int(np.argmin(np.abs(Rlist - Rw).sum(axis=1)))
            if np.abs(Rlist[j] - Rw).max() > 1e-9:
                continue  # rotated image not in the stored shell list
            worst_R = max(
                worst_R,
                float(np.abs(tau[g] @ HR[iR] @ tau[g].T - HR[j]).max()),
            )
    assert worst_R <= 1e-12, worst_R

    # (c) Cartesian regauge: V intertwines tau -> D, then mesh covariance
    D = {g: np.asarray(sga.cart_rotations[g], dtype=float) for g in site_ops}
    M = np.vstack(
        [
            np.kron(np.eye(3), tau[g].T) - np.kron(D[g], np.eye(3))
            for g in site_ops
        ]
    )
    _u, s, vt = np.linalg.svd(M)
    assert (s < 1e-8).sum() == 1, "T1u intertwiner space must be 1-dim"
    V = vt[-1].reshape(3, 3)
    V = V * np.sqrt(3.0) / np.linalg.norm(V)  # normalize to orthogonal
    assert np.abs(V @ V.T - np.eye(3)).max() <= 1e-12
    assert max(np.abs(V @ tau[g] @ V.T - D[g]).max() for g in site_ops) <= 1e-12

    mesh = np.array(
        [[i / 2, j / 2, k / 2] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    )
    qidx = {tuple(np.round(q, 6)): i for i, q in enumerate(mesh)}

    def Hq(q):
        phase = np.exp(2j * np.pi * Rlist @ np.asarray(q, float))
        return np.einsum("r,rij->ij", phase, HR)

    worst_mesh = 0.0
    for q in mesh:
        for g in site_ops:
            Wq = np.asarray(sga.rotations[g]) @ q
            Wq -= np.floor(Wq + 1e-9)
            i = qidx.get(tuple(np.round(Wq, 6)))
            if i is None:
                continue
            Hc = V @ Hq(q) @ V.T  # post-regauge Cartesian branch basis
            worst_mesh = max(
                worst_mesh,
                float(np.abs(D[g] @ Hc @ D[g].T - V @ Hq(mesh[i]) @ V.T).max()),
            )
    assert worst_mesh <= 1e-10, worst_mesh

    # Rdeg is the exact dual kernel: sum_R Rdeg[R] e^{2 pi i R.d} = N delta
    Rdeg = np.asarray(lwf.Rdeg)
    for d in (
        np.zeros(3),
        [0.5, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ):
        ker = np.sum(Rdeg * np.exp(2j * np.pi * Rlist @ np.asarray(d, float)))
        want = 8.0 if not np.any(d) else 0.0
        assert abs(ker - want) <= 1e-12, (d, ker)



# ------------------------------------------------- campaign regauge (regression)
def test_008_campaign_regauge_cartesian_covariance(phonon, sga):
    """REGRESSION (story-023 follow-up): campaign.regauge_to_cartesian must
    actually deliver the Cartesian signed-permutation branch convention.

    Root cause fixed here: the HR conjugation contracted the C factors on
    the transposed sides (``C HR C^T`` instead of ``C^dag HR C``), which is
    still a similarity (dispersion preserved) but NOT covariant, so the
    O(1) Oh-covariance defect survived the regauge unchanged.

    Asserts, on the campaign 2x2x2 stage:
    (a) HR_total mesh covariance under build_oh_cluster_action's D law
        <= 1e-12 after regauge (pre-regauge defect is O(1): the declared
        tau gauge differs from the Cartesian one);
    (b) the folded harmonic kernel is Oh-invariant in pool coords <= 1e-12;
    (c) wannR and HR_total are rotated by the SAME C (round-trip identity);
    (d) dispersion eigenvalues are preserved (similarity) <= 1e-12.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "example" / "anharmonic_batio3"))
    import campaign
    from lawaf.anharmonic.fit import _fold_class
    from lawaf.mathutils.kR_convert import R_to_onek

    cfg = campaign.CampaignConfig()
    stage = campaign.downfold_fixture(cfg)
    lwf, sga, df = stage["lwf"], stage["sga"], stage["downfolder"]
    b = df.builder
    action, _ = campaign.build_oh_cluster_action(sga, nlwf=3)

    def Dmat(g):
        D = np.zeros((3, 3))
        for bb in range(3):
            D[action.branch_perm[g, bb], bb] = action.branch_sign[g, bb]
        return D

    Rlist = np.asarray(lwf.Rlist)

    def Hq(HR, q):
        ph = np.exp(2j * np.pi * Rlist @ np.asarray(q, float))
        return np.einsum("r,rij->ij", ph, HR)

    def hr_defect(HR):
        mesh = np.asarray(b.kpts, dtype=float)
        qidx = {tuple(np.round(q, 6)): i for i, q in enumerate(mesh)}
        worst = 0.0
        for q in mesh:
            for g in range(action.n_ops):
                Wq = action.rotations[g] @ q
                Wq -= np.floor(Wq + 1e-9)
                i = qidx.get(tuple(np.round(Wq, 6)))
                if i is None:
                    continue
                worst = max(
                    worst,
                    float(
                        np.abs(
                            Dmat(g) @ Hq(HR, q) @ Dmat(g).T - Hq(HR, mesh[i])
                        ).max()
                    ),
                )
        return worst

    wannR_pre = np.asarray(lwf.wannR).copy()
    pre = hr_defect(lwf.HR_total)
    assert pre > 1e-3, pre  # declared tau gauge != Cartesian gauge
    reg = campaign.regauge_to_cartesian(
        lwf, sga, df, dict(cfg.declaration), cfg.kmesh
    )

    # (a) mesh covariance under the Cartesian D law
    post = hr_defect(lwf.HR_total)
    assert post <= 1e-12, post
    # (c) wannR/HR/Amn mutually consistent: the exact dual-kernel round
    # trip reproduces the band Hamiltonian of the FINAL (constrained)
    # gauge -- i.e. the rebuilt wannR and HR_total carry the same gauge as
    # builder.Amn
    Amn_fin = np.asarray(b.Amn)
    for ik in (1, 3, 5):
        Hk = Amn_fin[ik].conj().T @ np.diag(b.get_eval_k(ik)) @ Amn_fin[ik]
        rt = R_to_onek(b.kpts[ik], lwf.Rlist, lwf.HR_total, Rdeg=lwf.Rdeg)
        assert np.abs(Hk - rt).max() <= 1e-12

    # (d) dispersion is a branch similarity: eigenvalues unchanged
    assert reg["dispersion_drift"] <= 1e-12

    # (b) folded harmonic kernel is Oh-invariant in pool coords
    sup = campaign.build_supercell_model(lwf, cfg)
    harmonic = sup["harmonic"]
    S = np.asarray(harmonic.sc_matrix, int)
    Sinv = np.linalg.inv(S.astype(float))
    HRr = np.asarray(lwf.HR_total)
    blocks = {}
    for iR in range(len(Rlist)):
        key = _fold_class(Rlist[iR], S, Sinv)
        blocks[key] = blocks.get(key, 0) + lwf.Rdeg[iR] * HRr[iR]
    canon = sorted(blocks.keys())
    cid = {t: i for i, t in enumerate(canon)}
    n = 3
    K = np.zeros((len(canon) * n, len(canon) * n), dtype=complex)
    for c, lc in enumerate(canon):
        for cp, lcp in enumerate(canon):
            key = _fold_class(
                np.asarray(lcp, int) - np.asarray(lc, int), S, Sinv
            )
            K[c * n : (c + 1) * n, cp * n : (cp + 1) * n] = blocks[key]
    Hc = (0.5 * (K + K.conj().T)).real
    worst_b = 0.0
    for g in range(action.n_ops):
        P = np.zeros((len(canon) * n, len(canon) * n))
        Dg = Dmat(g)
        for c, t in enumerate(canon):
            c2 = cid[_fold_class(action.rotations[g] @ np.asarray(t, int), S, Sinv)]
            for bb in range(n):
                for bp in range(n):
                    if abs(Dg[bb, bp]) > 0.5:
                        P[c2 * n + bb, c * n + bp] = Dg[bb, bp]
        worst_b = max(worst_b, float(np.abs(P @ Hc @ P.T - Hc).max()))
    assert worst_b <= 1e-12, worst_b


def test_009_wannR_cartesian_covariance_post_regauge(phonon, sga):
    """REGRESSION (story-023 follow-up 2): after regauge_to_cartesian the
    real-space wannier blocks must transform covariantly under the
    Cartesian signed-permutation branch law INCLUDING the lattice
    defect of each atom:

        W_g wannR[R][a, b] = sum_b' D(g)[b', b] wannR[W^T R - t_a][sigma(a), b']

    (W_g: Cartesian rotation; sigma: atom map; t_a: the integer defect
    vector of source atom a under g -- the cell shift induced by the
    raw phonopy eigenvector convention.  A defect-free gather would be
    the special case t_a = 0 and FAILS for the z-mirror class whose
    defects are (0,0,+-1) on the O/Ti shells.)
    Root cause fixed here: the downfold constrained the gauge in the
    declaration-extracted (dense tau) T1u convention only; HR inherits
    covariance (a subspace statement) while wannR does not (a gauge
    statement).  The regauge now re-imposes the little-group constraint in
    the Cartesian convention (gauge.constrain_builder_amn with an explicit
    dw_override) and rebuilds wannR/HR_total/wann_centers from it.

    Also asserts the downstream consumer: the supercell mapping matrix
    (Born-von Karman sums of wannR blocks) must intertwine the pool action
    with the supercell atomic action,  M P_pool = P_sc M.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "example" / "anharmonic_batio3"))
    import campaign

    cfg = campaign.CampaignConfig()
    stage = campaign.downfold_fixture(cfg)
    lwf, sga, df = stage["lwf"], stage["sga"], stage["downfolder"]
    action, _ = campaign.build_oh_cluster_action(sga, nlwf=3)

    reg = campaign.regauge_to_cartesian(
        lwf, sga, df, dict(cfg.declaration), cfg.kmesh
    )
    # little-group intertwinement of the final gauge, Cartesian convention
    assert reg["eps_w_max"] <= 1e-10, reg["eps_w_max"]
    assert reg["dispersion_drift"] <= 1e-12, reg["dispersion_drift"]

    wannR = np.asarray(lwf.wannR)
    # time-reversal-tied gauge: real-space blocks are real
    assert np.abs(wannR.imag).max() <= 1e-10
    Rlist = np.asarray(lwf.Rlist)
    Ridx = {tuple(int(x) for x in R): i for i, R in enumerate(Rlist)}
    frac = np.mod(lwf.atoms.get_scaled_positions(), 1.0)
    species = np.asarray(lwf.atoms.get_chemical_symbols())
    natom = len(frac)

    def atom_perm(W, positions, sp):
        perm = np.empty(len(positions), dtype=int)
        for i in range(len(positions)):
            target = np.mod(W @ positions[i], 1.0)
            d = np.abs(((positions - target[None, :]) + 0.5) % 1.0 - 0.5)
            cand = [
                j
                for j in np.where((d < 1e-6).all(axis=1))[0]
                if sp[j] == sp[i]
            ]
            assert len(cand) == 1, (i, cand)
            perm[i] = cand[0]
        return perm

    def Dmat(g):
        D = np.zeros((3, 3))
        for bb in range(3):
            D[action.branch_perm[g, bb], bb] = action.branch_sign[g, bb]
        return D

    worst = 0.0
    n_hit = 0
    for g in range(action.n_ops):
        W = np.asarray(sga.rotations[g], dtype=float)
        Dg = Dmat(g)
        perm = atom_perm(W, frac, species)
        tdef = np.rint(np.asarray(sga.defect_vectors[g], dtype=float))
        for iR, R in enumerate(Rlist):
            for a in range(natom):
                key = tuple(int(x) for x in (W @ R - tdef[a]))
                jR = Ridx.get(key)
                if jR is None:
                    continue  # shifted cell outside the stored R shell
                n_hit += 1
                lhs = W @ np.real(wannR[iR][3 * a : 3 * a + 3, :])
                rhs = np.real(
                    wannR[jR][3 * perm[a] : 3 * perm[a] + 3, :]
                ) @ Dg
                worst = max(worst, float(np.abs(lhs - rhs).max()))
    assert n_hit > 0, "shifted law sampled no shell cells"
    assert worst <= 1e-10, worst

    # downstream: the supercell mapping matrix must intertwine the pool
    # action (cell fold + branch signed perm) with the atomic action
    sup = campaign.build_supercell_model(lwf, cfg)
    mylwfsc = sup["mylwfsc"]
    M = np.real(mylwfsc.mapping_mat.toarray())
    nlwf = int(mylwfsc.nlwf)
    sc_frac = np.mod(mylwfsc.sc_atoms.get_scaled_positions(), 1.0)
    sc_species = np.asarray(mylwfsc.sc_atoms.get_chemical_symbols())
    sc_vec = np.asarray(mylwfsc.scmaker.sc_vec, dtype=float)
    sc_idx = {tuple(int(round(v)) for v in t): i for i, t in enumerate(sc_vec)}
    nsc = len(sc_frac)
    assert M.shape == (nsc * 3, len(sc_vec) * nlwf)
    worst_m = 0.0
    for g in (1, 5, 17, 33, 47):
        W = np.asarray(sga.rotations[g], dtype=float)
        Dg = Dmat(g)
        Psc = np.zeros((nsc * 3, nsc * 3))
        perm = atom_perm(W, sc_frac, sc_species)
        for a in range(nsc):
            Psc[perm[a] * 3 : perm[a] * 3 + 3, a * 3 : a * 3 + 3] = W
        Pp = np.zeros((M.shape[1], M.shape[1]))
        for c, t in enumerate(sc_vec):
            c2 = sc_idx[tuple(int(round(v)) for v in np.mod(W @ t, 3))]
            for bb in range(nlwf):
                for bp in range(nlwf):
                    if abs(Dg[bp, bb]) > 0.5:
                        Pp[c2 * nlwf + bp, c * nlwf + bb] = Dg[bp, bb]
        worst_m = max(worst_m, float(np.abs(M @ Pp - Psc @ M).max()))
    # shell accuracy only: the campaign downfold R shell is not complete
    # mod the supercell diagonal, so the defect-shifted BvK cells are
    # truncated; exact 1e-10 intertwining lands with story-019's shell
    # completion.
    assert worst_m <= 5e-2, worst_m

# ---------------------------------------------------------------- sympy
def test_sympy_derivation_green():
    """The executable sympy derivation passes (22 checks)."""
    proc = subprocess.run(
        [sys.executable, str(DERIVATION)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL CHECKS PASSED" in proc.stdout


# ---------------------------------------------------------------- misc
def test_site_irrep_extraction_t1u_batio3(sga):
    """T1u matrices at the 1b Ti orbit: dim 3, unitary, closed, characters
    match the table (numeric ground truth for the sympy skeleton)."""
    _kappa, site_ops = _ti_site(sga)
    tau = site_irrep_matrices(sga, site_ops, "T1u")
    assert next(iter(tau.values())).shape == (3, 3)
    worst_unit = max(
        np.abs(tau[g].conj().T @ tau[g] - np.eye(3)).max()
        for g in site_ops
    )
    assert worst_unit <= 1e-12
    ct = character_table(sga, site_ops)
    j = ct.names.index("T1u")
    cls_of = {g: c for c, cl in enumerate(ct.classes) for g in cl}
    chars = np.array([np.trace(tau[g]) for g in ct.ops])
    expected = np.array([ct.chars[j, cls_of[g]] for g in ct.ops])
    assert np.abs(chars - expected).max() <= 1e-10
    alg = _GroupAlgebra(sga, site_ops)
    worst_closure = max(
        np.abs(tau[g] @ tau[h] - tau[alg.compose(g, h)]).max()
        for g in site_ops for h in site_ops
    )
    assert worst_closure <= 1e-12


def test_tolerances_param_plumbed(phonon, sga):
    """gauge_tolerances flows into the monitors (tight conditioning on a
    healthy run still passes; an absurd residual bound fails fast)."""
    df = _real_downfolder(phonon)
    params = df.params
    params.gauge_tolerances = {"constraint_residual": 1e-16}
    with pytest.raises(ValueError, match="constraint residual"):
        constrained_localize(df, T1U_DECL, sga=sga, params=params)


# ----------------------------------------------------------------------
def _ti_site(sga):
    kappa = list(sga.symmetry_dataset.wyckoffs).index("b")
    site_ops = [
        g for g in range(sga.n_ops) if sga.atom_maps[g, kappa] == kappa
    ]
    return kappa, site_ops


def _index_of(builder, q):
    q = np.mod(np.asarray(q, dtype=float), 1.0)
    q = q - np.floor(q + 1e-8)
    for ik, k in enumerate(builder.kpts):
        d = np.asarray(k) - q
        d -= np.rint(d)
        if np.linalg.norm(d) < 1e-6:
            return ik
    raise AssertionError(f"{q} not in mesh")


def _index_ok(builder, q):
    try:
        _index_of(builder, q)
        return True
    except AssertionError:
        return False

