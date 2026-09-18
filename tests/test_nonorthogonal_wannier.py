"""Non-orthogonalized Wannier function contracts.

Covers the ``params.orthogonal=False`` surface (quick task
2026-09-18-nonorthogonal-wannier):

- projected wannierizations keep the RAW gauge (``A^dag A != I``) and the
  overlap propagates through the whole pipeline (``Swann_k -> SwannR``,
  WS-folded with H), while ``orthogonal=True`` orthonormalizes;
- scdm-k ALWAYS orthonormalizes (its occupation weighting is designed to
  be undone by the polar factor; raw weighted columns are rank-deficient);
- ``Lawaf.set_parameters`` defaults to ``orthogonal=True``;
- bands come from the pencil ``(H^w, S^w)`` and, on the downfolding mesh,
  are identical to the orthonormal-gauge control (same subspace, exact
  Rayleigh-Ritz equality) — checked on a synthetic non-orthogonal LCAO
  model (electron, exact to machine precision on a full manifold) and on
  the BaTiO3 DM fixture (phonon);
- netcdf round trips preserve ``SwannR``.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import eigh

from lawaf.interfaces.downfolder import Lawaf
from lawaf.interfaces.phonopy.phonon_downfolder import PhonopyDownfolder
from lawaf.params import WannierParams
from lawaf.wannierization import ProjectedWannierizer, ScdmkWannierizer

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"


# ---------------------------------------------------------------- synthetic
class NonOrthoLCAOModel:
    """Two-band model on a contracted basis: H' = X^dag H X, S' = X^dag X.

    The parent is an orthogonal nearest-neighbour tight-binding model, so
    the pencil (H', S') has the parent's exact spectrum and any full-rank
    gauge reproduces it — the exactness the electron assertions rely on.
    """

    is_orthogonal = False

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        nb = 2
        self.nb = nb
        rlist = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)

        def herm(scale):
            a = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
            return (a + a.conj().T) * scale

        hop = herm(0.3)
        X = np.eye(nb) + 0.4 * (
            rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        )
        assert np.linalg.cond(X) < 20  # well-conditioned contraction
        self.X = X
        self.H0 = X.conj().T @ herm(1.0) @ X
        self.hop = X.conj().T @ hop @ X
        self.S = X.conj().T @ X
        self._rlist = rlist
        self.atoms = Atoms("H2", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3))

    def _Hk(self, k):
        Hk = self.H0.copy()
        for R in self._rlist[1:]:
            ph = np.exp(2j * np.pi * np.dot(R, k))
            Hk += self.hop * ph + self.hop.conj().T * ph.conj()
        return Hk

    def HS_and_eigen(self, kpts):
        nk = len(kpts)
        Hks = np.zeros((nk, self.nb, self.nb), dtype=complex)
        Sks = np.zeros((nk, self.nb, self.nb), dtype=complex)
        evals = np.zeros((nk, self.nb))
        evecs = np.zeros((nk, self.nb, self.nb), dtype=complex)
        for ik, k in enumerate(kpts):
            Hk = self._Hk(k)
            Hks[ik], Sks[ik] = Hk, self.S
            evals[ik], evecs[ik] = eigh(Hk, self.S)
        return Hks, Sks, evals, evecs

    def solve_all(self, kpts):
        return self.HS_and_eigen(kpts)[2:]


@pytest.fixture(scope="module")
def synth():
    return NonOrthoLCAOModel()


def _downfolded(model, orthogonal):
    params = dict(
        method="projected",
        kmesh=(3, 3, 3),
        nwann=2,
        selected_basis=[0, 1],
        weight_func="unity",
        orthogonal=orthogonal,
    )
    df = Lawaf(model, params=params)
    return df, df.downfold()


def test_electron_nonorthogonal_exact_and_has_overlap(synth):
    df, lwf = _downfolded(synth, False)
    assert not lwf.is_orthogonal
    assert lwf.SwannR is not None
    i0 = int(np.where(np.all(lwf.Rlist == 0, axis=1))[0][0])
    # raw projected gauge of a contracted basis: S^w = A^dag A != I
    assert np.linalg.norm(lwf.SwannR[i0] - np.eye(2)) > 1e-3
    # on the mesh the pencil reproduces the parent spectrum exactly
    # (full manifold, Rayleigh-Ritz identity); off-mesh the raw gauge
    # pays an R-space truncation cost quantified below
    e_ref = np.array([eigh(synth._Hk(k), synth.S)[0] for k in df.kpts])
    e_lwf = np.array([lwf.solve_k(k)[0] for k in df.kpts])
    assert np.abs(e_ref - e_lwf).max() < 1e-10
    rng = np.random.default_rng(11)
    qtest = rng.uniform(-0.5, 0.5, size=(5, 3))
    e_off = np.array([eigh(synth._Hk(q), synth.S)[0] for q in qtest])
    e_off_lwf = np.array([lwf.solve_k(q)[0] for q in qtest])
    assert np.abs(e_off - e_off_lwf).max() < 0.05


def test_electron_orthogonal_control_matches(synth):
    df, lwf = _downfolded(synth, True)
    assert lwf.is_orthogonal
    assert lwf.SwannR is None
    e_ref = np.array([eigh(synth._Hk(k), synth.S)[0] for k in df.kpts])
    e_lwf = np.array([lwf.solve_k(k)[0] for k in df.kpts])
    assert np.abs(e_ref - e_lwf).max() < 1e-10


def test_electron_netcdf_roundtrip_preserves_swann(synth, tmp_path):
    _, lwf = _downfolded(synth, False)
    f = tmp_path / "ewf.nc"
    lwf.write_to_netcdf(f)
    from HamiltonIO.lawaf import LawafHamiltonian

    lwf2 = LawafHamiltonian.load_from_netcdf(f)
    assert not lwf2.is_orthogonal
    assert lwf2.SwannR is not None
    assert np.abs(lwf2.SwannR - lwf.SwannR).max() < 1e-12


def test_ewf_hs_and_eigen_refeed(synth):
    """A LawafHamiltonian re-fed as a model returns (Hks, Sks, ...)."""
    _, lwf = _downfolded(synth, False)
    H, S, e, v = lwf.HS_and_eigen(lwf.kpts)
    assert H.shape == (len(lwf.kpts), 2, 2)
    assert S is not None and S.shape == H.shape
    # S-orthonormal eigenvectors
    for ik in range(len(lwf.kpts)):
        assert np.abs(v[ik].conj().T @ S[ik] @ v[ik] - np.eye(2)).max() < 1e-10


# ---------------------------------------------------------------- builders
def _builder_inputs():
    rng = np.random.default_rng(5)
    nk, nb = 8, 6
    evecs = np.zeros((nk, nb, nb), dtype=complex)
    evals = rng.uniform(-1, 1, size=(nk, nb))
    for ik in range(nk):
        q, r = np.linalg.qr(
            rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        )
        evecs[ik] = q
    kpts = np.array(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)]
    ) / 2.0
    kweights = np.full(nk, 1.0 / nk)
    return evals, evecs, kpts, kweights


def test_projected_keeps_raw_gauge_when_nonorthogonal():
    evals, evecs, kpts, kweights = _builder_inputs()
    params = WannierParams(
        method="projected", kmesh=(2, 2, 2), nwann=3,
        selected_basis=[0, 1, 2], weight_func="unity", orthogonal=False,
    )
    wann = ProjectedWannierizer(
        evals=evals, evecs=evecs, kpts=kpts, kweights=kweights, params=params
    )
    Amn = wann.get_Amn()
    for ik in range(len(kpts)):
        gram = Amn[ik].conj().T @ Amn[ik]
        assert np.abs(gram - np.eye(3)).max() > 1e-3  # raw, not polarized
        # ... but still full rank
        assert np.linalg.eigvalsh(gram).min() > 1e-6
    _, _, Sk = wann.get_wannk_and_Hk()
    assert Sk is not None
    for ik in range(len(kpts)):
        assert np.abs(Sk[ik] - Amn[ik].conj().T @ Amn[ik]).max() < 1e-12


def test_projected_orthonormalizes_by_default():
    evals, evecs, kpts, kweights = _builder_inputs()
    params = WannierParams(
        method="projected", kmesh=(2, 2, 2), nwann=3,
        selected_basis=[0, 1, 2], weight_func="unity",
    )
    wann = ProjectedWannierizer(
        evals=evals, evecs=evecs, kpts=kpts, kweights=kweights, params=params
    )
    Amn = wann.get_Amn()
    for ik in range(len(kpts)):
        assert np.abs(Amn[ik].conj().T @ Amn[ik] - np.eye(3)).max() < 1e-10
    _, _, Sk = wann.get_wannk_and_Hk()
    assert Sk is None


def test_scdmk_always_orthonormalizes():
    evals, evecs, kpts, kweights = _builder_inputs()
    params = WannierParams(
        method="scdmk", kmesh=(2, 2, 2), nwann=3,
        selected_basis=[0, 1, 2], weight_func="unity", orthogonal=False,
    )
    wann = ScdmkWannierizer(
        evals=evals, evecs=evecs, kpts=kpts, kweights=kweights, params=params
    )
    Amn = wann.get_Amn()
    for ik in range(len(kpts)):
        assert np.abs(Amn[ik].conj().T @ Amn[ik] - np.eye(3)).max() < 1e-10


def test_set_parameters_defaults_to_orthogonal():
    class _M:
        is_orthogonal = True
        atoms = Atoms("H", cell=np.eye(3))

        def solve_all(self, kpts):
            return np.zeros((len(kpts), 1)), np.zeros(
                (len(kpts), 1, 1), dtype=complex
            )


    df = Lawaf(_M(), params=dict(method="scdmk", kmesh=(2, 2, 2), nwann=1))
    assert df.params.orthogonal is True


def _phonon(orthogonal, tmp_path):
    params = dict(
        method="projected",
        nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func="unity",
        kmesh=(2, 2, 2),
        gamma=True,
        orthogonal=orthogonal,
    )
    df = PhonopyDownfolder(
        phonopy_yaml=str(FIXTURE),
        mode="DM",
        params=params,
        symmetrize_fc=False,
        is_nac=False,
    )
    return df, df.downfold(
        output_path=str(tmp_path),
        write_hr_nc=None,
        write_hr_txt=None,
    )


def test_mlwf_refuses_nonorthogonal_gauge():
    from lawaf.wannierization.mlwf import MLWFWannierizer

    evals, evecs, kpts, kweights = _builder_inputs()
    params = WannierParams(
        method="mlwf", kmesh=(2, 2, 2), nwann=3,
        selected_basis=[0, 1, 2], weight_func="unity", orthogonal=False,
    )
    with pytest.raises(NotImplementedError, match="orthonormal gauges"):
        MLWFWannierizer(
            evals=evals, evecs=evecs, kpts=kpts, kweights=kweights,
            params=params,
        )


def test_windowed_selection_raw_gauge_refused(tmp_path):
    """window_bands zero out non-selected band rows; with the raw
    (non-orthonormalized) gauge the Gram A^dag A goes singular at the
    selection arms — refused with k context, not a scipy crash."""
    params = dict(
        method="projected",
        nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func="unity",
        kmesh=(2, 2, 2),
        gamma=True,
        orthogonal=False,
        window_bands={
            (0.0, 0.0, 0.0): (0, 1, 2),
            (0.5, 0.0, 0.0): (0, 1, 4),
            (0.5, 0.5, 0.0): (0, 1, 4),
            (0.5, 0.5, 0.5): (0, 1, 2),
        },
    )
    df = PhonopyDownfolder(
        phonopy_yaml=str(FIXTURE),
        mode="DM",
        params=params,
        symmetrize_fc=False,
        is_nac=False,
    )
    with pytest.raises(ValueError, match="rank-deficient"):
        df.downfold(output_path=str(tmp_path), write_hr_nc=None,
                    write_hr_txt=None)




def test_phonon_nonorthogonal_pencil(tmp_path):
    df, lwf = _phonon(False, tmp_path)
    assert lwf.SwannR is not None
    q = np.array([0.25, 0.11, -0.37])
    w = np.linalg.eigvalsh(lwf.get_Sk(q))
    assert np.all(w > 1e-6)
    assert np.abs(w - 1.0).max() > 1e-2  # non-trivial metric at generic q
    # on-mesh: same subspace as the orthonormal gauge -> identical bands
    df_or, lwf_or = _phonon(True, tmp_path)
    assert lwf_or.SwannR is None
    e_no = np.array([lwf.solve_k(k)[0] for k in df.kpts])
    e_or = np.array([lwf_or.solve_k(k)[0] for k in df_or.kpts])
    assert np.abs(e_no - e_or).max() < 1e-8


def test_phonon_netcdf_roundtrip_preserves_swann(tmp_path):
    _, lwf = _phonon(False, tmp_path)
    f = tmp_path / "lwf.nc"
    lwf.write_to_netcdf(f)
    from lawaf.interfaces.phonopy.lwf import LWF

    lwf2 = LWF.load_from_netcdf(f)
    assert lwf2.SwannR is not None
    assert np.abs(lwf2.SwannR - lwf.SwannR).max() < 1e-10
    q = np.array([0.25, 0.11, -0.37])
    assert np.abs(lwf2.solve_k(q)[0] - lwf.solve_k(q)[0]).max() < 1e-8


# --------------------------------------------- constant GL gauge transform
class OrthoTB:
    """Genuinely orthogonal two-band parent (orthonormal psi)."""

    is_orthogonal = True

    def __init__(self, seed=31):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        self.H0 = a + a.conj().T
        b = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        self.hop = 0.3 * (b + b.conj().T)
        self._r = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
        self.atoms = Atoms("H2", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3))

    def _Hk(self, k):
        H = self.H0.copy()
        for R in self._r[1:]:
            ph = np.exp(2j * np.pi * np.dot(R, k))
            H += self.hop * ph + self.hop.conj().T * ph.conj()
        return H

    def solve_all(self, kpts):
        nk = len(kpts)
        e = np.zeros((nk, 2))
        v = np.zeros((nk, 2, 2), dtype=complex)
        for ik, k in enumerate(kpts):
            e[ik], v[ik] = eigh(self._Hk(k))
        return e, v


def _G(seed=4, n=2, scale=0.2):
    rng = np.random.default_rng(seed)
    return np.eye(n) + scale * (
        rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    )


def test_gauge_transform_with_mlwf():
    """constant G on top of the MV-optimized orthonormal gauge: onsite
    overlap G^dag G delta_R0 and pencil-exact bands at every k (ws off)."""
    m = OrthoTB()
    G = _G()
    res = {}
    for tag, extra in [("ctl", {}), ("mlwfG", dict(nonorthogonal_gauge=G))]:
        df = Lawaf(
            m,
            params=dict(
                method="mlwf", kmesh=(3, 3, 3), nwann=2, selected_basis=[0, 1],
                weight_func="unity", mlwf_max_iter=50, use_ws_distance=False,
                **extra,
            ),
        )
        res[tag] = (df, df.downfold())
    lwf = res["mlwfG"][1]
    assert not lwf.is_orthogonal
    assert lwf.SwannR is not None
    assert res["ctl"][1].SwannR is None
    i0 = int(np.where(np.all(lwf.Rlist == 0, axis=1))[0][0])
    others = np.delete(lwf.SwannR, i0, axis=0)
    assert np.abs(others).max() < 1e-12  # onsite-only overlap
    assert np.linalg.norm(lwf.SwannR[i0] - G.conj().T @ G) < 1e-12
    rng = np.random.default_rng(12)
    q = rng.uniform(-0.5, 0.5, size=(5, 3))
    e_c = np.array([res["ctl"][1].solve_k(x)[0] for x in q])
    e_g = np.array([lwf.solve_k(x)[0] for x in q])
    assert np.abs(e_c - e_g).max() < 1e-12


def test_gauge_transform_with_window_selection(tmp_path):
    """The previously-refused composition (window_bands + non-orthogonal)
    works through a constant G: overlap G^dag G delta_R0, bands equal to
    the orthonormal control at arbitrary q (ws off -> exact everywhere)."""
    G = _G(seed=9, n=3, scale=0.25)
    base = dict(
        method="projected", nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)}, use_proj=True,
        weight_func="unity", kmesh=(2, 2, 2), gamma=True,
        use_ws_distance=False,
        window_bands={
            (0.0, 0.0, 0.0): (0, 1, 2),
            (0.5, 0.0, 0.0): (0, 1, 4),
            (0.5, 0.5, 0.0): (0, 1, 4),
            (0.5, 0.5, 0.5): (0, 1, 2),
        },
    )
    res = {}
    for tag, extra in [("ctl", {}), ("G", dict(nonorthogonal_gauge=G))]:
        df = PhonopyDownfolder(
            phonopy_yaml=str(FIXTURE), mode="DM", params=dict(base, **extra),
            symmetrize_fc=False, is_nac=False,
        )
        res[tag] = (df, df.downfold(output_path=str(tmp_path / tag),
                                    write_hr_nc=None, write_hr_txt=None))
    lwf = res["G"][1]
    assert lwf.SwannR is not None
    assert res["ctl"][1].SwannR is None
    i0 = int(np.where(np.all(lwf.Rlist == 0, axis=1))[0][0])
    assert np.abs(np.delete(lwf.SwannR, i0, axis=0)).max() < 1e-12
    assert np.linalg.norm(lwf.SwannR[i0] - G.conj().T @ G) < 1e-12
    rng = np.random.default_rng(13)
    q = rng.uniform(-0.5, 0.5, size=(5, 3))
    e_c = np.array([res["ctl"][1].solve_k(x)[0] for x in q])
    e_g = np.array([lwf.solve_k(x)[0] for x in q])
    assert np.abs(e_c - e_g).max() < 1e-10


def test_gauge_transform_shape_and_rank_refused(tmp_path):
    m = OrthoTB()
    with pytest.raises(ValueError, match="nonorthogonal_gauge must be"):
        Lawaf(
            m,
            params=dict(method="projected", kmesh=(3, 3, 3), nwann=2,
                        selected_basis=[0, 1], weight_func="unity",
                        use_ws_distance=False, nonorthogonal_gauge=np.eye(3)),
        ).downfold()
    with pytest.raises(ValueError, match="rank-deficient"):
        Lawaf(
            m,
            params=dict(
                method="projected", kmesh=(3, 3, 3), nwann=2,
                selected_basis=[0, 1], weight_func="unity",
                use_ws_distance=False,
                nonorthogonal_gauge=np.array([[1.0, 0.5], [0.5, 0.25]]),
            ),
        ).downfold()
