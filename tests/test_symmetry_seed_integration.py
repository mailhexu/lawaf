"""Params + downfolder wfn_anchor integration tests — story-012.

TDD: written before implementation. Covers the story-012 acceptance
criteria: WannierParams symmetry_seed fields + set_parameters passthrough,
post-construction ``builder.wfn_anchor`` injection for every anchor q,
override forwarding (opd=160 → R3m), printed seed metadata, the scdmk
tuple-key consumption, and the symmetry_seed=False default leaving the
legacy path untouched.
"""

from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.interfaces.phonopy import phonon_downfolder as pdf  # noqa: E402
from lawaf.interfaces.phonopy import symmetry_seeds as ss  # noqa: E402
from lawaf.params import WannierParams  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = (0.0, 0.0, 0.0)
X = (0.5, 0.0, 0.0)
SOFT = (0, 1, 2)

BASE_PARAMS = dict(
    method="projected",
    nwann=3,
    anchors={GAMMA: SOFT},
    use_proj=True,
    weight_func="Gauss",
    weight_func_params=(-20.0, 20.0),  # wide window: all bands (cm^-1)
    kmesh=(3, 3, 3),
)


@pytest.fixture(scope="module")
def phonon():
    pytest.importorskip("spgrep_modulation")
    ph = phonopy.load(FIXTURE, is_nac=False)
    ph.symmetrize_force_constants()
    return ph


def _downfolder(phonon, **overrides):
    params = dict(BASE_PARAMS)
    params.update(overrides)
    return pdf.PhonopyDownfolder(phonon=phonon, params=params)


def _downfold(phonon, **overrides):
    """Run a downfold into a tmp dir; returns the downfolder."""
    df = _downfolder(phonon, **overrides)
    df.downfold(output_path=str(Path(__file__).parent / ".tmp_story012"),
                write_hr_nc=None, write_hr_txt=None)
    return df


# ---------------------------------------------------------------- TEST-001
def test_params_roundtrip():
    defaults = WannierParams()
    assert defaults.symmetry_seed is False
    assert defaults.symmetry_seed_opd is None
    assert defaults.symmetry_seed_opd_index is None

    explicit = WannierParams(
        symmetry_seed=True, symmetry_seed_opd=160, symmetry_seed_opd_index=2
    )
    assert explicit.symmetry_seed is True
    assert explicit.symmetry_seed_opd == 160
    assert explicit.symmetry_seed_opd_index == 2


def test_set_parameters_passthrough():
    df = pdf.PhonopyDownfolder.__new__(pdf.PhonopyDownfolder)
    # bypass __init__ (no phonon needed for parameter plumbing)
    df._params = {}
    df.model = type("M", (), {"is_orthogonal": True})()
    df.set_parameters(
        method="projected",
        symmetry_seed=True,
        symmetry_seed_opd=160,
        symmetry_seed_opd_index=7,
    )
    assert df.params.symmetry_seed is True
    assert df.params.symmetry_seed_opd == 160
    assert df.params.symmetry_seed_opd_index == 7


# ---------------------------------------------------------------- TEST-002
def test_projected_symmetry_seed_svmin_and_metadata(phonon, capsys):
    legacy = _downfold(phonon)
    seeded = _downfold(phonon, symmetry_seed=True)
    # wfn_anchor injected post-construction, tuple-keyed
    assert seeded.builder.wfn_anchor is not None
    assert tuple(GAMMA) in seeded.builder.wfn_anchor
    assert legacy.builder.wfn_anchor is None
    # retained subspace identical to the legacy run: per-k overlap svmin = 1
    # AND the projectors are the SEED columns (not the raw eigenvectors) —
    # non-vacuous proof that the seeds are consumed
    P = np.array(seeded.builder.projectors)
    seed = seeded.builder.wfn_anchor[tuple(GAMMA)]
    _, evecs_raw = ss._reference_solve(phonon, np.zeros(3))
    assert not np.allclose(P[0], evecs_raw[:, 0])
    np.testing.assert_allclose(P[0], seed[:, 0])
    np.testing.assert_allclose(P[1], seed[:, 1])
    np.testing.assert_allclose(P[2], seed[:, 2])
    ws, wl = seeded.builder.wannk, legacy.builder.wannk
    assert ws.shape == wl.shape
    svmin = min(
        np.linalg.svd(
            ws[ik].conj().T @ wl[ik], compute_uv=False
        ).min()
        for ik in range(ws.shape[0])
    )
    assert svmin > 1 - 1e-8
    # seed metadata printed with every SeedBandRecord field (FR-006)
    out = capsys.readouterr().out
    for token in (
        "symmetry seed",
        "eigenspace",
        "family",
        "P4mm",
        "#99",
        "axis-pure",
        "frequency",
    ):
        assert token in out, f"metadata missing {token!r}"


# ---------------------------------------------------------------- TEST-003
def test_scdmk_symmetry_seed_completes(phonon):
    df = _downfold(
        phonon,
        method="scdmk",
        symmetry_seed=True,
    )
    assert df.builder.wfn_anchor is not None
    assert tuple(GAMMA) in df.builder.wfn_anchor
    # non-vacuous: scdmk anchor columns come from the seed matrix
    seed = df.builder.wfn_anchor[tuple(GAMMA)]
    np.testing.assert_allclose(df.builder.psi_anchors[0], seed[:, 0])
    assert df.lwf.wannR.shape[-1] == 3  # three seed bands retained


def test_anchors_none_derived_and_written_back(phonon):
    """params.anchors=None derives {anchor_kpt: anchor_ibands} and writes
    it back so the set_params re-run consumes the seeds (review round 1)."""
    params = dict(BASE_PARAMS)
    del params["anchors"]
    df = pdf.PhonopyDownfolder(
        phonon=phonon,
        params={**params, "anchor_kpt": (0, 0, 0), "symmetry_seed": True},
    )
    df._prepare_data()
    assert df.params.anchors == {(0.0, 0.0, 0.0): (0, 1, 2)}
    seed = df.builder.wfn_anchor[(0.0, 0.0, 0.0)]
    P = np.array(df.builder.projectors)
    np.testing.assert_allclose(P[0], seed[:, 0])


# ---------------------------------------------------------------- TEST-004
def test_multi_anchor_populates_all_q(phonon):
    df = _downfolder(
        phonon,
        method="scdmk",
        anchors={GAMMA: SOFT, X: SOFT},
        symmetry_seed=True,
    )
    df._prepare_data()
    assert set(df.builder.wfn_anchor) == {tuple(GAMMA), tuple(X)}


# ---------------------------------------------------------------- TEST-005
def test_fallback_anchor_injects_raw_eigenvectors(phonon, monkeypatch):
    """A guard-failing anchor injects the report's raw eigenvectors
    (= legacy behavior for that anchor) with a warning."""
    rep = ss.get_symmetry_anchor_wfn(phonon, np.zeros(3), bands=SOFT)
    assert not rep.fell_back  # sanity: fixture path succeeds

    def failing(phonon_obj, qpoint, bands, **kwargs):
        q = np.asarray(qpoint, dtype=float)
        evals, evecs = ss._reference_solve(phonon_obj, q)
        return ss.SymmetrySeedReport(
            qpoint=tuple(q),
            psi=evecs.copy(),
            fell_back=True,
            warnings=["stubbed guard failure"],
        )

    monkeypatch.setattr(ss, "get_symmetry_anchor_wfn", failing)
    with pytest.warns(UserWarning, match="stubbed guard failure"):
        df = _downfolder(phonon, symmetry_seed=True)
        df._prepare_data()
    injected = df.builder.wfn_anchor[tuple(GAMMA)]
    evals_raw, evecs_raw = ss._reference_solve(phonon, np.zeros(3))
    np.testing.assert_array_equal(injected, evecs_raw)

# ---------------------------------------------------------------- TEST-006
def test_opd_override_forwarded_to_r3m(phonon, capsys):
    df = _downfolder(phonon, symmetry_seed=True, symmetry_seed_opd=160)
    df._prepare_data()
    out = capsys.readouterr().out
    assert "R3m" in out
    assert "#160" in out


def test_opd_index_override_forwarded(phonon, capsys):
    # family index 3 is the ndir==1 Amm2 line; assert it is reported verbatim
    df = _downfolder(phonon, symmetry_seed=True, symmetry_seed_opd_index=3)
    df._prepare_data()
    out = capsys.readouterr().out
    assert "Amm2" in out
    assert "#38" in out
    assert "family 3" in out


# ---------------------------------------------------------------- TEST-007
def test_symmetry_seed_default_off(phonon):
    df = _downfolder(phonon)  # no symmetry_seed key
    df._prepare_data()
    assert df.params.symmetry_seed is False
    assert df.builder.wfn_anchor is None
