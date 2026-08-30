"""Tests for lawaf.interfaces.phonopy.symmetry_seeds — story-010 seed core.

Eigenspace mapping (eigenvalue-matched), whole-eigenspace + svmin guard with
per-anchor fallback, lawaf-gauge conversion, commensurability denominators,
lazy optional import.

Oracle evidence: research memo 2026-08-29-spgrep-modulation-projectors
(BaTiO3 DM_dip_wang, Pm-3m, Γ = 4×T1u + T2u; gauge law v_lawaf(κ) =
e^{+2πi r_κ·q} v_spgrep(κ) verified at q=(½,0,¼)).

Dependency-free contracts (denominators, warning filter, missing-dependency
error) run without the ``symmetry`` extra; only the phonon-backed tests skip
when spgrep-modulation is absent (review STD-001).
"""

import builtins
import importlib
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.interfaces.phonopy import symmetry_seeds as ss  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = np.zeros(3)
NONTRIM_Q = np.array([0.5, 0.0, 0.25])  # denominators (2, 1, 4)


@pytest.fixture(scope="module")
def phonon():
    pytest.importorskip("spgrep_modulation")
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def model(phonon):
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    return PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)


# ---------------------------------------------------------------- TEST-001
def test_gamma_soft_block_svmin(phonon, model):
    """Anchor (0,1,2) maps to the soft-mode eigenspace, lawaf gauge, svmin≈1."""
    rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert not rep.fell_back
    assert rep.warnings == []
    _evals, evecs = model.solve(GAMMA)
    converted = rep.psi[:, 0:3]
    sv = np.linalg.svd(evecs[:, 0:3].conj().T @ converted, compute_uv=False)
    assert sv.min() >= 1 - 1e-6


# ---------------------------------------------------------------- TEST-002
def test_guard_mismatch_falls_back(phonon, monkeypatch):
    """Injected eigenspace/eigenvalue mismatch → fallback report + warning."""
    real_build = ss.build_modulation

    def poisoned(phonon_obj, qpoint, **kwargs):
        md = real_build(phonon_obj, qpoint, **kwargs)
        # swap eigenvectors between two eigenspaces while keeping eigenvalue
        # order → positional mapping lands on the wrong subspace → the svmin
        # guard must fire (sorting the eigenspaces cannot repair this)
        es = list(md.eigenspaces)
        (e0, v0, r0), (e1, v1, r1) = es[0], es[1]
        es[0] = (e0, v1, r0)
        es[1] = (e1, v0, r1)
        return SimpleNamespace(eigenspaces=es)

    monkeypatch.setattr(ss, "build_modulation", poisoned)
    with pytest.warns(UserWarning, match="subspace guard"):
        rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert rep.fell_back
    # fallback psi = the reference solve's raw eigenvectors (same subspace;
    # bitwise equality is impossible — lawaf's legacy degenerate-block
    # alignment is a stochastic optimizer, the very arbitrariness this
    # feature replaces)
    _evals, evecs = ss._reference_solve(phonon, GAMMA)
    sv = np.linalg.svd(
        evecs[:, 0:3].conj().T @ rep.psi[:, 0:3], compute_uv=False
    )
    assert sv.min() >= 1 - 1e-6


def test_guard_eigenvalue_mismatch_falls_back(phonon, monkeypatch):
    """Injected wrong eigenspace EIGENVALUE → eigenvalue matching fires."""
    real_build = ss.build_modulation

    def poisoned(phonon_obj, qpoint, **kwargs):
        md = real_build(phonon_obj, qpoint, **kwargs)
        es = list(md.eigenspaces)
        e, v, r = es[0]
        es[0] = (e + 0.5, v, r)  # same vectors, wrong eigenvalue
        return SimpleNamespace(eigenspaces=es)

    monkeypatch.setattr(ss, "build_modulation", poisoned)
    with pytest.warns(UserWarning, match="eigenvalue"):
        rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert rep.fell_back

def test_guard_overcomplete_dims_falls_back(phonon, monkeypatch):
    """Over-complete eigenspace decomposition → fallback, never IndexError."""
    real_build = ss.build_modulation

    def poisoned(phonon_obj, qpoint, **kwargs):
        md = real_build(phonon_obj, qpoint, **kwargs)
        es = list(md.eigenspaces)
        # inflate one eigenspace's dim claim beyond the solvable modes
        e, v, r = es[-1]
        es[-1] = (e, np.concatenate([v, v[:1]], axis=0), r)
        return SimpleNamespace(eigenspaces=es)


    monkeypatch.setattr(ss, "build_modulation", poisoned)
    with pytest.warns(UserWarning, match="incomplete eigenspace"):
        rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert rep.fell_back



def test_degeneracy_tol_none_is_valid(phonon):
    """Explicit degeneracy_tol=None (spgrep default grouping) must work."""
    rep = ss.get_symmetry_anchor_wfn(
        phonon, GAMMA, bands=(0, 1, 2), degeneracy_tol=None
    )
    assert not rep.fell_back


# ---------------------------------------------------------------- TEST-003
def test_partial_eigenspace_anchor_falls_back(phonon):
    """Anchor covering 2 of 3 soft-triplet bands → fallback + warning."""
    with pytest.warns(UserWarning, match="whole|part of eigenspace"):
        rep = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1))
    assert rep.fell_back
    assert np.allclose(rep.qpoint, GAMMA)
    assert any("cannot be symmetry-canonical" in w for w in rep.warnings)
    # raw-reference fallback for the requested bands
    _evals, evecs = ss._reference_solve(phonon, GAMMA)
    sv = np.linalg.svd(
        evecs[:, 0:2].conj().T @ rep.psi[:, 0:2], compute_uv=False
    )
    assert sv.min() >= 1 - 1e-6


# ---------------------------------------------------------------- TEST-004
def test_nontrim_gauge_and_sign(phonon, model):
    """At q=(½,0,¼): +sign projector residual ≤1e-10; −sign must FAIL."""
    md = ss.build_modulation(phonon, NONTRIM_Q)
    evals, evecs = model.solve(NONTRIM_Q)
    spos = phonon.primitive.scaled_positions
    worst = {+1: 0.0, -1: 0.0}
    for eigval, eigvecs, _irrep in md.eigenspaces:
        dim = eigvecs.shape[0]
        v = eigvecs.reshape(dim, -1).conj().T  # (3N, dim) atom-major
        m = v @ v.conj().T
        j = int(np.argmin(np.abs(evals - eigval)))
        block = evecs[:, j : j + dim]
        pl = block @ block.conj().T
        for sign in (+1, -1):
            ph = np.repeat(np.exp(sign * 2j * np.pi * spos @ NONTRIM_Q), 3)
            pg = (ph[:, None] * m) * ph.conj()[None, :]
            worst[sign] = max(worst[sign], float(np.abs(pl - pg).max()))
    assert worst[+1] <= 1e-10, f"gauge law violated: {worst[+1]:.2e}"
    assert worst[-1] > 1e-3, "flipped sign did not fail (test is vacuous)"

    # end-to-end: all bands at the non-TRIM anchor pass the guard
    n_bands = evecs.shape[1]
    rep = ss.get_symmetry_anchor_wfn(
        phonon, NONTRIM_Q, bands=tuple(range(n_bands))
    )
    assert not rep.fell_back
    sv = np.linalg.svd(evecs.conj().T @ rep.psi, compute_uv=False)
    assert sv.min() >= 1 - 1e-6


# ---------------------------------------------------------------- TEST-005
def test_noncommensurate_q_falls_back(phonon):
    """q with no denominator ≤ 12 → fallback + warning naming the q."""
    q = np.array([1.0 / 13.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="commensurate"):
        rep = ss.get_symmetry_anchor_wfn(phonon, q, bands=(0, 1, 2))
    assert rep.fell_back
    assert np.allclose(rep.qpoint, q)
    assert any("0.076923" in w for w in rep.warnings)


# ---------------------------------------------------------------- TEST-006
def test_missing_dependency_message(monkeypatch):
    """Missing spgrep-modulation → RuntimeError naming the extra, chained.

    Dependency-free: no phonon object needed (STD-001).
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("spgrep_modulation"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as excinfo:
        ss._import_modulation()
    assert "lawaf[symmetry]" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


# ---------------------------------------------------------------- TEST-007
def test_poisoned_phon_cache_is_irrelevant(phonon, tmp_path, monkeypatch):
    """A stale ./phon_cache pickle cannot corrupt the guard reference.

    Reference eigenvalues match to 1e-12 (fresh DM re-symmetrization drifts
    at machine epsilon; the legacy stochastic aligner only rotates within
    degenerate eigenvector blocks, so psi is compared as its subspace).
    """
    evals_clean, evecs_clean = ss._reference_solve(phonon, GAMMA)
    rep_clean = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert not rep_clean.fell_back
    poisoned_dir = tmp_path / "phon_cache"
    poisoned_dir.mkdir()
    (poisoned_dir / "cache.pickle").write_bytes(b"garbage-not-a-pickle")
    monkeypatch.chdir(tmp_path)
    evals_poisoned, _ = ss._reference_solve(phonon, GAMMA)
    rep_poisoned = ss.get_symmetry_anchor_wfn(phonon, GAMMA, bands=(0, 1, 2))
    assert not rep_poisoned.fell_back
    # reference unchanged: fresh DM re-symmetrization drifts at machine
    # epsilon (observed 8e-16), so equality to 1e-12 is exact for practical
    # purposes — any stale cache hit from a different model would differ
    # by orders of magnitude
    np.testing.assert_allclose(
        evals_clean, evals_poisoned, rtol=0, atol=1e-12
    )
    sv = np.linalg.svd(
        evecs_clean[:, 0:3].conj().T @ rep_poisoned.psi[:, 0:3],
        compute_uv=False,
    )
    assert sv.min() >= 1 - 1e-6


# ------------------------------------------------- NFR-003 warning filter
def test_benign_spgrep_warning_filtered():
    """Module import installs a *narrowed* ignore-filter; others propagate."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        importlib.reload(ss)
        assert any(
            f[0] == "ignore"
            and f[3] is not None
            and "spgrep_modulation" in f[3].pattern
            and f[1] is not None
            and "Inconsistent eigenvalue" in f[1].pattern
            and f[2] is UserWarning
            for f in warnings.filters
        )
        # an unrelated warning from the same module must still propagate
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn_explicit(
                "some other spgrep issue", UserWarning, "f.py", 1,
                module="spgrep_modulation.xyz",
            )
        assert len(caught) == 1


# ------------------------------------------------------- denominators unit
@pytest.mark.parametrize(
    ("q", "expected"),
    [
        (np.zeros(3), np.array([1, 1, 1])),
        (np.array([0.5, 0.0, 0.25]), np.array([2, 1, 4])),
        (np.array([1 / 3, 1 / 3, 1 / 2]), np.array([3, 3, 2])),
    ],
)
def test_smallest_denominators(q, expected):
    assert np.array_equal(ss.smallest_denominators(q), expected)


def test_smallest_denominators_none():
    assert ss.smallest_denominators(np.array([1 / 13, 0, 0])) is None
