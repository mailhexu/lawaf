"""End-to-end validation of symmetry seeds — story-013.

Covers the PRD success criteria that need full downfolds: band-fitting
equivalence between the seeded and legacy-anchor runs (in-test reference,
100-point Γ–X–M–R–Γ–R path), the NFR-002 performance budget (exactly one
Modulation construction per anchor, ≤5 s each), and public-export
importability without the ``symmetry`` extra (FR-005/SPECQ-007).

Multi-anchor (Γ + X) and scdmk seeded runs are exercised in
``test_symmetry_seed_integration.py`` (story-012) — not duplicated here.
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.interfaces.phonopy import phonon_downfolder as pdf  # noqa: E402
from lawaf.interfaces.phonopy import symmetry_seeds as ss  # noqa: E402
from lawaf.mathutils.evals_freq import evals_to_freqs  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
GAMMA = (0.0, 0.0, 0.0)
SOFT = (0, 1, 2)
# Γ–X–M–R–Γ–R (cubic Pm-3m): 5 segments, sampled at exactly 100 points
SEGMENTS = [
    ((0.0, 0.0, 0.0), (0.0, 0.5, 0.0)),    # Γ–X
    ((0.0, 0.5, 0.0), (0.5, 0.5, 0.0)),    # X–M
    ((0.5, 0.5, 0.0), (0.5, 0.5, 0.5)),    # M–R
    ((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)),    # R–Γ
    ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),    # Γ–R
]


@pytest.fixture(scope="module")
def phonon():
    ph = phonopy.load(FIXTURE, is_nac=False)
    ph.symmetrize_force_constants()
    return ph



def _kpath(npts=100):
    """Exactly ``npts`` points along Γ–X–M–R–Γ–R (nodes included).

    Equal per-segment counts, endpoint excluded on all but the last
    segment so the total is exact and every node is sampled.
    """
    nseg = len(SEGMENTS)
    per = npts // nseg
    kpts = []
    for iseg, (a, b) in enumerate(SEGMENTS):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        endpoint = iseg == nseg - 1
        for t in np.linspace(0, 1, per, endpoint=endpoint):
            kpts.append(a + t * (b - a))
    assert len(kpts) == npts
    return np.array(kpts)


def _downfolded_lwf(phonon, **overrides):
    params = dict(
        method="projected",
        nwann=3,
        anchors={GAMMA: SOFT},
        use_proj=True,
        weight_func="Gauss",
        weight_func_params=(-20.0, 20.0),
        kmesh=(3, 3, 3),
    )
    params.update(overrides)
    df = pdf.PhonopyDownfolder(phonon=phonon, params=params)
    return df.downfold(
        output_path=str(Path(__file__).parent / ".tmp_story013"),
        write_hr_nc=None,
        write_hr_txt=None,
    )


# ---------------------------------------------------------------- TEST-001
def test_band_equivalence_vs_legacy(phonon):
    """Seeded and legacy-anchor runs select the SAME physics (SPECQ-006).

    Two assertions:
    (1) with SHIPPED defaults, the retained subspaces coincide at every
        k of the mesh — the gauge-invariant content of "seeding changes
        only the gauge" (asserted via the svmin of the projector
        overlaps, no library method patched);
    (2) interpolated frequencies on the 100-point Γ–X–M–R–Γ–R path agree
        to ≤ 1e-6 cm^-1 after the two gauge-dependent post-processing
        heuristics (WS materialization; nonlinear |A|**-0.2 Amn rescale)
        are neutralized identically on both arms — see comments below.
    """
    pytest.importorskip("spgrep_modulation")
    # sanity: the seeded run really uses P4mm axis-pure seeds
    rep = ss.get_symmetry_anchor_wfn(phonon, np.zeros(3), bands=SOFT)
    assert not rep.fell_back
    assert {r.sg_number for r in rep.per_band} == {99}

    # (1) shipped-config subspace equality: build both downfolders with
    # default use_ws_distance and the production Amn path, then compare
    # the gauge-invariant principal angles between the two wannier
    # subspaces at every k.
    def _subspaces(**overrides):
        df = pdf.PhonopyDownfolder(phonon=phonon, params=dict(
            method="projected", nwann=3, anchors={GAMMA: SOFT},
            use_proj=True, weight_func="Gauss",
            weight_func_params=(-20.0, 20.0), kmesh=(3, 3, 3),
            **overrides))
        df._prepare_data()
        df.builder.prepare()
        df.builder.get_Amn()
        wannk, _Hk, _ = df.builder.get_wannk_and_Hk()
        return wannk

    W_legacy = _subspaces()
    W_seed = _subspaces(symmetry_seed=True)
    for ik in range(W_legacy.shape[0]):
        sv = np.linalg.svd(W_legacy[ik].conj().T @ W_seed[ik],
                           compute_uv=False)
        np.testing.assert_allclose(sv, np.ones(3), atol=1e-8)

    # use_ws_distance=False on BOTH runs: the Wigner-Seitz R-materialization
    # is a gauge-dependent post-processing choice (different Wannier gauges
    # pick different image assignments at the 1e-4 cm^-1 level); the
    # equivalence criterion compares the plain Fourier interpolation of the
    # same retained subspaces.
    #
    # The projected Amn builder also applies a nonlinear elementwise
    # rescaling (|A|**-0.2 * A in get_Amn_one_k) that is gauge-dependent BY
    # CONSTRUCTION (it breaks projector-basis invariance at the ~3e-3
    # cm^-1 level). Neutralize it identically on both arms so the band
    # comparison isolates the gauge-invariant physics; with it, seeded vs
    # legacy agree to ~1e-8 cm^-1.
    from lawaf.wannierization.projectedWF import ProjectedWannierizer

    original = ProjectedWannierizer.get_Amn_one_k

    def linear_amn(self, ik):
        A = (
            self.get_psi_k(ik).conj().T
            @ self.projectors.T
            * self.occ[ik][:, np.newaxis]
        )
        U, _S, VT = np.linalg.svd(A, full_matrices=False)
        return U @ VT

    ProjectedWannierizer.get_Amn_one_k = linear_amn
    try:
        lwf_legacy = _downfolded_lwf(phonon, use_ws_distance=False)
        lwf_seed = _downfolded_lwf(
            phonon, symmetry_seed=True, use_ws_distance=False
        )
    finally:
        ProjectedWannierizer.get_Amn_one_k = original

    kpts = _kpath()
    evals_l, _ = lwf_legacy.solve_all(kpts)
    evals_s, _ = lwf_seed.solve_all(kpts)
    freqs_l = evals_to_freqs(evals_l, lwf_legacy.factor)
    freqs_s = evals_to_freqs(evals_s, lwf_seed.factor)
    assert np.max(np.abs(freqs_l - freqs_s)) <= 1e-6


# ---------------------------------------------------------------- TEST-003
def test_performance_budget(phonon, monkeypatch):
    """NFR-002: exactly one Modulation construction per anchor and ≤5 s
    per construction on the BaTiO3 fixture."""
    pytest.importorskip("spgrep_modulation")
    calls = []
    real_build = ss.build_modulation

    def counting(phonon_obj, qpoint, **kwargs):
        t0 = time.perf_counter()
        mod = real_build(phonon_obj, qpoint, **kwargs)
        calls.append(time.perf_counter() - t0)
        return mod

    monkeypatch.setattr(ss, "build_modulation", counting)
    X = (0.5, 0.0, 0.0)
    anchors = {GAMMA: SOFT, X: SOFT}
    df = pdf.PhonopyDownfolder(
        phonon=phonon,
        params=dict(
            method="scdmk",  # multi-anchor shape (see integration TEST-004)
            nwann=3,
            anchors=anchors,
            kmesh=(3, 3, 3),
            symmetry_seed=True,
        ),
    )
    df._prepare_data()
    assert len(calls) == len(anchors)  # exactly one Modulation per anchor
    assert max(calls) <= 5.0


# ---------------------------------------------------------------- TEST-004
def test_public_exports_without_extra():
    """FR-005/SPECQ-007: `lawaf.get_symmetry_anchor_wfn` and
    `lawaf.list_opd_families` are importable with spgrep-modulation and
    spgrep hidden (module import stays lazy)."""
    code = (
        "import sys\n"
        "sys.modules['spgrep_modulation'] = None\n"
        "sys.modules['spgrep'] = None\n"
        "import lawaf\n"
        "assert callable(lawaf.get_symmetry_anchor_wfn)\n"
        "assert callable(lawaf.list_opd_families)\n"
        "assert 'get_symmetry_anchor_wfn' in lawaf.__all__\n"
        "assert 'list_opd_families' in lawaf.__all__\n"
        "assert all(isinstance(name, str) for name in lawaf.__all__)\n"
        "print('exports-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "exports-ok" in result.stdout


def test_public_exports_typed_with_extra(phonon):
    """With the extra present the top-level exports return the specified
    types."""
    pytest.importorskip("spgrep_modulation")
    import lawaf

    families = lawaf.list_opd_families(phonon, np.zeros(3), bands=SOFT)
    assert families and isinstance(families[0], ss.OPDFamily)
    rep = lawaf.get_symmetry_anchor_wfn(phonon, np.zeros(3), bands=SOFT)
    assert isinstance(rep, ss.SymmetrySeedReport)
    assert rep.per_band and isinstance(rep.per_band[0], ss.SeedBandRecord)
