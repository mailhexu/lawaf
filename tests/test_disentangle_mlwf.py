"""Story-031 tests: MLWF psi_sel integration seam (ADR-004).

Synthetic 2x2x2 models exercise the guard and the seam mechanics; the
BaTiO3 DM fixture (phonopy, no MACE) carries the PRD acceptance runs:
2x2x2 reproduction against the exclude_bands path, and the amended 4x4x4
acceptance (monotone MV, healthy gauged svd, no collapse).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from lawaf.params import WannierParams
from lawaf.wannierization.mlwf import MLWFWannierizer

REPO = Path(__file__).resolve().parents[1]
FIXTURE = (
    REPO / "example" / "Phonopy" / "BaTiO3" / "DM_dip_wang"
    / "phonopy_params.yaml"
)


def mesh(n):
    """n x n x n Gamma-centred mesh + unit weights."""
    k = np.array(
        [[i / n, j / n, l / n] for i in range(n) for j in range(n)
         for l in range(n)]
    )
    return k, np.full(len(k), 1.0 / len(k))


def synthetic(n=2, nband=4):
    """Phase-only evecs (perfect transport) + descending evals."""
    kpts, kw = mesh(n)
    nk = len(kpts)
    evals = np.tile(np.array([-5.0, -3.0, 1.0, 4.0]), (nk, 1))
    phase = 2 * np.pi * kpts @ np.array([0.1, 0.2, 0.3])
    evecs = np.zeros((nk, nband, nband), dtype=complex)
    for ik in range(nk):
        evecs[ik] = np.diag(np.exp(1j * phase[ik] * np.arange(1, nband + 1)))
    return evals, evecs, kpts, kw


def run_mlwf_synthetic(params):
    evals, evecs, kpts, kw = synthetic()
    w = MLWFWannierizer(evals, evecs, kpts, kw, params)
    w.prepare()
    w.get_Amn()
    return w


class TestSeamGuards:
    def test_no_guidance_named_error(self):
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
        )
        with pytest.raises(ValueError, match="dis_win_min|window_bands"):
            run_mlwf_synthetic(params)

    def test_selection_engages_with_intervals(self):
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
            dis_win_min=-5.5, dis_win_max=1.5,
            dis_froz_min=-5.5, dis_froz_max=-3.5,
        )
        w = run_mlwf_synthetic(params)
        assert w.selection is not None
        assert w.selection.U_sel.shape == (8, 4, 2)
        assert w.selection.pair_svd_min.min() > 1 - 1e-8
        assert np.isfinite(w.spreads["omega"])
        # Amn stays a full-manifold (nband, nwann) object for downstream
        assert w.Amn.shape == (8, 4, 2)

    def test_nband_eq_nwann_untouched(self):
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=4,
            mlwf_initial_guess="identity",
        )
        w = run_mlwf_synthetic(params)
        assert w.selection is None  # fast path: no selection stage
        assert np.isfinite(w.spreads["omega"])

    def test_frozen_only_guidance_rejected(self):
        """Frozen bounds alone leave the free complement unrestricted
        (F3 drift): the guard must reject them (review R10)."""
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
            dis_froz_min=-5.5, dis_froz_max=-3.5,
        )
        with pytest.raises(ValueError, match="dis_win_min|window_bands"):
            run_mlwf_synthetic(params)

    def test_noop_outer_window_rejected(self):
        """Bounds spanning the whole spectrum restrict nothing: the
        resolved feasible sets equal the full manifold everywhere and
        the guard must reject the run (review R11)."""
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
            dis_win_min=-100.0, dis_win_max=100.0,
        )
        with pytest.raises(ValueError, match="span the whole"):
            run_mlwf_synthetic(params)

    def test_selection_uses_retained_rows_with_exclude_bands(self):
        """Regression (review R3): exclude_bands removes rows from psi
        but self.evals keeps the full spectrum; windows must see the
        retained rows (retained index i maps to ibands[i])."""
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
            exclude_bands=(3,),  # drop the 4.0 band -> retained [-5,-3,1]
            # window bounds compare against RETAINED rows: feasible
            # {0,1} = original bands {-5,-3}
            dis_win_min=-5.5, dis_win_max=0.0,
            dis_froz_min=-5.5, dis_froz_max=-3.5,
        )
        w = run_mlwf_synthetic(params)
        assert w.selection is not None
        assert w.selection.U_sel.shape == (8, 3, 2)  # retained rows
        # frozen row 0 = original band 0 (-5); free row 1 = original -3
        for ik in range(8):
            support = np.abs(w.selection.U_sel[ik]).argmax(axis=0)
            assert set(support) == {0, 1}
        assert np.isfinite(w.spreads["omega"])

    def test_guidance_records_sources(self):
        params = WannierParams(
            method="mlwf", kmesh=(2, 2, 2), gamma=True, nwann=2,
            mlwf_initial_guess="identity",
            dis_win_min=-5.5, dis_win_max=1.5,
            dis_froz_min=-5.5, dis_froz_max=-3.5,
        )
        w = run_mlwf_synthetic(params)
        src = w.selection.guidance["sources"]
        assert src["outer_window"] == (-5.5, 1.5)
        assert src["inner_window"] == (-5.5, -3.5)
        assert src["window_bands"] is False


# ---------------------------------------------------------------- BaTiO3


def _bato3_phonon():
    import phonopy

    phonon = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    phonon.symmetrize_force_constants()
    return phonon


def _bato3_builder(kmesh, **extra):
    from lawaf.interfaces.phonopy import phonon_downfolder as pdf

    params = dict(
        method="mlwf", nwann=3,
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True, weight_func="unity",
        kmesh=tuple(kmesh), gamma=True,
        mlwf_max_iter=500,
    )
    params.update(extra)
    phonon = _bato3_phonon()
    return phonon, params, pdf


def _bato3_Hwannk(kmesh, **extra):
    """Run the BaTiO3 downfolder and return (builder, Hwannk)."""
    from lawaf.interfaces.phonopy import phonon_downfolder as pdf

    phonon, params, _pdf = _bato3_builder(kmesh, **extra)
    df = _pdf.PhonopyDownfolder(phonon=phonon, params=params)
    df._prepare_data()
    df.atoms = df.model.atoms
    df.builder.prepare()
    df.builder.get_Amn()
    _wannk, Hwannk, _ = df.builder.get_wannk_and_Hk()
    return df, df.builder, np.asarray(Hwannk)


def _win_ev(nu_cm1, factor=524.16):
    """cm^-1 -> builder-evals units (freqs_to_evals, lawaf convention)."""
    return np.sign(nu_cm1 / factor) * (nu_cm1 / factor) ** 2


@pytest.mark.slow
@pytest.mark.skipif(not FIXTURE.exists(), reason="BaTiO3 fixture not found")
class TestBato3Acceptance:
    def test_2x2x2_selection_matches_exclude_bands_path(self):
        """PRD success 2: energy windows on the coarse mesh; on-mesh
        eigenvalues of the selected subspace match the exclude_bands
        (lowest-3) window path at every mesh point, Omega comparable
        to the committed 21.46 crystal units.

        Note (story log): the story-AC pin variant {X:125, M:125} is
        not achievable on this fixture — M{0,1,4} straddles the Eu pair
        at 3.008 THz (story-025 checker) and the legal M{0,1,2} bundle
        is provably discontinuous against X{0,1,4} (its third
        direction lives in X's excluded 111-pair). The energy-window
        selection reduces to the same lowest-3 content, which is the
        PRD's actual comparison.
        """
        _df, _b, href = _bato3_Hwannk(
            (2, 2, 2), exclude_bands=tuple(range(3, 15)))
        # dis_min_svd=0 for this fixture only: the windowed lowest-3
        # content is INTRINSICALLY exactly singular at Gamma<->X in the
        # displacement metric (Gamma soft-triplet member has exactly
        # zero overlap with X bands {0,1,2}; its continuation lives at
        # 580/718 cm-1, far outside the window). The committed
        # exclude_bands production content is the same singular family
        # (Omega_I 58.2, memo F2); the svd guard is a general-user
        # default, not an assertion about this fixture.
        _df, b, hsel = _bato3_Hwannk(
            (2, 2, 2),
            dis_win_min=_win_ev(-210), dis_win_max=_win_ev(150),
            dis_froz_min=_win_ev(-210), dis_froz_max=_win_ev(-120),
            dis_min_svd=0.0,
        )
        assert b.selection is not None
        for ik in range(len(hsel)):
            er = np.sort(np.linalg.eigvalsh(href[ik]))
            es = np.sort(np.linalg.eigvalsh(hsel[ik]))
            scale = max(1.0, float(np.abs(er).max()))
            assert np.abs(er - es).max() < 1e-8 * scale, f"k={ik}"
        # Omega comparable to the committed 2x2x2 window value 21.46
        assert 10.0 < b.spreads["omega"] < 35.0

    def test_4x4x4_selection_mlwf_healthy(self):
        """Amended PRD success 1: no collapse, monotone MV, termination."""
        # dis_min_svd=0 for this fixture: like the 2x2x2 case, the
        # windowed content carries structural exact zeros on X<->M arm
        # pairs (measured 3.9e-18 at k=60<->61); the same class the
        # production exclude_bands content belongs to. Raw-content svd
        # is not the health metric here -- no collapse, monotone MV and
        # self-termination are (amended PRD success 1); see the
        # selection diagnostics in the 032 e2e test.
        _df, b, hsel = _bato3_Hwannk(
            (4, 4, 4),
            dis_win_min=_win_ev(-210), dis_win_max=_win_ev(150),
            dis_froz_min=_win_ev(-210), dis_froz_max=_win_ev(-120),
            dis_min_svd=0.0,
        )
        sel = b.selection
        assert sel is not None
        # windows are near-fixed points; a slow descent tail accepted
        # at budget is healthy (amended PRD success 1)
        assert sel.n_iter <= b.params.dis_max_iter
        assert sel.subspace_change_trace[-1] < 1e-4
        assert sel.pair_svd_mean.min() > 0.3  # prototype-class content
        omegas = [h["omega"] for h in b.mlwf_history]
        assert all(
            omegas[i + 1] <= omegas[i] + 1e-12 for i in range(len(omegas) - 1)
        )
        assert len(omegas) < b.params.mlwf_max_iter  # terminated by itself
        assert np.isfinite(b.spreads["omega"])
        # the model carries nwann=3 on-mesh branches
        assert hsel.shape[-1] == 3
