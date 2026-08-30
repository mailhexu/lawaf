"""Story-030 tests: window resolution into per-k feasible/frozen sets.

ADR-003: two guidance vocabularies, one resolved per-k structure —
energy intervals (``dis_win_*`` outer / ``dis_froz_*`` inner) plus
validated star-expanded per-q ``window_bands`` pins (pin > inner >
outer). Band indices in pins are ORIGINAL indices, mapped through the
retained rows exactly like ``_apply_window_band_weights``.
"""

import inspect

import numpy as np
import pytest

from lawaf.params import WannierParams
from lawaf.wannierization.disentangle import (
    InfeasibleWindowError,
    resolve_windows,
)


class _FakeWindowBands:
    """Duck-typed stand-in for the validated WindowBands object."""

    def __init__(self, bands):
        self.bands = bands


def mesh8():
    """2x2x2 Gamma-centred mesh."""
    return np.array(
        [[i / 2, j / 2, k / 2] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    )


def eig8(nband=4):
    """Deterministic eigvals: well-separated bands, no boundary bleed."""
    return np.tile(np.array([-5.0, -3.0, 1.0, 4.0]), (8, 1))


class TestIntervalResolution:
    def test_outer_and_inner(self):
        kpts, ev = mesh8(), eig8()
        feasible, frozen = resolve_windows(
            kpts, ev, nwann=2, win_min=-5.5, win_max=1.5,
            froz_min=-5.5, froz_max=-3.5,
        )
        for ik in range(8):
            assert set(frozen[ik]) == {0}
            assert set(feasible[ik]) == {0, 1, 2}  # -5.5..1.5 keeps bands 0-2

    def test_no_inner_window(self):
        feasible, frozen = resolve_windows(mesh8(), eig8(), nwann=2,
                                           win_min=-5.5, win_max=1.5)
        assert all(f == () for f in frozen)
        assert set(feasible[0]) == {0, 1, 2}

    def test_no_sources_all_bands_feasible(self):
        feasible, frozen = resolve_windows(mesh8(), eig8(), nwann=2)
        assert all(set(f) == {0, 1, 2, 3} for f in feasible)
        assert all(f == () for f in frozen)


class TestPinOverride:
    def test_pin_freezes_and_pins_feasible(self):
        kpts, ev = mesh8(), eig8()
        wb = _FakeWindowBands({(0.0, 0.0, 0.0): (0, 1, 3)})
        feasible, frozen = resolve_windows(
            kpts, ev, nwann=3, window_bands=wb,
            win_min=-5.5, win_max=2.0,
        )
        ig = 0  # mesh8()[0] is Gamma
        assert set(frozen[ig]) == {0, 1, 3}
        assert set(feasible[ig]) == {0, 1, 3}
        # unpinned k follow the interval
        assert set(feasible[1]) == {0, 1, 2}
        assert frozen[1] == ()

    def test_pin_original_indices_mapped_through_ibands(self):
        kpts, ev = mesh8(), eig8()
        wb = _FakeWindowBands({(0.0, 0.0, 0.0): (1, 2, 3)})
        feasible, frozen = resolve_windows(
            kpts, ev, nwann=3, window_bands=wb, ibands=(1, 2, 3)
        )
        # original 1,2,3 -> retained rows 0,1,2
        assert set(frozen[0]) == {0, 1, 2}

    def test_pin_excluded_band_rejected(self):
        kpts, ev = mesh8(), eig8()
        wb = _FakeWindowBands({(0.0, 0.0, 0.0): (0, 1, 3)})
        with pytest.raises(ValueError, match="excluded"):
            resolve_windows(kpts, ev, nwann=3, window_bands=wb,
                            ibands=(1, 2, 3))

    def test_pin_size_must_match_nwann(self):
        kpts, ev = mesh8(), eig8()
        wb = _FakeWindowBands({(0.0, 0.0, 0.0): (0, 1)})
        with pytest.raises(ValueError, match="nwann"):
            resolve_windows(kpts, ev, nwann=3, window_bands=wb)

    def test_pin_off_mesh_rejected(self):
        kpts, ev = mesh8(), eig8()
        wb = _FakeWindowBands({(0.3, 0.0, 0.0): (0, 1, 2)})
        with pytest.raises(ValueError, match="mesh"):
            resolve_windows(kpts, ev, nwann=3, window_bands=wb)


class TestInfeasible:
    def test_inner_window_larger_than_nwann(self):
        with pytest.raises(InfeasibleWindowError, match="freezes"):
            resolve_windows(mesh8(), eig8(), nwann=2,
                            win_min=-5.5, win_max=5.0,
                            froz_min=-5.5, froz_max=1.5)

    def test_outer_too_tight(self):
        with pytest.raises(InfeasibleWindowError, match="feasible"):
            resolve_windows(mesh8(), eig8(), nwann=3,
                            win_min=-5.5, win_max=-2.5)


class TestParams:
    def test_dis_fields_declared_no_warning(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = WannierParams(
                dis_win_min=-210.0, dis_win_max=150.0,
                dis_froz_min=-210.0, dis_froz_max=-120.0,
                dis_mix_ratio=0.5, dis_max_iter=100,
                dis_tol=1e-10, dis_min_svd=1e-8,
            )
        assert p.dis_mix_ratio == 0.5
        assert p.dis_min_svd == 1e-8

    def test_typo_still_warns(self):
        with pytest.warns(UserWarning, match="dis_win_minx"):
            WannierParams(dis_win_minx=1.0)

    def test_defaults(self):
        p = WannierParams()
        assert p.dis_win_min is None and p.dis_froz_max is None
        assert p.dis_max_iter == 100

    def test_downfolder_signature_plumbs_dis_params(self):
        from lawaf.interfaces.downfolder import Lawaf as DownfolderManager

        sig = inspect.signature(DownfolderManager.set_parameters)
        for name in ("dis_win_min", "dis_win_max", "dis_froz_min",
                     "dis_froz_max", "dis_mix_ratio", "dis_max_iter",
                     "dis_tol", "dis_min_svd",
                     "dis_slow_tail_change"):
            assert name in sig.parameters, name
