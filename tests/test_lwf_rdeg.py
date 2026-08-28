"""Story 6: LWF Rdeg persistence in netcdf round-trips."""
import numpy as np
import pytest

from lawaf.lwf.lwf import LWF


def make_lwf(Rdeg=None):
    rng = np.random.default_rng(7)
    Rlist = np.array([[i, j, k] for i in range(-1, 2)
                      for j in range(-1, 2) for k in range(-1, 2)], dtype=int)
    nR, nwann, nbasis = len(Rlist), 2, 6
    H = rng.random((nR, nwann, nwann)) + 1j * rng.random((nR, nwann, nwann))
    H = 0.5 * (H + H.conj().transpose(0, 2, 1))
    wannR = rng.random((nR, nbasis, nwann))
    if Rdeg == "random":
        Rdeg = rng.uniform(0.5, 1.0, nR)
    return LWF(wannR=wannR, HwannR=H, Rlist=Rlist, Rdeg=Rdeg)


def test_nc_roundtrip_preserves_rdeg(tmp_path):
    lwf = make_lwf(Rdeg="random")
    f = tmp_path / "lwf.nc"
    lwf.write_nc(str(f))
    loaded = LWF.load_nc(str(f))
    assert np.allclose(loaded.Rdeg, lwf.Rdeg)


def test_nc_roundtrip_default_rdeg_ones(tmp_path):
    lwf = make_lwf(Rdeg=None)
    assert np.allclose(lwf.Rdeg, 1.0)
    f = tmp_path / "lwf2.nc"
    lwf.write_nc(str(f))
    loaded = LWF.load_nc(str(f))
    assert np.allclose(loaded.Rdeg, 1.0)


def test_get_wann_Hk_applies_rdeg():
    """H(k) must weight each R by Rdeg (legacy convention)."""
    from lawaf.mathutils.kR_convert import R_to_onek
    lwf = make_lwf()
    k = np.array([0.13, 0.21, 0.07])
    hk = lwf.get_wann_Hk(k)
    ref = R_to_onek(k, lwf.Rlist, lwf.HwannR, lwf.Rdeg)
    assert np.allclose(hk, ref, atol=1e-12)
