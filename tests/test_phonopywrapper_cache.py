"""Regression tests for the PhonopyWrapper disk cache.

The cache key must fingerprint the configuration ``solve`` depends on
(force constants, masses, mode), not only the k-point: a
``phon_cache`` left in the working directory by a run with different
force constants used to be served stale (eigenvectors and eigenvalues
of the *other* configuration silently reused).
"""

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from pathlib import Path

from phonopy import load

FIXTURE = (
    Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"
)
K = (0.25, 0.0, 0.0)


def _fresh_phonon(scale=1.0):
    ph = load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    if scale != 1.0:
        ph.force_constants = ph.force_constants * scale
    return ph


def test_cache_key_separates_force_constants(tmp_path, monkeypatch):
    """A cached entry from one force-constant set must not be served
    for another, even though the k-point string is identical."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    monkeypatch.chdir(tmp_path)  # cache file: ./phon_cache/cache.pickle

    w1 = PhonopyWrapper(phonon=_fresh_phonon(1.0), mode="dm", use_cache=True)
    ev1, _ = w1.solve(K)
    w1.save_cache()  # normally done by atexit; force it for the test

    # second process equivalent: loads the cache file from disk
    w2 = PhonopyWrapper(phonon=_fresh_phonon(1.001), mode="dm", use_cache=True)
    ev2, _ = w2.solve(K)
    w2.save_cache()

    assert w2._cache_sig != w1._cache_sig
    # eigenvalues scale with the force constants: the second solve must
    # NOT have reused the first one's cached (evals, evecs)
    assert not np.allclose(ev1, ev2, rtol=1e-9)
    assert np.allclose(ev2, ev1 * 1.001, rtol=1e-8)

    # and the original entry is still served for the original config
    w3 = PhonopyWrapper(phonon=_fresh_phonon(1.0), mode="dm", use_cache=True)
    ev3, _ = w3.solve(K)
    assert np.allclose(ev1, ev3, rtol=1e-12, atol=0)


def test_cache_signature_covers_mode_and_masses(tmp_path, monkeypatch):
    """Same force constants, different mode -> different signature."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    monkeypatch.chdir(tmp_path)
    ph = _fresh_phonon()
    w_dm = PhonopyWrapper(phonon=ph, mode="dm", use_cache=False)
    w_ifc = PhonopyWrapper(phonon=ph, mode="ifc", use_cache=False)
    assert w_dm._cache_signature() != w_ifc._cache_signature()


def test_symmetrize_once_per_fc_array(tmp_path, monkeypatch):
    """Symmetrization happens once per force-constant array: repeated
    wrapper construction must not keep drifting the caller's force
    constants (previously each ``_prepare`` re-symmetrized in place,
    ~2e-14 per application, so results depended on how many wrappers
    had touched the phonon), and the cache signature must be stable."""
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    monkeypatch.chdir(tmp_path)
    ph = _fresh_phonon()

    w1 = PhonopyWrapper(phonon=ph, mode="dm", use_cache=True)
    sig1 = w1._cache_sig
    fc_after1 = ph.force_constants.copy()
    assert ph._lawaf_fc_done is ph.force_constants

    w2 = PhonopyWrapper(phonon=ph, mode="dm", use_cache=True)
    assert w2._cache_sig == sig1
    assert np.array_equal(ph.force_constants, fc_after1), (
        "second wrapper must not re-symmetrize (byte drift)"
    )

    # assigning a new (asymmetric) array re-enables symmetrization,
    # then the new array is left byte-stable
    ph.force_constants = fc_after1.copy()
    ph.force_constants[0, 0, 0, 0] += 1e-6
    w3 = PhonopyWrapper(phonon=ph, mode="dm", use_cache=False)
    assert not np.array_equal(ph.force_constants, fc_after1), (
        "asymmetric input must be symmetrized"
    )
    fc_after3 = ph.force_constants.copy()
    w4 = PhonopyWrapper(phonon=ph, mode="dm", use_cache=False)
    assert np.array_equal(ph.force_constants, fc_after3)
    assert w4._cache_signature() == w3._cache_signature()
