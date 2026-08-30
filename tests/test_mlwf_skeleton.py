"""Tests for the MLWFWannierizer skeleton (story-005).

ADR-3 (orthogonal v1, S != None refuses naming the S-metric Mmn),
ADR-5 (registration mlwf/maxloc/mv, declared mlwf_* params, unknown-key
warning).
"""

import warnings

import numpy as np
import pytest

from lawaf.interfaces.downfolder import select_wannierizer
from lawaf.params import WannierParams
from lawaf.utils.kpoints import monkhorst_pack
from lawaf.wannierization.mlwf import MLWFWannierizer
from lawaf.wannierization.projectedWF import ProjectedWannierizer


@pytest.mark.parametrize("method", ["mlwf", "MLWF", "maxloc", "mv"])
def test_registration(method):
    assert select_wannierizer(method) is MLWFWannierizer


def test_unknown_method_still_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        select_wannierizer("not-a-method")


def test_mlwf_params_defaults_and_roundtrip():
    p = WannierParams(method="mlwf")
    assert p.mlwf_tol == 1e-10
    assert p.mlwf_max_iter == 100
    assert p.mlwf_initial_guess == "projected"
    assert p.mlwf_fixed_gauge is False
    p2 = WannierParams(method="mlwf", mlwf_tol=1e-8, mlwf_max_iter=7,
                       mlwf_initial_guess="scdmk", mlwf_fixed_gauge=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = p2.to_dict()
        p3 = WannierParams()
        p3.from_dict(d)
    assert (p3.mlwf_tol, p3.mlwf_max_iter,
            p3.mlwf_initial_guess, p3.mlwf_fixed_gauge) == \
        (1e-8, 7, "scdmk", True)


def test_unknown_param_warns():
    with pytest.warns(UserWarning, match="mlwf_tolX"):
        WannierParams(method="scdmk", mlwf_tolX=1.0)
    # known keys do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        WannierParams(method="scdmk", mlwf_tol=1e-6, kmesh=(2, 2, 2))


def _basis_inputs(nk=8, nb=4, nw=2):
    rng = np.random.default_rng(0)
    evecs = np.empty((nk, nb, nb), dtype=complex)
    for ik in range(nk):
        q, _ = np.linalg.qr(
            rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        )
        evecs[ik] = q
    params = WannierParams(method="mlwf", kmesh=(2, 2, 2), nwann=nw,
                           use_proj=True, weight_func="unity")
    return params, rng, evecs

def test_mlwf_params_cross_downfolder_boundary():
    """ADR-5: mlwf_* fields survive the Lawaf.set_parameters path so
    serialized parameter dicts can configure the mlwf backend."""
    from lawaf.interfaces.downfolder import Lawaf

    lf = Lawaf(object())
    lf.set_parameters(method="mlwf", mlwf_tol=1e-8, mlwf_max_iter=7,
                      mlwf_initial_guess="scdmk", mlwf_fixed_gauge=True)
    assert lf.params.mlwf_tol == 1e-8
    assert lf.params.mlwf_max_iter == 7
    assert lf.params.mlwf_initial_guess == "scdmk"
    assert lf.params.mlwf_fixed_gauge is True


def test_legacy_downfold_keys_do_not_warn():
    """Keys forwarded by existing scripts through downfold(**params) must
    stay accepted (mu, sigma, write_hr_nc, write_hr_txt, post_func)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = WannierParams(method="scdmk")
        p.update({"mu": 1.0, "sigma": 2.0, "write_hr_nc": "hr.nc",
                  "write_hr_txt": "hr.txt", "post_func": None})
    assert p.mu == 1.0


def test_classmethod_name_is_not_a_known_key():
    from lawaf.params import WannierParams as WP

    assert "from_toml" not in WP._known_keys()
    assert "method" in WP._known_keys()


def test_mlwf_refuses_nonorthogonal():
    params, rng, evecs = _basis_inputs()
    Sk = np.stack([np.eye(4) * 1.05] * 8)
    with pytest.raises(NotImplementedError, match="S-metric"):
        MLWFWannierizer(
            params=params, evals=rng.standard_normal((8, 4)),
            evecs=evecs, kpts=monkhorst_pack([2, 2, 2]),
            kweights=np.full(8, 1 / 8), Sk=Sk,
        )


def test_mlwf_orthogonal_accepts_and_projected_guess():
    """Orthogonal path constructs and the initial guess is the projected
    Amn (pass-through of ProjectedWannierizer.get_Amn_one_k)."""
    params, rng, evecs = _basis_inputs()
    kpts = monkhorst_pack([2, 2, 2])
    kwargs = dict(
        params=params, evals=rng.standard_normal((8, 4)),
        evecs=evecs, kpts=kpts, kweights=np.full(8, 1 / 8),
    )
    wann = MLWFWannierizer(**kwargs)
    proj = np.zeros((2, 4), dtype=complex)  # nwann x nbasis projectors
    proj[0, 0] = 1.0
    proj[1, 1] = 1.0
    wann.set_projectors(proj)
    ref = ProjectedWannierizer(**kwargs)
    ref.set_projectors(proj)
    for ik in range(8):
        np.testing.assert_allclose(
            wann.get_Amn_one_k(ik), ref.get_Amn_one_k(ik), atol=1e-12
        )
