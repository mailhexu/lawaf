"""Tests for Wannierizer.write_w90 — the export integration (story-004).

Acceptance: a Wannierizer fed with real (SrMnO3, SMO w90-HR) or synthetic
states exports a complete wannier90 input set that the actual
``wannier90.x`` binary runs to completion.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from lawaf.params import WannierParams
from lawaf.utils.kpoints import monkhorst_pack
from lawaf.wannierization.wannierizer import Wannierizer

W90 = Path.home() / ".local/qe74_release/bin/wannier90.x"
SMO = Path(__file__).parent.parent / "example" / "Wannier90" / "SMO_wannier"


def _random_unitary_ensemble(nk, n, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((nk, n, n), dtype=complex)
    for ik in range(nk):
        q, _ = np.linalg.qr(
            rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        )
        out[ik] = q
    return out


class _FixedAmnWannierizer(Wannierizer):
    """Wannierizer returning a planted unitary Amn."""

    def set_fixed_amn(self, amn):
        self._fixed = amn

    def get_Amn_one_k(self, ik):
        return self._fixed[ik]


def _make_synthetic(nk=8, nb=4):
    params = WannierParams(
        method="scdmk", kmesh=(2, 2, 2), nwann=nb, use_proj=False,
        weight_func="unity",
    )
    rng = np.random.default_rng(0)
    wann = _FixedAmnWannierizer(
        params=params,
        evals=rng.standard_normal((nk, nb)),
        evecs=_random_unitary_ensemble(nk, nb, 1),
        kpts=monkhorst_pack([2, 2, 2]),
        kweights=np.full(nk, 1.0 / nk),
    )
    wann.set_fixed_amn(_random_unitary_ensemble(nk, nb, 2))
    wann.get_Amn()
    return wann


def test_write_w90_refuses_nonorthogonal(tmp_path):
    params = WannierParams(method="scdmk", kmesh=(2, 2, 2), nwann=4,
                           use_proj=False, weight_func="unity")
    nk, nb = 8, 4
    wann = _FixedAmnWannierizer(
        params=params,
        evals=np.zeros((nk, nb)),
        evecs=_random_unitary_ensemble(nk, nb, 1),
        kpts=monkhorst_pack([2, 2, 2]),
        kweights=np.full(nk, 1.0 / nk),
        Sk=np.stack([np.eye(nb) * 1.1] * nk),  # non-orthogonal basis
    )
    wann.set_fixed_amn(_random_unitary_ensemble(nk, nb, 2))
    wann.get_Amn()
    with pytest.raises(NotImplementedError):
        wann.write_w90(
            tmp_path / "seed", lattice=3.9 * np.eye(3),
            symbols=["Sr"], frac=[[0, 0, 0]],
        )


def test_write_w90_refuses_appended_anchor(tmp_path):
    """A downfolder-appended off-mesh anchor k-point (nkpt+1, weight 0)
    cannot be represented by mp_grid and must be rejected up front."""
    wann = _make_synthetic()
    wann.kpts = np.vstack([wann.kpts, [[0.11, 0.11, 0.11]]])
    with pytest.raises(ValueError, match="anchor"):
        wann.write_w90(
            tmp_path / "seed", lattice=3.9 * np.eye(3),
            symbols=["Sr"], frac=[[0, 0, 0]],
        )


def test_write_w90_synthetic_end_to_end(tmp_path):
    if not W90.exists():
        pytest.skip("wannier90.x not available")
    wann = _make_synthetic()
    prefix = tmp_path / "seed"  # pathlib.Path prefix must work
    wann.write_w90(
        prefix, lattice=3.9 * np.eye(3), symbols=["Sr"],
        frac=[[0, 0, 0]], projections=None,
    )
    for ext in (".win", ".amn", ".eig", ".mmn"):
        assert Path(str(prefix) + ext).exists()
    r = subprocess.run([str(W90), "seed"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    wout = (tmp_path / "seed.wout").read_text()
    assert "Neighbour not found" not in wout
    assert (tmp_path / "seed.chk").exists()


def test_write_w90_smo_srmmo3_end_to_end(tmp_path):
    """Acceptance run: real SrMnO3 w90-HR model -> lawaf SCDMk downfold ->
    lawaf w90 export -> wannier90.x localization run."""
    if not W90.exists():
        pytest.skip("wannier90.x not available")
    if not (SMO / "abinito_w90_down_hr.dat").exists():
        pytest.skip("SMO example data not available")
    from lawaf import W90Downfolder

    df = W90Downfolder(
        folder=str(SMO), prefix="abinito_w90_down",
        params=dict(
            method="scdmk", kmesh=(4, 4, 4), nwann=5,
            weight_func="Gauss", weight_func_params=(10.0, 3.0),
            use_proj=False, orthogonal=True,
        ),
    )
    df.downfold()
    # structure from the original win (ang; cubic SrMnO3, a=3.81)
    text = (SMO / "abinito_w90_down.win").read_text()
    lat_block = text.split("begin unit_cell_cart")[1].split("end")[0]
    rows = [ln for ln in lat_block.splitlines()[1:] if ln.strip()]
    lattice = np.array([[float(x) for x in ln.split()] for ln in rows[:3]])
    at_block = text.split("begin atoms_cart")[1].split("end")[0]
    symbols, cart = [], []
    for ln in at_block.splitlines()[1:]:
        parts = ln.split()
        if len(parts) == 4:
            symbols.append(parts[0])
            cart.append([float(x) for x in parts[1:4]])
    frac = np.array(cart) @ np.linalg.inv(lattice)

    prefix = tmp_path / "seed"
    df.builder.write_w90(
        str(prefix), lattice=lattice, symbols=symbols, frac=frac,
        projections=None,
        extra={"dis_num_iter": 200, "num_iter": 200,
               "dis_win_min": -15.0, "dis_win_max": 15.0},
    )
    r = subprocess.run([str(W90), "seed"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (tmp_path / "seed.wout")
    wout = (tmp_path / "seed.wout").read_text()
    assert "Neighbour not found" not in wout
    assert (tmp_path / "seed.chk").exists()
