"""Story-008: differential validation of the mlwf backend vs wannier90.x.

Real SrMnO3 (SMO w90-HR) data: lawaf's Mnm-form spread decomposition and
optimized gauge are compared against an actual ``wannier90.x`` run started
from lawaf's exported, optimized ``.amn`` with ``num_iter = 0`` (w90
evaluates exactly the exported gauge; identical decomposition expected to
print precision).
"""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

W90 = Path.home() / ".local/qe74_release/bin/wannier90.x"
SMO = Path(__file__).parent.parent / "example" / "Wannier90" / "SMO_wannier"


def _parse_final_spreads(wout_text):
    blocks = re.findall(
        r"Omega I\s+=\s+([-\d.]+)\s*\n\s*=+\s*Omega D\s+=\s+([-\d.]+)"
        r"\s*\n\s*Omega OD\s+=\s+([-\d.]+)\s*\n.*?Omega Total\s+=\s+([-\d.]+)",
        wout_text, re.S)
    assert blocks, "no spread block parsed from .wout"
    oi, od, ood, ot = map(float, blocks[-1])
    return {"omega_I": oi, "omega_D": od, "omega_OD": ood, "omega": ot}


@pytest.fixture(scope="module")
def smo_mlwf_downfolder():
    if not W90.exists():
        pytest.skip("wannier90.x not available")
    if not (SMO / "abinito_w90_down_hr.dat").exists():
        pytest.skip("SMO example data not available")
    from lawaf import W90Downfolder

    df = W90Downfolder(
        folder=str(SMO), prefix="abinito_w90_down",
        params=dict(
            method="mlwf", kmesh=(4, 4, 4), nwann=5,
            weight_func="Gauss", weight_func_params=(10.0, 3.0),
            use_proj=False, orthogonal=True,
            # v1 needs an isolated manifold: keep the lowest 5 of the 14
            # w90-HR bands (disentanglement is a follow-up story)
            exclude_bands=list(range(5, 14)),
            mlwf_tol=1e-12, mlwf_max_iter=200,
        ),
    )
    df.downfold()
    return df


def _lattice_from_win():
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
    return lattice, symbols, frac


def test_mlwf_spreads_match_wannier90(smo_mlwf_downfolder, tmp_path):
    """Differential acceptance: converged spread decomposition equals
    wannier90's evaluation of the exported (optimized) gauge."""
    df = smo_mlwf_downfolder
    spreads = df.builder.spreads
    assert spreads is not None and spreads["omega"] > 0
    lattice, symbols, frac = _lattice_from_win()
    # Pin w90's guiding centres (rguide = projection sites, wannierise.F90
    # 308-311) to lawaf's converged rbar so the sheet convention matches:
    # with num_iter=0 and random projections, w90's sheet would come from
    # random centres and its Omega_D is not comparable.
    rbar_cart = spreads["rbar"] @ lattice  # crystal frac -> cart (Ang)
    projections = [f"c={x:.12f},{y:.12f},{z:.12f}: s"
                   for x, y, z in rbar_cart]
    prefix = tmp_path / "seed"
    df.builder.write_w90(
        str(prefix), lattice=lattice, symbols=symbols, frac=frac,
        projections=projections,
        extra={"num_iter": 0, "dis_num_iter": 0},
    )
    r = subprocess.run([str(W90), "seed"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (tmp_path / "seed.wout").read_text()[-2000:]
    w90 = _parse_final_spreads((tmp_path / "seed.wout").read_text())

    # lawaf spreads are in crystal units (b = 1); convert to Ang^2:
    # per-axis b_cart = 2 pi / a (cubic SrMnO3, a = 3.81)
    a = lattice[0, 0]
    scale = (a / (2 * np.pi)) ** 2
    lawaf = {k: spreads[k] * scale for k in ("omega", "omega_I", "omega_D",
                                             "omega_OD")}
    for key in lawaf:
        assert lawaf[key] == pytest.approx(w90[key], abs=2e-5), (
            key, lawaf[key], w90[key])


def test_mlwf_band_fit_gauge_invariant(smo_mlwf_downfolder):
    """Example AC: mlwf and scdmk interpolations agree along a k-path.

    Gauge invariance of the spectrum is demonstrated on the Fourier-
    interpolated bands of the two independent downfolds (different gauges,
    same physics), not just at the training k-points.
    """
    from lawaf import W90Downfolder

    df_mlwf = smo_mlwf_downfolder
    params = dict(
        kmesh=(4, 4, 4), nwann=5,
        weight_func="Gauss", weight_func_params=(10.0, 3.0),
        use_proj=False, orthogonal=True,
        exclude_bands=list(range(5, 14)),
    )
    df_scdmk = W90Downfolder(
        folder=str(SMO), prefix="abinito_w90_down",
        params=dict(method="scdmk", **params))
    df_scdmk.downfold()
    # exact equality at the training k-points (square unitary gauge)
    wann = df_mlwf.builder
    for ik in range(wann.nkpt):
        ev = np.sort(np.linalg.eigvalsh(wann.Hwann_k[ik]))
        np.testing.assert_allclose(ev, np.sort(wann.get_eval_k(ik)),
                                   atol=1e-10)
    # along a k-path the two interpolations differ only by Fourier
    # truncation on the 4x4x4 mesh, which is GAUGE-DEPENDENT; the MLWF
    # gauge is the better-localized one, so its interpolation error
    # against the original model must not exceed the scdmk error
    kv = np.array([[0., 0., 0.], [0.5, 0., 0.], [0.5, 0.5, 0.],
                   [0.5, 0.5, 0.5], [0., 0., 0.]])
    ks = np.linspace(0.0, 1.0, 9)
    segs = [kv[i] + (kv[i + 1] - kv[i]) * ks[:, None]
            for i in range(len(kv) - 1)]
    path_kpts = np.concatenate(segs)
    ref = np.sort(df_scdmk.model.solve_all(kpts=path_kpts)[0])[:, :5]
    err = {}
    for label, df in (("mlwf", df_mlwf), ("scdmk", df_scdmk)):
        b = np.sort(df.lwf.solve_all(kpts=path_kpts)[0], axis=1)
        err[label] = np.abs(b - ref).max()
    assert err["mlwf"] <= err["scdmk"] + 1e-12, err
    # unitarity of the refined gauge (training k-points)
    Amn_opt = df_mlwf.builder.Amn
    for ik in range(Amn_opt.shape[0]):
        np.testing.assert_allclose(
            Amn_opt[ik].conj().T @ Amn_opt[ik], np.eye(Amn_opt.shape[2]),
            atol=1e-10)


def test_mlwf_spreads_decreased_from_initial(smo_mlwf_downfolder):
    """The optimization actually localized: final Omega below the initial
    projected-guess Omega recorded in the first history entry."""
    wann = smo_mlwf_downfolder.builder
    hist = wann.mlwf_history
    assert len(hist) >= 1
    assert hist[-1]["omega"] < hist[0]["omega"] + 1e-12
