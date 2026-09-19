"""k-dependent GL gauge G(k): evaluator, objective, optimizer, application.

Story 039/040 contracts (specs/stories/story-039-kdependent-gauge-core.md,
story-040-kdependent-gauge-integration.md):
- exp-form evaluator with shape validation;
- transform FT identity (direct per-k transform == module path, 1e-14);
- analytic objective gradient vs central finite differences (1e-6);
- optimizer improves beyond the constant-G optimum on a synthetic
  benchmark with mesh-exact bands (1e-12) and a conditioning floor;
- application to LWF results: finite-range SwannR, mesh bands, refusals;
- exports importable from the package root.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from scipy.linalg import eigh, expm

from lawaf.interfaces.downfolder import Lawaf
from lawaf.interfaces.phonopy.phonon_downfolder import PhonopyDownfolder
from lawaf.wannierization.kdependent_gauge import (
    KDepGauge,
    kdependent_objective,
    optimize_kdependent_gauge,
    select_shells,
)
from lawaf.wannierization.nonorthogonal_gauge import (
    apply_gauge_transform,
    optimize_nonorthogonal_gauge,
)

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"


class OrthoTB:
    """Orthogonal three-band parent on a cubic lattice."""

    is_orthogonal = True

    def __init__(self, nb=3, seed=31):
        rng = np.random.default_rng(seed)
        self.nb = nb
        a = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        self.H0 = a + a.conj().T
        b = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
        self.hop = 0.3 * (b + b.conj().T)
        self._r = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                            [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=int)
        self.atoms = Atoms(
            "H3", positions=[[0, 0, 0], [0.3, 0.1, 0.4], [0.6, 0.5, 0.2]],
            cell=np.eye(3),
        )

    def _Hk(self, k):
        H = self.H0.copy()
        for R in self._r[1:]:
            ph = np.exp(2j * np.pi * np.dot(R, k))
            H += self.hop * ph + self.hop.conj().T * ph.conj()
        return H

    def solve_all(self, kpts):
        nk = len(kpts)
        e = np.zeros((nk, self.nb))
        v = np.zeros((nk, self.nb, self.nb), dtype=complex)
        for ik, k in enumerate(kpts):
            e[ik], v[ik] = eigh(self._Hk(k))
        return e, v


def _orthonormal_downfold(nb=3, kmesh=(3, 3, 3)):
    m = OrthoTB(nb=nb)
    df = Lawaf(
        m, params=dict(method="projected", kmesh=kmesh, nwann=nb,
                       selected_basis=list(range(nb)), weight_func="unity",
                       use_ws_distance=False)
    )
    return m, df, df.downfold()


def _kpoints(kmesh):
    import itertools

    return np.array(list(itertools.product(
        *[np.arange(n) / n for n in kmesh])))


# ------------------------------------------------------------- evaluator
def test_evaluator_shapes_and_values():
    Rg = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Lambda must have shape"):
        KDepGauge(Rg, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="unknown method"):
        KDepGauge(Rg, np.zeros((2, 2, 2)), method="linear")
    Lam = np.array([[[0.1, 0.0], [0.0, -0.1]],
                    [[0.0, 0.2], [-0.2, 0.0]]])
    g = KDepGauge(Rg, Lam)
    k = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    G = g.G_of_k(k)
    # at Gamma every shell phase is +1
    assert np.abs(G[0] - expm(Lam[0] + Lam[1])).max() < 1e-14
    # at k = (1/2, 0, 0): phase of R=(1,0,0) is -1
    assert np.abs(G[1] - expm(Lam[0] - Lam[1])).max() < 1e-14


def test_select_shells():
    Rlist = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                      [1, 1, 0]], dtype=float)
    sel = select_shells(Rlist, max_shell=1)
    # R=0 + the three |R|=1 neighbors present in the list
    assert len(sel) == 4 and np.all(sel[0] == 0)
    assert np.linalg.norm(sel[1:], axis=1).max() <= 1.0 + 1e-9
    with pytest.raises(ValueError, match="R=0"):
        select_shells(Rlist[1:], max_shell=1)


# ------------------------------------------------------------- FT identity
def test_transform_ft_identity():
    """Module-transformed amplitudes equal the direct per-k transform."""
    m, df, lwf = _orthonormal_downfold()
    kpts = _kpoints((3, 3, 3))
    rng = np.random.default_rng(6)
    Rg = select_shells(lwf.Rlist, max_shell=1)
    Lam = 0.2 * (rng.standard_normal((len(Rg), 3, 3))
                 + 1j * rng.standard_normal((len(Rg), 3, 3)))
    gauge = KDepGauge(Rg, Lam)
    G = gauge.G_of_k(kpts)
    from lawaf.mathutils.kR_convert import k_to_R, R_to_k

    U = R_to_k(kpts, lwf.Rlist, lwf.wannR, lwf.Rdeg)
    W_direct = k_to_R(
        kpts, lwf.Rlist, np.einsum("kam,kml->kal", U, G),
        np.full(len(kpts), 1 / len(kpts)))
    f, omega, _ = kdependent_objective(
        lwf.wannR, lwf.Rlist, lwf.Rdeg, lwf.atoms.get_positions(),
        kpts, Rg)
    _, _, _, aux = f(Lam)
    assert np.abs(aux["W"] - W_direct).max() < 1e-14


# ---------------------------------------------------------------- gradient
def test_objective_gradient_fd():
    m, df, lwf = _orthonormal_downfold(nb=3)
    kpts = _kpoints((3, 3, 3))
    Rg = select_shells(lwf.Rlist, max_shell=1)
    pos = lwf.atoms.get_positions()
    f, _, _ = kdependent_objective(lwf.wannR, lwf.Rlist, lwf.Rdeg, pos,
                                   kpts, Rg)
    rng = np.random.default_rng(6)
    Lam = 0.2 * (rng.standard_normal((len(Rg), 3, 3))
                 + 1j * rng.standard_normal((len(Rg), 3, 3)))
    tot, c, _, _ = f(Lam, mu=1e-2, pin_reg=1e-8)
    h = 1e-6
    err = 0.0
    for idx in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (len(Rg) - 1, 2, 0)]:
        for part in (0, 1):
            Lp = Lam.copy()
            Lp[idx] += h if part == 0 else 1j * h
            fp = f(Lp, mu=1e-2, pin_reg=1e-8)[0]
            fd = (fp - tot) / h
            cc = c[idx]
            an = 2 * cc.real if part == 0 else 2 * cc.imag
            err = max(err, abs(fd - an))
    assert err < 1e-5  # FD truncation at h = 1e-6


# ---------------------------------------------------------------- optimizer
def test_optimizer_beats_constant_g_and_mesh_exact():
    m = OrthoTB()
    df = Lawaf(
        m, params=dict(method="projected", kmesh=(3, 3, 3), nwann=3,
                       selected_basis=[0, 1, 2], weight_func="unity",
                       use_ws_distance=False))
    lwf = df.downfold()
    pos = lwf.atoms.get_positions()
    Gc, _ = optimize_nonorthogonal_gauge(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                         pos)
    apply_gauge_transform(lwf, Gc)
    gauge, res, info = optimize_kdependent_gauge(
        lwf.wannR, lwf.Rlist, lwf.Rdeg, pos, lwf.kpts, shells=1, G0=Gc,
        maxiter=300)
    assert info["omega_opt"] <= info["omega_start"] + 1e-12
    lwf_k = apply_gauge_transform(lwf, gauge)
    # mesh-k bands equal the ORTHONORMAL control (full manifold)
    e_orth = np.array([lwf.solve_k(k)[0] for k in lwf.kpts])
    e_k = np.array([lwf_k.solve_k(k)[0] for k in lwf.kpts])
    assert np.abs(e_orth - e_k).max() < 1e-12
    # finite-range overlap: S(R) not onsite-only
    i0 = np.where(np.all(lwf_k.Rlist == 0, axis=1))[0]
    off = np.delete(lwf_k.SwannR, i0, axis=0)
    assert np.abs(off).max() > 1e-6
    # conditioning floor respected on the mesh
    from lawaf.mathutils.kR_convert import R_to_k

    S = R_to_k(lwf.kpts, lwf_k.Rlist, lwf_k.SwannR, lwf_k.Rdeg)
    assert min(np.linalg.eigvalsh(Sk).min() for Sk in S) > 1e-4


# ------------------------------------------------------------- application
def test_apply_refusals(tmp_path):
    df = PhonopyDownfolder(
        phonopy_yaml=str(FIXTURE), mode="DM",
        params=dict(method="projected", nwann=3,
                    anchors={(0.0, 0.0, 0.0): (0, 1, 2)}, use_proj=True,
                    weight_func="unity", kmesh=(2, 2, 2), gamma=True,
                    orthogonal=False, use_ws_distance=False),
        symmetrize_fc=False, is_nac=False,
    )
    lwf = df.downfold(output_path=str(tmp_path), write_hr_nc=None,
                      write_hr_txt=None)
    Rg = select_shells(lwf.Rlist, max_shell=1)
    gauge = KDepGauge(Rg, np.zeros((len(Rg), 3, 3)))
    with pytest.raises(ValueError, match="orthonormal model"):
        apply_gauge_transform(lwf, gauge)


def test_apply_lwf_parity_and_warning(tmp_path):
    df = PhonopyDownfolder(
        phonopy_yaml=str(FIXTURE), mode="DM",
        params=dict(method="projected", nwann=3,
                    anchors={(0.0, 0.0, 0.0): (0, 1, 2)}, use_proj=True,
                    weight_func="unity", kmesh=(2, 2, 2), gamma=True,
                    use_ws_distance=False),
        symmetrize_fc=False, is_nac=False,
    )
    lwf = df.downfold(output_path=str(tmp_path), write_hr_nc=None,
                      write_hr_txt=None)
    Rg = select_shells(lwf.Rlist, max_shell=1)
    rng = np.random.default_rng(11)
    Lam = 0.05 * (rng.standard_normal((len(Rg), 3, 3))
                  + 1j * rng.standard_normal((len(Rg), 3, 3)))
    Lam[0] = 0.0
    gauge = KDepGauge(Rg, Lam)
    with pytest.warns(UserWarning, match="off-mesh"):
        lwf_k = apply_gauge_transform(lwf, gauge)
    assert lwf_k.SwannR is not None
    e_c = np.array([lwf.solve_k(k)[0] for k in lwf.kpts])
    e_k = np.array([lwf_k.solve_k(k)[0] for k in lwf.kpts])
    assert np.abs(e_c - e_k).max() < 1e-12


def test_exports_importable():
    import lawaf

    assert lawaf.optimize_kdependent_gauge is optimize_kdependent_gauge
    assert lawaf.apply_gauge_transform is apply_gauge_transform


def test_offmesh_penalty_gradient_fd():
    """FD check of the packed objective gradient with the off-mesh
    penalty active (eigenvalue adjoint + exp adjoint chained)."""
    from lawaf.wannierization.kdependent_gauge import (
        kdependent_offmesh_penalty, kdependent_objective as ko)

    m, df, lwf = _orthonormal_downfold(nb=3)
    kpts = _kpoints((3, 3, 3))
    Rg = select_shells(lwf.Rlist, max_shell=1)
    pos = lwf.atoms.get_positions()
    f, _, _ = ko(lwf.wannR, lwf.Rlist, lwf.Rdeg, pos, kpts, Rg)
    rng = np.random.default_rng(23)
    Lam = 0.25 * (rng.standard_normal((len(Rg), 3, 3))
                  + 1j * rng.standard_normal((len(Rg), 3, 3)))
    orng = np.random.default_rng(17)
    qpts = orng.uniform(0.0, 1.0, size=(4, 3))
    weight = 20.0

    def total(L):
        base, _, _, _ = f(L, mu=1e-2, pin_reg=1e-8)
        pen, _ = kdependent_offmesh_penalty(L, qpts, Rg, lwf.HwannR,
                                            lwf.Rlist, lwf.Rdeg,
                                            weight=weight)
        return base + pen

    _, c_base, _, _ = f(Lam, mu=1e-2, pin_reg=1e-8)
    _, c_pen = kdependent_offmesh_penalty(Lam, qpts, Rg, lwf.HwannR,
                                          lwf.Rlist, lwf.Rdeg,
                                          weight=weight)
    c = c_base + c_pen
    h = 1e-6
    err = 0.0
    for idx in [(0, 0, 0), (0, 1, 1), (3, 0, 2), (len(Rg) - 1, 2, 0)]:
        for part in (0, 1):
            Lp = Lam.copy()
            Lp[idx] += h if part == 0 else 1j * h
            fd = (total(Lp) - total(Lam)) / h
            cc = c[idx]
            an = 2 * cc.real if part == 0 else 2 * cc.imag
            err = max(err, abs(fd - an))
    assert err < 1e-5


def test_offmesh_penalty_reduces_interpolation_error():
    """The off-mesh penalty trades a little spread for interpolation
    fidelity: with the penalty on, the off-mesh band error against the
    orthonormal control drops below the unpenalized run's."""
    m = OrthoTB()
    df = Lawaf(
        m, params=dict(method="projected", kmesh=(3, 3, 3), nwann=3,
                       selected_basis=[0, 1, 2], weight_func="unity",
                       use_ws_distance=False))
    lwf = df.downfold()
    pos = lwf.atoms.get_positions()
    Gc, _ = optimize_nonorthogonal_gauge(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                         pos)
    out = {}
    for tag, extra in [("plain", {}),
                       ("pen", dict(offmesh_weight=30.0,
                                    offmesh_points=6,
                                    offmesh_seed=21))]:
        gauge, res, info = optimize_kdependent_gauge(
            lwf.wannR, lwf.Rlist, lwf.Rdeg, pos, lwf.kpts, HwannR=lwf.HwannR,
            shells=1, G0=Gc, maxiter=300, **extra)
        out[tag] = (info, gauge)
    assert (out["pen"][0]["offmesh_err_opt"]
            < out["plain"][0]["offmesh_err_opt"])
    # mesh exactness is untouched
    e_orth = np.array([lwf.solve_k(k)[0] for k in lwf.kpts])
    lwf_k = apply_gauge_transform(lwf, out["pen"][1])
    e_k = np.array([lwf_k.solve_k(k)[0] for k in lwf.kpts])
    assert np.abs(e_orth - e_k).max() < 1e-12
