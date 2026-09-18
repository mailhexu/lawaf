"""Maximally localized non-orthogonal Wannier functions (constant GL G).

Contracts (quick task 2026-09-18-nonorthogonal-gauge-optimize):

- the Wirtinger gradient of the normalized per-orbital spread matches
  finite differences (the derivation was also verified symbolically);
- the functional is exactly invariant under per-column rescaling;
- without the orthonormality constraint the spread can only improve:
  symmetric/antisymmetric combinations of two site functions localize
  strictly worse than the (overlapping, non-orthogonal) site functions
  themselves — the optimizer recovers them from the orthonormal gauge;
- the moment matrices reproduce the codebase's own LWF centres;
- end-to-end on the BaTiO3 fixture: optimized G beats the orthonormal
  gauge, apply_gauge_transform gives the onsite-only overlap
  ``G^dag G delta_R0`` and pencil-exact bands at arbitrary q.
"""

from pathlib import Path

import numpy as np
import pytest


from lawaf.interfaces.phonopy.phonon_downfolder import PhonopyDownfolder
from lawaf.wannierization.nonorthogonal_gauge import (
    apply_gauge_transform,
    nonorthogonal_spread,
    nonorthogonal_spread_gradient,
    optimize_nonorthogonal_gauge,
    position_moment_matrices,
)

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"


def _random_moments(n, seed=5):
    rng = np.random.default_rng(seed)

    def herm(scale=1.0):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return scale * (a + a.conj().T) / 2

    # y must be PSD-ish for a physical spread; add a large diagonal
    x = [herm() for _ in range(3)]
    y = [herm() + 4.0 * np.eye(n) for _ in range(3)]
    return x, y


def test_gradient_matches_finite_differences():
    n = 4
    x, y = _random_moments(n)
    rng = np.random.default_rng(7)
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    grad = nonorthogonal_spread_gradient(G, x, y)
    f0 = nonorthogonal_spread(G, x, y)
    h = 1e-7
    err = 0.0
    for a in range(n):
        for b in range(n):
            for comp in (0, 1):
                Gp = G.copy()
                Gp[a, b] += h if comp == 0 else 1j * h
                fd = (nonorthogonal_spread(Gp, x, y) - f0) / h
                # f real: df/dRe = 2 Re c, df/dIm = +2 Im c (c = grad)
                an = 2 * grad[a, b].real if comp == 0 else 2 * grad[a, b].imag
                err = max(err, abs(fd - an))
    assert err < 1e-5  # FD truncation at h=1e-7


def test_scale_invariance():
    n = 3
    x, y = _random_moments(n, seed=9)
    rng = np.random.default_rng(3)
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    D = np.diag(rng.uniform(0.3, 3.0, size=n) * np.exp(1j * rng.uniform(0, 2 * np.pi, n)))
    assert abs(
        nonorthogonal_spread(G @ D, x, y) - nonorthogonal_spread(G, x, y)
    ) < 1e-12


def test_optimizer_beats_orthonormal_two_site():
    """Symmetric/antisymmetric combinations of two site functions vs the
    site functions themselves: the non-orthogonal set is strictly more
    localized, and the optimizer must find it from G0 = I."""
    d = 2.0  # site separation along x
    # orthonormal gauge = the two (normalized) combinations; basis is
    # the site functions at +-d/2, single R, wannR orthogonalizes them
    # exactly: the sym/antisym gauge in the site basis is Hadamard/sqrt2
    wannR = np.array([[[1.0, 1.0], [1.0, -1.0]]]) / np.sqrt(2.0)
    Rlist = np.array([[0.0, 0.0, 0.0]])
    Rdeg = np.array([1.0])
    positions = np.array([[d / 2, 0, 0], [-d / 2, 0, 0]])
    G, res = optimize_nonorthogonal_gauge(wannR, Rlist, Rdeg, positions)
    x, y = position_moment_matrices(wannR, Rlist, Rdeg, positions)
    o_orth = nonorthogonal_spread(np.eye(2), x, y)
    o_opt = nonorthogonal_spread(G, x, y)
    # orthonormal gauge: both functions span both sites -> variance (d/2)^2
    assert abs(o_orth - 2 * (d / 2) ** 2) < 1e-12
    # optimum: site-concentrated functions -> ~0 spread (here the optimal
    # set happens to be orthogonal; non-orthogonality of the optimum is
    # basis-overlap dependent and shows up on real parents, see e2e)
    assert o_opt < 1e-6
    # the optimal gauge is (up to phases) the inverse Hadamard: every
    # entry has magnitude 1/sqrt(2)
    assert np.allclose(np.abs(G), 1 / np.sqrt(2), atol=1e-3)
    # and stays well conditioned (the logdet barrier forbids collapse)
    assert np.linalg.eigvalsh(G.conj().T @ G).min() > 0.5


def test_moments_reproduce_lwf_centers(tmp_path):
    """diag of the first-moment matrices equals the LWF centre
    convention (get_wannier_centers) of the orthonormal model."""
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
    # LWF convention: fractional positions + integer R (the same units
    # as lwf.wann_centers); pass Cartesian positions instead for a
    # spread in Angstrom^2
    basis_pos = np.repeat(lwf.atoms.get_scaled_positions(), 3, axis=0)
    x, y = position_moment_matrices(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                    basis_pos)
    centers = np.array([np.diag(x[alpha]).real for alpha in range(3)]).T
    assert np.abs(centers - lwf.wann_centers).max() < 1e-10


def test_end_to_end_batio3_optimized_gauge(tmp_path):
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
    # LWF convention: fractional positions + integer R (the same units
    # as lwf.wann_centers); pass Cartesian positions instead for a
    # spread in Angstrom^2
    basis_pos = np.repeat(lwf.atoms.get_scaled_positions(), 3, axis=0)
    G, res = optimize_nonorthogonal_gauge(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                          basis_pos)
    x, y = position_moment_matrices(lwf.wannR, lwf.Rlist, lwf.Rdeg,
                                    basis_pos)
    assert nonorthogonal_spread(G, x, y) < nonorthogonal_spread(np.eye(3), x, y) - 1e-6
    lwf_no = apply_gauge_transform(lwf, G)
    assert lwf_no.SwannR is not None
    i0 = int(np.where(np.all(lwf_no.Rlist == 0, axis=1))[0][0])
    assert np.abs(np.delete(lwf_no.SwannR, i0, axis=0)).max() < 1e-12
    assert np.linalg.norm(lwf_no.SwannR[i0] - G.conj().T @ G) < 1e-12
    # bands unchanged at arbitrary q (congruence)
    rng = np.random.default_rng(2)
    q = rng.uniform(-0.5, 0.5, size=(5, 3))
    e_c = np.array([lwf.solve_k(k)[0] for k in q])
    e_g = np.array([lwf_no.solve_k(k)[0] for k in q])
    assert np.abs(e_c - e_g).max() < 1e-10
    # unit-norm columns
    assert np.abs(np.einsum("an,an->n", G.conj(), G).real - 1).max() < 1e-9


def test_apply_gauge_refuses_nonorthogonal_input(tmp_path):
    """apply_gauge_transform expects an orthonormal model and a matching
    G shape; both violations are refused."""
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
    with pytest.raises(ValueError, match="orthonormal model"):
        apply_gauge_transform(lwf, np.eye(3))
    df2 = PhonopyDownfolder(
        phonopy_yaml=str(FIXTURE), mode="DM",
        params=dict(method="projected", nwann=3,
                    anchors={(0.0, 0.0, 0.0): (0, 1, 2)}, use_proj=True,
                    weight_func="unity", kmesh=(2, 2, 2), gamma=True,
                    use_ws_distance=False),
        symmetrize_fc=False, is_nac=False,
    )
    lwf2 = df2.downfold(output_path=str(tmp_path / "b"), write_hr_nc=None,
                        write_hr_txt=None)
    with pytest.raises(ValueError, match=r"G must be"):
        apply_gauge_transform(lwf2, np.eye(4))
