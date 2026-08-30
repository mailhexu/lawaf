"""Story-006 tests: MV d_omega optimizer and Mmn-form spread decomposition.

Oracles from the executed derivation
(lawaf/docs/derivations/mlwf_domega_derivation.py, checks 5a/5c), stated
in CRYSTAL units (b = 1 per axis, see MLWFWannierizer._recip_from_kpts):
the physical closed forms Nk^2/(4 pi^2) sin^2(pi/Nk) become sin^2(pi/Nk).
"""

import numpy as np
import pytest

from lawaf.io.w90 import compute_Mmn
from scipy.linalg import expm

from lawaf.wannierization.mlwf import (
    MLWFWannierizer,
    _assert_pair_hermiticity,
    d_omega_optimize,
    gauged_mmn,
    omega_decomposition,
)

IDENT = np.eye(3)


def ring_mesh(nk):
    """1D ring as an N x 1 x 1 crystal-unit MP mesh + identity recip."""
    kpts = np.array([[n / nk, 0.0, 0.0] for n in range(nk)])
    return kpts


def xy_model_psi(nk, bands=2):
    """Derivation check-5a/5c model: u_+-k = (1, +-e^{i phi_k})/sqrt(2)."""
    psi = np.zeros((nk, 2, bands), dtype=complex)
    for k in range(nk):
        phi = 2 * np.pi * k / nk
        psi[k, :, 0] = np.array([1, np.exp(1j * phi)]) / np.sqrt(2)
        if bands == 2:
            psi[k, :, 1] = np.array([1, -np.exp(1j * phi)]) / np.sqrt(2)
    return psi


def build(nk, bands=2):
    """The exact derivation section-3 ring: neighbours +b/-b, w = 1/2 each.

    (The closed-form oracles are stated for this mesh; MLWFWannierizer's
    kmesh_nnlist integration is covered end-to-end by story-008.)
    """
    nnlist = np.array([[(k + 1) % nk, (k - 1) % nk] for k in range(nk)])
    nncell = np.array([[[1, 0, 0], [-1, 0, 0]] for _ in range(nk)])
    bvecs = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    wb = np.array([0.5, 0.5])
    psi = xy_model_psi(nk, bands=bands)
    mmn = compute_Mmn(psi, nnlist, nncell)
    return mmn, nnlist, nncell, bvecs, wb


class TestOmegaDecomposition:
    def test_single_band_closed_form(self):
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=1)
        s = omega_decomposition(mmn, wb, bvecs)
        # derivation check 5a in crystal units: Omega = Omega_I = sin^2(pi/N)
        assert s["omega_I"] == pytest.approx(np.sin(np.pi / nk) ** 2, abs=1e-12)
        assert s["omega_D"] == pytest.approx(0.0, abs=1e-12)
        assert s["omega_OD"] == pytest.approx(0.0, abs=1e-12)
        assert s["omega"] == pytest.approx(np.sin(np.pi / nk) ** 2, abs=1e-12)

    def test_two_band_identity_gauge(self):
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        s = omega_decomposition(mmn, wb, bvecs)
        # derivation check 5c in crystal units
        assert s["omega_I"] == pytest.approx(0.0, abs=1e-12)
        assert s["omega_D"] == pytest.approx(0.0, abs=1e-12)
        assert s["omega_OD"] == pytest.approx(2 * np.sin(np.pi / nk) ** 2,
                                              abs=1e-12)

    def test_per_band_and_gauge_invariance_of_omega_I(self):
        nk = 4
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        rng = np.random.default_rng(7)
        # random unitary gauge (right action)
        q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        U = [q.copy() for _ in range(nk)]
        s0 = omega_decomposition(mmn, wb, bvecs)
        s1 = omega_decomposition(gauged_mmn(mmn, nnlist, U), wb, bvecs)
        assert s1["omega_I"] == pytest.approx(s0["omega_I"], abs=1e-12)
        assert len(s0["omega_n"]) == 2
        assert sum(s0["omega_n"]) == pytest.approx(s0["omega"], abs=1e-12)


def projector_gauge(nk, bands=2, noise=0.0, seed=0):
    """Atomic-projector initial gauge for the xy-model chain (ADR-004):

    U0_k = (1/sqrt(2)) [[1, 1], [e^{i phi_k}, -e^{i phi_k}]] up to noise.
    """
    rng = np.random.default_rng(seed)
    U0 = []
    for k in range(nk):
        phi = 2 * np.pi * k / nk
        u = np.array([[1, 1], [np.exp(1j * phi), -np.exp(1j * phi)]]) / np.sqrt(2)
        if noise:
            h = rng.normal(size=(bands, bands)) + 1j * rng.normal(size=(bands, bands))
            a = noise * (h + h.conj().T) / 2
            u = u @ expm(1j * a)
        U0.append(u)
    return U0


class TestPairHermiticity:
    def test_consistent_mesh_passes(self):
        mmn, nnlist, nncell, bvecs, wb = build(8)
        _assert_pair_hermiticity(mmn, nnlist, nncell, bvecs)

    def test_corrupted_mmn_raises(self):
        mmn, nnlist, nncell, bvecs, wb = build(8)
        mmn[0, 0] += 0.1
        with pytest.raises(AssertionError, match="hermiticity"):
            _assert_pair_hermiticity(mmn, nnlist, nncell, bvecs)


class TestDOmegaOptimize:
    def test_single_band_one_sweep(self):
        """AC: single isolated band converges in 1 sweep with analytic Omega."""
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=1)
        rng = np.random.default_rng(3)
        ph = 0.3 * rng.uniform(-1, 1, nk)  # mild phase-gauge perturbation
        U0 = [np.array([[np.exp(1j * ph[k])]]) for k in range(nk)]
        U, hist = d_omega_optimize(mmn, nnlist, bvecs, wb, U0=U0, tol=1e-12,
                                   max_iter=100)
        target = np.sin(np.pi / nk) ** 2
        # the self-consistent phase update reaches the analytic minimum
        # within ONE sweep: the first recorded history entry (post-sweep-1)
        # already sits at the target, and the sweep-2 delta is < tol
        assert hist[0]["omega"] == pytest.approx(target, abs=1e-8)
        assert hist[-1]["omega"] == pytest.approx(target, abs=1e-8)
        assert len(hist) <= 2
        om = [h["omega"] for h in hist]
        assert all(om[i + 1] <= om[i] + 1e-12 for i in range(len(om) - 1))

    def test_two_band_chain_reaches_atomic_zero(self):
        """AC: 1D two-band chain Omega matches the analytic MLWF value 0.

        Start from the ADR-004 projector initial guess with noise (the
        perfectly symmetric identity gauge is a known local minimum of the
        MV functional, which is why the initial guess matters).
        """
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        U0 = projector_gauge(nk, bands=2, noise=0.2, seed=1)
        U, hist = d_omega_optimize(mmn, nnlist, bvecs, wb, U0=U0, tol=1e-12,
                                   max_iter=400)
        assert hist[-1]["omega"] < 1e-8

    def test_omega_monotone(self):
        """AC: Omega monotone non-increasing (line-search acceptance)."""
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        rng = np.random.default_rng(11)
        q, _ = np.linalg.qr(rng.normal(size=(2, 2))
                            + 1j * rng.normal(size=(2, 2)))
        U0 = [q.copy() for _ in range(nk)]
        U, hist = d_omega_optimize(mmn, nnlist, bvecs, wb, U0=U0, tol=1e-13,
                                   max_iter=100)
        om = [h["omega"] for h in hist]
        assert all(om[i + 1] <= om[i] + 1e-12 for i in range(len(om) - 1))

    def test_fixed_gauge_skips_unitary_update(self):
        """FR-007: fixed gauge leaves Omega_OD untouched."""
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        U, hist = d_omega_optimize(mmn, nnlist, bvecs, wb, tol=1e-12,
                                   fixed_gauge=True, max_iter=50)
        assert hist[-1]["omega_OD"] == pytest.approx(
            hist[0]["omega_OD"], abs=1e-12)
        assert hist[-1]["omega_OD"] > 1e-3  # unchanged from identity gauge

    def test_projector_start_noisy_converges(self):
        # (nk=4 with this noise lands the Wannier centre exactly at the
        # half-cell boundary rbar = -b/2 — the known MV97 branch-frustration
        # case where Omega_D has a finite floor; nk=8 avoids it)
        nk = 8
        mmn, nnlist, _, bvecs, wb = build(nk, bands=2)
        U0 = projector_gauge(nk, bands=2, noise=0.3, seed=5)
        U, hist = d_omega_optimize(mmn, nnlist, bvecs, wb, U0=U0,
                                   tol=1e-12, max_iter=400)
        assert hist[-1]["omega"] < 1e-8


class TestMLWFWannierizerIntegration:
    """Story-006/007 wiring: get_Amn refines the gauge and attaches spreads."""

    def _make(self, nk=8, seed=0):
        from lawaf.params import WannierParams
        from lawaf.utils.kpoints import monkhorst_pack

        rng = np.random.default_rng(seed)
        nb = nw = 2
        evecs = np.empty((nk, nb, nb), dtype=complex)
        for ik in range(nk):
            q, _ = np.linalg.qr(rng.standard_normal((nb, nb))
                                + 1j * rng.standard_normal((nb, nb)))
            evecs[ik] = q
        params = WannierParams(method="mlwf", kmesh=(2, 2, 2), nwann=nw,
                               use_proj=True, weight_func="unity",
                               mlwf_max_iter=20)
        wann = MLWFWannierizer(
            params=params, evals=rng.standard_normal((nk, nb)),
            evecs=evecs, kpts=monkhorst_pack([2, 2, 2]),
            kweights=np.full(nk, 1.0 / nk),
        )
        proj = np.zeros((nw, nb), dtype=complex)
        proj[0, 0] = 1.0
        proj[1, 1] = 1.0
        wann.set_projectors(proj)
        return wann

    def test_get_amn_refines_and_attaches_spreads(self):
        wann = self._make()
        Amn = wann.get_Amn()
        assert wann.spreads is not None
        for key in ("omega", "omega_I", "omega_D", "omega_OD", "rbar",
                    "omega_n"):
            assert key in wann.spreads
        # gauge refinement keeps Amn columns orthonormal (right unitary action)
        for ik in range(Amn.shape[0]):
            np.testing.assert_allclose(
                Amn[ik].conj().T @ Amn[ik], np.eye(Amn.shape[2]), atol=1e-10)
        # monotone history
        om = [h["omega"] for h in wann.mlwf_history]
        assert all(om[i + 1] <= om[i] + 1e-12 for i in range(len(om) - 1))
        assert wann.spreads["omega"] <= om[0] + 1e-12

    def test_k_to_r_attaches_diagnostics(self):
        wann = self._make()
        wann.get_Amn()
        wann.get_wannk_and_Hk()
        Rlist = np.array([[0, 0, 0], [1, 0, 0]])
        Rdeg = np.array([1.0, 1.0])
        lwf = wann.k_to_R(Rlist=Rlist, Rdeg=Rdeg)
        assert lwf.spreads is not None
        np.testing.assert_allclose(lwf.wann_centers, wann.spreads["rbar"],
                                   atol=1e-12)
        assert lwf.spreads["omega"] == pytest.approx(wann.spreads["omega"])
