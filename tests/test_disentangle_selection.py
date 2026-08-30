"""Story-029 tests: disentanglement selection core (ADR-002/005).

Synthetic ring models: psi_k = expm(i phi_k H) with block-structured
Hermitian H, so Mmn from compute_Mmn is exactly unitary and
pair-symmetric. The sigma_x block {1,2} rotating by +/- pi/4 between ring
neighbours makes Z proportional to the identity on the candidate block
(a true overlap tie -> settled by the stable energy-then-index snap,
recorded in z_gap; raises only if the tie accompanies a stall); the
sigma_z variant (phase-only rotation) is distinguishable and resolves
deterministically.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from lawaf.io.w90 import compute_Mmn
from lawaf.wannierization.disentangle import (
    DegenerateSelectionError,
    DisentanglementError,
    InfeasibleWindowError,
    SelectionResult,
    select_subspace,
    SingularOverlapError,
)

SUPPORT = "lawaf.wannierization.disentangle"


def ring(nk):
    kpts = np.array([[n / nk, 0.0, 0.0] for n in range(nk)])
    nnlist = np.array([[(k + 1) % nk, (k - 1) % nk] for k in range(nk)])
    nncell = np.array([[[1, 0, 0], [-1, 0, 0]] for _ in range(nk)])
    return kpts, nnlist, nncell


def block_h(nband, block, rate):
    """Hermitian generator: `rate` * sigma_{x|z} on `block`, zero elsewhere."""
    H = np.zeros((nband, nband), dtype=complex)
    i, j = block
    if abs(rate) > 0:
        pauli = np.array([[0, 1], [1, 0]], dtype=complex)
        H[np.ix_([i, j], [i, j])] = rate * pauli
    return H


def psi_ring(nk, nband, H):
    """psi_k = expm(i * k * H): constant per-step rotation around the ring."""
    psi = np.zeros((nk, nband, nband), dtype=complex)
    for k in range(nk):
        psi[k] = expm(1j * k * H)
    return psi


@pytest.fixture
def tie_model():
    """sigma_x(pi/4) on {1,2}: Z on the candidate block is exactly c*I."""
    nk, nband = 4, 4
    kpts, nnlist, nncell = ring(nk)
    H = block_h(nband, (1, 2), np.pi / 4)
    mmn = compute_Mmn(psi_ring(nk, nband, H), nnlist, nncell)
    wb = np.array([0.5, 0.5])
    return mmn, nnlist, wb, nk, nband


@pytest.fixture
def phase_model():
    """Band 1 phase-only (perfect transport, Z=1); block {2,3} sigma_x
    pi/4 (Z = 1/2 I): the overlap criterion cleanly prefers band 1."""
    nk, nband = 4, 4
    kpts, nnlist, nncell = ring(nk)
    H = block_h(nband, (2, 3), np.pi / 4)
    H[1, 1] = 0.3  # pure phase on band 1
    mmn = compute_Mmn(psi_ring(nk, nband, H), nnlist, nncell)
    wb = np.array([0.5, 0.5])
    return mmn, nnlist, wb, nk, nband


class TestInfeasibleWindows:
    def test_frozen_larger_than_nwann(self, tie_model):
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [(0, 1, 2)] * nk
        feasible = [(0, 1, 2, 3)] * nk
        with pytest.raises(InfeasibleWindowError, match="frozen"):
            select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)

    def test_feasible_too_tight(self, tie_model):
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [()] * nk
        feasible = [(0,)] * nk
        with pytest.raises(InfeasibleWindowError, match="feasible"):
            select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)


class TestDegeneracySafety:
    def test_exact_overlap_tie_settles(self, tie_model):
        """1 free slot, candidates {1,2} with Z proportional to I: the
        overlap criterion cannot arbitrate, but the tie is
        objective-indifferent by symmetry -> the stable energy-then-index
        pick settles deterministically and the tie stays visible
        (settle-or-raise policy, user-approved 2026-08-30)."""
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [(0,)] * nk
        feasible = [(0, 1, 2)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert isinstance(res, SelectionResult)
        assert res.z_gap.max() < 1e-10           # tie recorded, not hidden
        assert len(res.guidance["tie_snapped"]) == nk
        # deterministic energy-then-index free complement: band 1
        s = np.linalg.svd(
            res.U_sel[0].conj().T @ np.eye(nband)[:, [0, 1]],
            compute_uv=False,
        )
        assert s.min() > 1 - 1e-10
        r2 = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert np.array_equal(res.U_sel, r2.U_sel)   # bitwise repeat

    def _inverter_ring_mmn(self, nk, observer=False):
        """diag(1, sigma_x) 'inverter' overlaps on a ring: every k's top
        Z pick is the SWAP of its neighbours' pick, so the selection
        oscillates with period 2 and never settles. With ``observer``,
        an extra k whose Z = P + sigma_x P sigma_x = I on the candidate
        block is appended: a permanent exact tie."""
        nband = 3
        n_k = nk + 1 if observer else nk
        mmn = np.zeros((n_k, 2, nband, nband), dtype=complex)
        swap = np.diag([1.0, 0.0, 0.0])
        swap[np.ix_([1, 2], [1, 2])] = np.array([[0, 1], [1, 0]])
        for ik in range(nk):
            mmn[ik, 0] = swap
            mmn[ik, 1] = swap
        nnlist = np.array(
            [[(k + 1) % nk, (k - 1) % nk] for k in range(nk)]
            + ([[0, 0]] if observer else [])
        )
        if observer:
            mmn[nk, 0] = swap       # P + sigma_x P sigma_x = I: exact tie
            mmn[nk, 1] = np.eye(nband)
        wb = np.array([1.0, 1.0])
        return mmn, nnlist, wb, n_k, nband

    def test_stall_without_tie_raises(self):
        """Period-2 oscillation, healthy gaps everywhere: the fixed point
        does not settle -> DisentanglementError, NOT the tie subclass."""
        mmn, nnlist, wb, n_k, nband = self._inverter_ring_mmn(4)
        frozen = [(0,)] * n_k
        feasible = [(0, 1, 2)] * n_k
        with pytest.raises(DisentanglementError) as ei:
            select_subspace(
                mmn, nnlist, wb, feasible, frozen, nwann=2, max_iter=12)
        assert not isinstance(ei.value, DegenerateSelectionError)
        assert "did not settle" in str(ei.value)

    def test_stall_with_tie_raises_degenerate(self):
        """Same oscillation plus a permanently tied observer k: the tie
        is implicated in the stall -> DegenerateSelectionError with the
        k-point context."""
        mmn, nnlist, wb, n_k, nband = self._inverter_ring_mmn(
            4, observer=True)
        frozen = [(0,)] * n_k
        feasible = [(0, 1, 2)] * n_k
        with pytest.raises(DegenerateSelectionError, match="k=4"):
            select_subspace(
                mmn, nnlist, wb, feasible, frozen, nwann=2, max_iter=12)

    def test_whole_block_included_no_error(self, tie_model):
        """Both members feasible with 2 free slots: forced whole-block
        inclusion, no silent split, no error."""
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [(0,)] * nk
        feasible = [(0, 1, 2)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=3)
        sel = res.U_sel[0]
        # frozen column first, both block members present
        s = np.linalg.svd(
            res.U_sel[0].conj().T @ np.eye(nband)[:, [1, 2, 0]],
            compute_uv=False,
        )
        assert s.min() > 1 - 1e-10

    def test_distinguishable_tie_resolves_deterministically(self, phase_model):
        mmn, nnlist, wb, nk, nband = phase_model
        frozen = [(0,)] * nk
        feasible = [(0, 1, 2)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert isinstance(res, SelectionResult)
        assert res.z_gap.min() > 1e-6
        # selected subspace carries band 1 (static, phase-only rotation)
        s = np.linalg.svd(
            res.U_sel[0].conj().T @ np.eye(nband)[:, [1]], compute_uv=False
        )
        assert s.min() > 1 - 1e-8


class TestSingularOverlap:
    def test_collapsed_selection_raises(self):
        """Forced selection whose neighbour overlaps vanish -> named error
        (the 4e-18 pathology becomes explicit at selection time)."""
        nk, nband = 4, 4
        kpts, nnlist, nncell = ring(nk)
        # rotate selected band 0 into UNSELECTED band 3: neighbour
        # subspace overlap vanishes exactly
        H = block_h(nband, (0, 3), np.pi / 2)
        mmn = compute_Mmn(psi_ring(nk, nband, H), nnlist, nncell)
        wb = np.array([0.5, 0.5])
        frozen = [()] * nk
        feasible = [(0, 1)] * nk
        with pytest.raises(SingularOverlapError, match="svd"):
            select_subspace(
                mmn, nnlist, wb, feasible, frozen, nwann=2, min_svd=1e-8
            )


class TestIsolatedManifold:
    def test_forced_selection_spans_kept_bands(self, tie_model):
        """|feasible| == nwann everywhere: U_sel is the identity on the
        kept bands (degenerate with the exclude_bands fast path)."""
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [()] * nk
        feasible = [(0, 1)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        expect = np.eye(nband)[:, [0, 1]]
        for ik in range(nk):
            assert np.allclose(res.U_sel[ik], expect, atol=1e-12)


class TestDeterminism:
    def test_bitwise_repeat(self, tie_model):
        mmn, nnlist, wb, nk, nband = tie_model
        frozen = [(0,)] * nk
        feasible = [(0, 1, 2, 3)] * nk
        r1 = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=3)
        r2 = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=3)
        assert np.array_equal(r1.U_sel, r2.U_sel)
        assert r1.n_iter == r2.n_iter


class TestResultContract:
    def test_fields_and_shapes(self, phase_model):
        mmn, nnlist, wb, nk, nband = phase_model
        nntot = nnlist.shape[1]
        frozen = [(0,)] * nk
        feasible = [(0, 1, 2, 3)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert res.U_sel.shape == (nk, nband, 2)
        assert len(res.feasible) == nk and len(res.frozen) == nk
        assert res.z_gap.shape == (nk,)
        assert res.pair_svd_min.shape == (nk, nntot)
        assert res.pair_svd_mean.shape == (nk, nntot)
        assert res.n_iter >= 1
        assert len(res.subspace_change_trace) == res.n_iter
        assert isinstance(res.omega_i_selection, float)
        assert isinstance(res.guidance, dict)

    def test_subspace_change_terminates(self, phase_model):
        mmn, nnlist, wb, nk, nband = phase_model
        frozen = [()] * nk
        feasible = [(0, 1, 2, 3)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert res.n_iter <= 5
        assert res.subspace_change_trace[-1] < res.subspace_change_trace[0] + 1e-15

    def test_parallel_forced_family_zero_omega_i(self):
        """Static psi (H = 0): Mmn = I, forced selection -> Omega_I = 0."""
        nk, nband = 4, 4
        kpts, nnlist, nncell = ring(nk)
        mmn = compute_Mmn(psi_ring(nk, nband, np.zeros((nband, nband))), nnlist, nncell)
        wb = np.array([0.5, 0.5])
        frozen = [()] * nk
        feasible = [(0, 1)] * nk
        res = select_subspace(mmn, nnlist, wb, feasible, frozen, nwann=2)
        assert res.omega_i_selection == pytest.approx(0.0, abs=1e-12)
        assert res.pair_svd_min.min() == pytest.approx(1.0, abs=1e-12)


def test_cut_block_walk_takes_first_overflowing_block():
    """Regression (review R1): with energy blocks of sizes (2, 2, 3) and
    nf=5, the CUT block is the 3-block, not a re-scan of the first
    2-block. The old re-scan arbitrated an already-consumed block and
    returned duplicate (non-orthonormal) columns."""
    nk, nband, nwann = 4, 8, 6
    kpts, nnlist, nncell = ring(nk)
    H = block_h(nband, (4, 5), np.pi / 4)  # rich mix inside the 3-block
    psi = psi_ring(nk, nband, H)
    mmn = compute_Mmn(psi, nnlist, nncell)
    wb = [0.5, 0.5]
    evals = np.tile(
        [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0], (nk, 1)
    )
    feasible = [tuple(range(nband))] * nk
    frozen = [(0,)] * nk
    res = select_subspace(
        mmn, nnlist, wb, feasible, frozen, nwann, eigvals=evals
    )
    for ik in range(nk):
        cols = res.U_sel[ik]
        gram = cols.conj().T @ cols
        assert np.abs(gram - np.eye(nwann)).max() < 1e-12, (
            f"k={ik}: selected columns not orthonormal"
        )
        support = np.abs(cols).argmax(axis=0)
        # frozen 0 + the two whole blocks {1,2} and {3,4} + exactly one
        # direction from the cut 3-block {5,6,7}
        assert 0 in support and {1, 2} <= set(support) \
            and {3, 4} <= set(support)
        extra = set(support) - {0, 1, 2, 3, 4}
        assert extra and extra <= {5, 6, 7}, support


def test_pinned_kpoint_empty_candidates_survive():
    """Regression (review R9): fully pinned k (feasible == frozen,
    empty candidate set) must not crash the energy-block construction."""
    nk, nband = 4, 4
    nwann = 3
    kpts, nnlist, nncell = ring(nk)
    H = block_h(nband, (1, 2), np.pi / 4)
    psi = psi_ring(nk, nband, H)
    mmn = compute_Mmn(psi, nnlist, nncell)
    wb = [0.5, 0.5]
    evals = np.tile([0.0, 1.0, 2.0, 3.0], (nk, 1))
    feasible = [tuple(range(nwann))] * nk
    frozen = [tuple(range(nwann))] * nk  # everything pinned
    res = select_subspace(
        mmn, nnlist, wb, feasible, frozen, nwann, eigvals=evals
    )
    assert res.n_iter >= 1
    for ik in range(nk):
        assert np.abs(res.U_sel[ik] - np.eye(nband)[:, :nwann]).max() < 1e-12
