"""Disentanglement / continuity-preserving subspace selection.

Architecture (disentanglement ADR-001/002/005, approved 2026-08-30);
research memo ``specs/research/2026-08-30-disentanglement-fixed-point.md``
(Option C). Roles fixed by prototype evidence:

- energy windows carry the CONTENT: per-k feasible sets and frozen cores
  (resolved by :func:`resolve_windows`, story-030);
- the overlap fixed point ARBITRATES: it resolves degenerate-partner and
  feasibility ambiguities within those sets (w90 ``internal_zmatrix``
  correspondence: ``Z_k = sum_b w_b A_b A_b^dag``, ``A_b = M(k,b) U(k+b)``)
  and is never run with unrestricted candidates (memo F3/F7 drift).

The stage is pure numpy/scipy on the shared overlap core
(:func:`lawaf.io.w90.compute_Mmn` outputs); it never imports the MLWF
optimizer (FR-009 standalone reuse).
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "DisentanglementError",
    "InfeasibleWindowError",
    "DegenerateSelectionError",
    "SingularOverlapError",
    "SelectionResult",
    "resolve_windows",
]

#: Residual per-pass subspace change below which budget exhaustion
#: counts as the slow-descent tail (w90 ``dis_num_iter`` semantics,
#: amended PRD success 1 "terminates by tol, budget, or genuine local
#: minimum") rather than a stall. BaTiO3 4x4x4 plateaus at ~4e-6.

#: Eigenvalue clusters within this RELATIVE width of the restricted-Z
#: block scale are treated as tied for the energy-then-index snap:
#: symmetry-exact doublets cut at the rank have noise-level gaps that
#: can exceed ``tol`` while their eigh eigenvectors stay noise-driven
#: (BaTiO3 4x4x4 R corner: gap ~1e-13, collapsed svd 3e-17 without the
#: cluster snap).
_TIE_RTOL = 1e-8
_SLOW_TAIL_CHANGE = 1e-4  # default budget-acceptance threshold


class DisentanglementError(Exception):
    """Base class for selection-stage failures (FR-003/004)."""


class InfeasibleWindowError(DisentanglementError):
    """A per-k window cannot host the requested selection."""


class DegenerateSelectionError(DisentanglementError):
    """An overlap tie is implicated in a selection stall.

    Settled ties (the fixed point converges despite a sub-``tol``
    deciding Z-gap at some k) are accepted: the choice is
    objective-indifferent by symmetry, resolved by the stable
    energy-then-index rule, and recorded in ``SelectionResult.z_gap`` /
    ``guidance['tie_snapped']``. This error fires only when the fixed
    point still has O(1) subspace change at budget and a sub-``tol`` tie
    at the selection rank is present (user-approved settle-or-raise
    policy, 2026-08-30).
    """


class SingularOverlapError(DisentanglementError):
    """Post-selection neighbour overlaps are singular (FR-003).

    The 4e-18 gauged-collapse pathology of the pre-selection window path
    becomes an explicit error at selection time, before any gauge work.
    """


@dataclass(frozen=True)
class SelectionResult:
    """Diagnostics contract of the selection stage (FR-007, ADR-005)."""

    #: (nk, nband, nwann) orthonormal band-space columns; selected psi is
    #: ``psi(k) @ U_sel[k]``.
    U_sel: np.ndarray
    #: per-k feasible band indices actually used
    feasible: tuple[tuple[int, ...], ...]
    #: per-k frozen band indices actually used
    frozen: tuple[tuple[int, ...], ...]
    #: fixed-point passes executed
    n_iter: int
    #: per-iteration max subspace change (termination metric)
    subspace_change_trace: tuple[float, ...]
    #: per-k Z-eigenvalue gap at the selection rank (inf where forced)
    z_gap: np.ndarray
    #: per-pair minimum cross-overlap singular value, shape (nk, nntot)
    pair_svd_min: np.ndarray
    #: per-pair mean cross-overlap singular value, shape (nk, nntot)
    pair_svd_mean: np.ndarray
    #: Omega_I contribution of the selection (nk-normalized, Mmn form)
    omega_i_selection: float
    #: which guidance sources shaped the sets (filled by callers)
    guidance: Mapping = field(default_factory=dict)


def _validate_sets(feasible, frozen, nk, nband, nwann):
    """Pre-iteration feasibility checks (ADR-002); all errors carry k."""
    if len(feasible) != nk or len(frozen) != nk:
        raise InfeasibleWindowError(
            f"window sets must cover all {nk} k-points "
            f"(got {len(feasible)}/{len(frozen)})"
        )
    out_f, out_c = [], []
    for ik in range(nk):
        f = tuple(sorted(set(int(i) for i in feasible[ik])))
        c = tuple(sorted(set(int(i) for i in frozen[ik])))
        if any(i < 0 or i >= nband for i in f):
            raise InfeasibleWindowError(
                f"k={ik}: feasible index out of range: {f}"
            )
        if not set(c) <= set(f):
            raise InfeasibleWindowError(
                f"k={ik}: frozen {c} not a subset of feasible {f}"
            )
        if len(c) > nwann:
            raise InfeasibleWindowError(
                f"k={ik}: frozen core has {len(c)} bands > nwann={nwann}"
            )
        if len(set(f) - set(c)) < nwann - len(c):
            raise InfeasibleWindowError(
                f"k={ik}: feasible\\frozen holds {len(set(f) - set(c))} "
                f"bands < {nwann - len(c)} free slots"
            )
        out_f.append(f)
        out_c.append(c)
    return tuple(out_f), tuple(out_c)


def _energy_init(blocks, nf):
    """Whole energy blocks in (energy, index) order until nf slots.

    Returns ``(taken, cut)``: ``taken`` = whole blocks that fit,
    ``cut`` = the FIRST block that would overflow (None when the
    candidates are exhausted exactly)."""
    picked = []
    for blk in blocks:
        if len(picked) + len(blk) > nf:
            return picked, blk
        picked.extend(blk)
    return picked, None


def _energy_pick_init(blocks, nf):
    """Full-rank initialization: the fitting prefix plus the stable
    energy-then-index snap of the cut block (first r by index)."""
    taken, cut = _energy_init(blocks, nf)
    if cut is not None:
        r = nf - len(taken)
        taken = taken + list(cut[:r])
    return taken


def _energy_block_pick(
    Z_k, blocks, nf, nband, tie_rtol, tol, snap_sink, gap_sink
):
    """Free complement with energy blocks carrying the content.

    ``blocks`` are per-k energy-degenerate candidate groups as GLOBAL
    band indices, ordered by (energy, index). Walk them taking whole
    blocks while they fit. The first block that would overflow
    (r = nf - taken < block size) is the CUT block: the overlap Z
    arbitrates inside it via its top-r restricted eigenvectors (snapped
    to the stable energy-then-index order on a noise-level tie). Bands
    beyond the cut block are never touched: the overlap fixed point
    cannot pull in non-degenerate content (memo F3/F6)."""
    taken, cut = _energy_init(blocks, nf)
    if cut is None:  # candidates exhausted exactly
        return _cols_matrix(sorted(taken), nband)
    r = nf - len(taken)
    if r <= 0:  # cannot happen (the cut block overflowed); defensive
        return _cols_matrix(sorted(taken), nband)
    cb = np.array(cut, dtype=int)
    if r >= len(cb):  # block fits exactly
        return _cols_matrix(sorted(taken + list(cb)), nband)
    Zb = Z_k[np.ix_(cb, cb)]
    w, v = eigh(Zb)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    gap = float(w[r - 1] - w[r])
    gap_sink(gap)
    if gap < max(tol, tie_rtol * abs(w[0])):
        # Exact (or noise-level) objective tie inside the cut block:
        # objective-indifferent by symmetry; stable index order.
        snap_sink()
        chosen = sorted(cb[:r])
        return _cols_matrix(sorted(taken) + chosen, nband)
    # healthy arbitration gap: top-r restricted eigenvectors (mixing
    # within an energy-degenerate block is legitimate gauge content)
    out = np.zeros((nband, nf), dtype=complex)
    out[sorted(taken), np.arange(len(taken))] = 1.0
    out[np.ix_(cb, np.arange(len(taken), nf))] = v[:, :r]
    return out


def _cols_matrix(cols, nband):
    m = np.zeros((nband, len(cols)), dtype=complex)
    for j, b in enumerate(cols):
        m[b, j] = 1.0
    return m


def _energy_blocks(vals, rtol=1e-5):
    """Contiguous (energy-sorted) degenerate blocks: |v_i - v_j| <=
    rtol * max(|v|) within a block (legality-checker semantics).
    Empty input yields no blocks (fully pinned / fully frozen k)."""
    if len(vals) == 0:
        return []
    order = np.argsort(vals, kind="stable")
    blocks, start = [], 0
    scale = max(float(np.abs(vals).max()), 1e-300)
    for i in range(1, len(order) + 1):
        if i == len(order) or abs(vals[order[i]] - vals[order[i - 1]]) > rtol * scale:
            blocks.append(tuple(int(b) for b in order[start:i]))
            start = i
    return blocks


def select_subspace(
    mmn: np.ndarray,
    nnlist: np.ndarray,
    wb: Sequence[float],
    feasible: Sequence[Sequence[int]],
    frozen: Sequence[Sequence[int]],
    nwann: int,
    eigvals: Optional[np.ndarray] = None,
    mix_ratio: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-10,
    min_svd: float = 1e-8,
    slow_tail_change: float = _SLOW_TAIL_CHANGE,
) -> SelectionResult:
    """Overlap-guided subspace selection on window feasible sets (ADR-002).

    Fixed point on the raw full-manifold overlaps (w90
    ``internal_zmatrix`` correspondence)::

        Z_k = sum_b w_b A_b A_b^dag,   A_b = M(k, b) S(k + b)
        Z <- mix_ratio * Z_new + (1 - mix_ratio) * Z
        S(k) = [frozen columns | free complement]

    With ``eigvals`` given (ADR-002 "windows carry content"), the free
    complement is fixed by ENERGY ORDER (bands are energy-sorted per k);
    the Z fixed point arbitrates ONLY inside energy-degenerate blocks
    that the selection rank cuts (top-r eigvecs of Z restricted to the
    cut block). Without ``eigvals`` the Z fixed point ranks the whole
    feasible\frozen candidate set (synthetic models).

    Terminates on subspace change (max_k 1 - min svd(S_old^dag S_new) <
    ``tol``), NOT on Delta Omega_I. Exact or noise-level overlap ties at
    the selection rank (gap < ``max(tol, _TIE_RTOL * |w_max|)``) do not
    raise: the free complement snaps to the stable energy-then-index
    pick (bands are energy-sorted per k), which is deterministic and
    iteration-stable; the tie stays visible in ``z_gap`` and
    ``guidance`` (settle-or-raise policy, user-approved 2026-08-30).
    Budget exhaustion with residual change below ``slow_tail_change``
    is the w90-style slow descent tail and is accepted (amended PRD
    success 1) with ``guidance['budget_exhausted'] = True``.

    :param mmn: (nk, nntot, nband, nband) raw overlaps M^{k,b}
        (:func:`lawaf.io.w90.compute_Mmn` on the full manifold);
    :param nnlist: (nk, nntot) neighbour indices (kmesh_nnlist);
    :param wb: (nntot,) neighbour weights;
    :param feasible: per-k feasible band indices (outer window / pins);
    :param frozen: per-k frozen band indices (inner window / pins),
        subset of feasible, at most ``nwann`` per k;
    :param nwann: selected subspace dimension;
    :param eigvals: optional (nk, nband) band energies in builder
        units; when given, energy-degenerate blocks carry the content
        and Z arbitrates only cut blocks (memo F3/F6: overlap-only
        ranking drifts to smooth-but-wrong content);
    :param mix_ratio: damped-update mixing (w90 ``dis_mix_ratio``);
    :param max_iter: pass cap (windows are near-fixed points; large
        models can show a slow tail accepted at budget);
    :param tol: subspace-change convergence tolerance; also the
        Z-eigen-gap threshold below which a tie is flagged for the
        stall error;
    :param min_svd: post-selection neighbour svd health threshold.

    :raises InfeasibleWindowError: before iterating, with k context;
    :raises DegenerateSelectionError: an overlap tie is implicated in a
        stall (O(1) subspace change at budget);
    :raises DisentanglementError: O(1) subspace change at budget with
        no tie implicated (oscillating selection);
    :raises SingularOverlapError: collapsed post-selection overlaps.
    """
    mmn = np.asarray(mmn)
    nnlist = np.asarray(nnlist)
    wb = np.asarray(wb, dtype=float)
    nk, nntot, nband, _ = mmn.shape
    if not 0.0 < mix_ratio <= 1.0:
        raise ValueError(f"mix_ratio must be in (0, 1], got {mix_ratio}")
    if nnlist.shape != (nk, nntot):
        raise ValueError(f"nnlist shape {nnlist.shape} != {(nk, nntot)}")

    feasible, frozen = _validate_sets(feasible, frozen, nk, nband, nwann)
    if eigvals is not None:
        eigvals = np.asarray(eigvals, dtype=float)
        if eigvals.shape != (nk, nband):
            raise ValueError(
                f"eigvals shape {eigvals.shape} != {(nk, nband)}"
            )

    def band_cols(indices):
        return np.eye(nband)[:, indices]

    # candidate (feasible \ frozen) index arrays; per-k energy blocks
    # of the candidates when eigvals are given. init = first nfree in
    # (energy, index) order -- the window content itself.
    cands, nfree, cand_blocks, S = [], [], [], []
    for ik in range(nk):
        c = np.array(sorted(set(feasible[ik]) - set(frozen[ik])), dtype=int)
        cands.append(c)
        nf = nwann - len(frozen[ik])
        nfree.append(nf)
        if eigvals is not None:
            cand_blocks.append(
                tuple(
                    tuple(int(c[j]) for j in blk)
                    for blk in _energy_blocks(eigvals[ik][c])
                )
            )
            init = _energy_pick_init(cand_blocks[-1], nf)
        else:
            cand_blocks.append(None)
            init = list(c[:nf])
        S.append(band_cols(list(frozen[ik]) + list(init)))

    Z = None
    z_gap = np.full(nk, np.inf)
    tie_snapped = set()
    trace = []
    budget_exhausted = False
    tie_snapped_final = set()
    for _it in range(max_iter):
        tie_snapped_final = set()
        Znew = np.zeros((nk, nband, nband), dtype=complex)
        for ik in range(nk):
            for ib in range(nntot):
                A = mmn[ik, ib] @ S[nnlist[ik, ib]]
                Znew[ik] += wb[ib] * (A @ A.conj().T)
        Z = Znew if Z is None else mix_ratio * Znew + (1 - mix_ratio) * Z

        change = 0.0
        S_new = []
        for ik in range(nk):
            c = cands[ik]
            nf = nfree[ik]
            if nf == 0:
                S_new.append(band_cols(list(frozen[ik])))
                continue
            if cand_blocks[ik] is not None and nf < len(c):
                free = _energy_block_pick(
                    Z[ik], cand_blocks[ik], nf, nband,
                    tie_rtol=_TIE_RTOL, tol=tol,
                    snap_sink=lambda ik=ik: (
                        tie_snapped.add(ik), tie_snapped_final.add(ik)),
                    gap_sink=lambda g: z_gap.__setitem__(ik, g),
                )
            else:
                Zc = Z[ik][np.ix_(c, c)]
                w, v = eigh(Zc)
                order = np.argsort(w)[::-1]
                w, v = w[order], v[:, order]
                if len(c) > nf:
                    gap = float(w[nf - 1] - w[nf])
                    z_gap[ik] = gap
                    if gap < max(tol, _TIE_RTOL * abs(w[0])):
                        # Exact (or noise-level) objective tie: the overlap
                        # criterion cannot arbitrate, but by symmetry the
                        # choice is objective-indifferent — snap to the
                        # stable energy-then-index pick instead of eigh's
                        # noise-level eigenvectors (settle-or-raise).
                        tie_snapped.add(ik)
                        tie_snapped_final.add(ik)
                        free = band_cols(list(c[:nf]))
                    else:
                        free = np.zeros((nband, nf), dtype=complex)
                        free[c, :] = v[:, :nf]
                else:  # forced: every candidate taken
                    free = band_cols(list(c))
            Snew = np.column_stack([band_cols(list(frozen[ik])), free])
            sv = np.linalg.svd(S[ik].conj().T @ Snew, compute_uv=False)
            change = max(change, float(1.0 - sv.min()))
            S_new.append(Snew)
        S = S_new
        trace.append(change)
        if change < tol:
            break
    else:
        head = (
            f"selection fixed point did not settle in {max_iter} passes "
            f"(last subspace change {trace[-1]:.3e}); "
        )
        if trace[-1] < slow_tail_change:
            # w90-style slow descent tail (amended PRD success 1 accepts
            # budget termination); record and proceed to the svd health
            # checks below.
            budget_exhausted = True
        else:
            # settle-or-raise: only a tie in the FINAL pass (or a
            # final gap below tol) implicates degeneracy in the stall;
            # a momentary noise snap mid-iteration does not
            tied = [
                int(ik) for ik in range(nk)
                if ik in tie_snapped_final or z_gap[ik] < tol
            ]
            if tied:
                k0 = tied[0]
                raise DegenerateSelectionError(
                    head
                    + f"k={k0}: overlap tie at the selection rank "
                    f"(final Z-eigen-gap {z_gap[k0]:.3e}; ties are gaps "
                    f"below max(tol, {_TIE_RTOL:.0e}*|lambda_max|), tol "
                    f"= {tol:.1e}); "
                    "widen the window to include the whole degenerate "
                    "block or pin it explicitly"
                )
            raise DisentanglementError(head + "oscillating selection")

    # post-selection neighbour overlap spectrum (FR-003/007)
    svd_min = np.zeros((nk, nntot))
    svd_mean = np.zeros((nk, nntot))
    omega_acc = 0.0
    for ik in range(nk):
        for ib in range(nntot):
            c = S[ik].conj().T @ mmn[ik, ib] @ S[nnlist[ik, ib]]
            sv = np.linalg.svd(c, compute_uv=False)
            svd_min[ik, ib] = sv.min()
            svd_mean[ik, ib] = sv.mean()
            omega_acc += wb[ib] * float(np.sum(np.abs(c) ** 2))
    if svd_min.min() < min_svd:
        ik, ib = np.unravel_index(np.argmin(svd_min), svd_min.shape)
        raise SingularOverlapError(
            f"post-selection neighbour svd {svd_min[ik, ib]:.3e} < "
            f"{min_svd:.1e} at k={ik}, neighbour {ib} "
            f"({nnlist[ik, ib]}): the selected bundle cannot carry a "
            "well-conditioned gauge; check the window content"
        )
    omega_i = nwann * float(wb.sum()) - omega_acc / nk

    return SelectionResult(
        U_sel=np.array(S),
        feasible=feasible,
        frozen=frozen,
        n_iter=len(trace),
        subspace_change_trace=tuple(trace),
        z_gap=z_gap,
        pair_svd_min=svd_min,
        pair_svd_mean=svd_mean,
        omega_i_selection=float(omega_i),
        guidance={
            "tie_snapped": tuple(sorted(tie_snapped)),
            "budget_exhausted": budget_exhausted,
        },
    )


def _block_desc(blk, ibands):
    """Human-readable degenerate block: retained rows and (when mapped)
    original band indices."""
    orig = [ibands[b] if ibands is not None and b < len(ibands) else b for b in blk]
    return f"rows {list(blk)} (bands {orig})"


def resolve_windows(
    kpts: np.ndarray,
    eigvals: np.ndarray,
    nwann: int,
    window_bands=None,
    win_min: Optional[float] = None,
    win_max: Optional[float] = None,
    froz_min: Optional[float] = None,
    froz_max: Optional[float] = None,
    ibands: Optional[Sequence[int]] = None,
    degeneracy_rtol: float = 1e-5,
):
    """Resolve window guidance into per-k feasible/frozen band sets (ADR-003).

    Precedence per k: per-q ``window_bands`` pin (frozen = feasible =
    the pinned set) > inner energy window (frozen) > outer energy window
    (feasible). Energy bounds compare against ``eigvals`` in the
    builder's own units (phonopy path: ``freqs_to_evals``-converted
    values); pin indices are ORIGINAL band indices mapped through
    ``ibands`` (retained rows), exactly like
    ``_apply_window_band_weights``.

    :param kpts: (nk, 3) mesh k-points;
    :param eigvals: (nk, nband) retained-band eigenvalues (engine units,
        excluded bands already removed — same rows as ``ibands``);
    :param nwann: target subspace dimension;
    :param window_bands: validated star-expanded object with a ``bands``
        dict {q tuple: original band indices} (story-025 machinery);
    :param win_min/win_max: outer (feasible) energy interval;
    :param froz_min/froz_max: inner (frozen) energy interval;
    :param ibands: retained original-band indices (default: all).

    :returns: ``(feasible, frozen)`` tuples of per-k index tuples over
        retained rows.
    :raises InfeasibleWindowError: oversized frozen core, an inner
        window edge that splits a degenerate block beyond the frozen
        budget, or a feasible set that cannot host the free complement
        (with k context).
    :raises ValueError: pin problems (off-mesh q, excluded band, wrong
        size) — mirrors ``_apply_window_band_weights`` messages.
    """
    from .wannierizer import _window_kkey, _window_kname

    kpts = np.asarray(kpts, dtype=float)
    eigvals = np.asarray(eigvals, dtype=float)
    nk, nband = eigvals.shape
    if ibands is None:
        ibands = tuple(range(nband))
    retained_rows = {int(b): row for row, b in enumerate(ibands)}

    # pinned rows per mesh index (exact-key match, story-025 semantics)
    pinned = {}
    if window_bands is not None:
        bands_by_q = getattr(window_bands, "bands", None)
        if bands_by_q is None:
            raise ValueError(
                "window_bands must be symmetry-validated by a compatible "
                "downfolder before Wannierization"
            )
        mesh_indices = {}
        for ik, kpoint in enumerate(kpts):
            key = _window_kkey(kpoint)
            if key in mesh_indices:
                raise ValueError(
                    f"duplicate reciprocal k-point {_window_kname(key)} "
                    "in k-point mesh"
                )
            mesh_indices[key] = ik
        for qpoint, selected in bands_by_q.items():
            key = _window_kkey(qpoint)
            if key not in mesh_indices:
                raise ValueError(
                    f"window star arm {_window_kname(key)} is not present "
                    "in the k-point mesh"
                )
            selected = tuple(selected)
            if len(selected) != nwann:
                raise ValueError(
                    f"window at {_window_kname(key)} selects "
                    f"{len(selected)} bands; expected nwann={nwann}"
                )
            if len(set(selected)) != len(selected):
                raise ValueError(
                    f"window at {_window_kname(key)} repeats an "
                    "original-band index"
                )
            if any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or index not in retained_rows
                for index in selected
            ):
                raise ValueError(
                    f"window at {_window_kname(key)} contains an excluded "
                    "or invalid original-band index"
                )
            pinned[mesh_indices[key]] = tuple(
                retained_rows[index] for index in selected
            )

    feasible, frozen = [], []
    for ik in range(nk):
        if ik in pinned:
            rows = pinned[ik]
            feasible.append(rows)
            frozen.append(rows)
            continue
        ev = eigvals[ik]
        outer = (
            np.arange(nband)
            if win_min is None and win_max is None
            else np.where(
                (ev >= (win_min if win_min is not None else -np.inf))
                & (ev <= (win_max if win_max is not None else np.inf))
            )[0]
        )
        inner = (
            ()
            if froz_min is None and froz_max is None
            else tuple(
                np.where(
                    (ev >= (froz_min if froz_min is not None else -np.inf))
                    & (ev <= (froz_max if froz_max is not None else np.inf))
                )[0]
            )
        )
        feasible.append(tuple(int(i) for i in outer))
        frozen.append(tuple(int(i) for i in inner))

    # FR-004: a window edge must never split an energy-degenerate block
    # silently. Frozen edge: include the whole block if the frozen budget
    # allows, else raise a named error. Outer edge: include the whole block
    # (feasible growth is always safe; the free-complement walk in
    # select_subspace arbitrates a cut block by overlap inside the block).
    # Because eigvals are exactly star-degenerate on symmetry arms, the
    # decision is identical on every arm: star-consistent by construction.
    for ik in range(nk):
        if ik in pinned:
            continue
        ev = eigvals[ik]
        blocks = _energy_blocks(ev, degeneracy_rtol)
        froz = set(frozen[ik])
        for blk in blocks:
            inside = [b for b in blk if b in froz]
            if inside and len(inside) < len(blk):
                grown = froz | set(blk)
                if len(grown) > nwann:
                    raise InfeasibleWindowError(
                        f"k={ik} ({_window_kname(kpts[ik])}): inner window "
                        f"edge splits the degenerate block "
                        f"{_block_desc(blk, ibands)} and including the "
                        f"whole block needs {len(grown)} frozen bands > "
                        f"nwann={nwann}; move the inner edge or pin the "
                        "block explicitly"
                    )
                froz = grown
        frozen[ik] = tuple(sorted(froz))
        # frozen bands are always feasible (inner window takes
        # precedence over the outer interval, ADR-003)
        feas = set(feasible[ik]) | froz
        for blk in blocks:
            if any(b in feas for b in blk):
                feas |= set(blk)
        feasible[ik] = tuple(sorted(feas))

    # feasibility (pins already satisfy these by construction)
    for ik in range(nk):
        if len(frozen[ik]) > nwann:
            raise InfeasibleWindowError(
                f"k={ik} ({_window_kname(kpts[ik])}): inner window "
                f"freezes {len(frozen[ik])} bands > nwann={nwann}"
            )
        nfree = nwann - len(frozen[ik])
        if len(set(feasible[ik]) - set(frozen[ik])) < nfree:
            raise InfeasibleWindowError(
                f"k={ik} ({_window_kname(kpts[ik])}): feasible\\frozen "
                f"holds {len(set(feasible[ik]) - set(frozen[ik]))} bands "
                f"< {nfree} free slots"
            )
    return tuple(feasible), tuple(frozen)
