"""Wannier90-style Wigner-Seitz image assignment for k<->R transforms.

Faithful reimplementation of wannier90's ``ws_distance.F90``
(``ws_translate_dist`` / ``R_wz_sc``): for each pair of basis functions
(i, j) and each mesh-grid R vector, find the supercell image
``R + n * mp_grid`` that minimizes the Cartesian distance
``|R . a + r_j - r_i|``; all images degenerate within ``tol`` of the
minimum are collected (up to ``NDEGX``). Interpolation with these
assignments averages the phase over the degenerate images.

``apply_ws_distance`` materializes that semantics: matrix elements are
redistributed to their chosen images with weight ``1/ndeg``, producing a
plain ``(HR, Rlist)`` pair whose unweighted Fourier sum equals the W90
per-image phase average exactly.
"""
import itertools

import numpy as np

NDEGX = 8

def ws_translate_dist(Rlist, centers, cell, mp_grid,
                      search_size=(2, 2, 2), tol=1e-5, centers_j=None):

    """Find Wigner-Seitz supercell-image shifts per (R, i, j) pair.

    Find Wigner-Seitz supercell-image shifts per (R, i, j) pair.

    Single-pass variant of wannier90's two-pass ``R_wz_sc`` search: all
    near-minimal images are collected from one candidate window
    ``[-search_size-1, search_size+1]^3``. This matches the reference for
    the centers lawaf feeds it (fractional positions inside the home cell,
    so the shortest image never sits on the window boundary); it can
    differ from w90 for centers displaced by multiples of the supercell.

    Parameters
    Rlist : (nR, 3) int array
        Mesh-grid R vectors (unique modulo ``mp_grid``).
    centers : (n1, 3) float array
        Reduced (fractional) positions/centers of the row basis functions.
    centers_j : (n2, 3) float array, optional
        Centers of the column basis functions for rectangular tensors;
        defaults to ``centers`` (square tensors).
    cell : (3, 3) float array
        Rows are the primitive lattice vectors in Cartesian units; ``tol``
        is expressed in the same units.
    mp_grid : (3,) int
        The k-mesh defining the supercell whose images are searched.
    search_size : (3,) int, optional
        W90 ``ws_search_size`` (default 2, the wannier90 default): the
        image search runs over ``[-search_size-1, search_size+1]`` per
        axis.
    tol : float, optional
        W90 ``ws_distance_tol`` (default 1e-5, the wannier90 default):
        images within ``tol`` of the minimal distance are all treated as
        degenerate.

    Returns
    -------
    shifts : (nR, n1, n2, NDEGX, 3) int array
        Supercell-image shifts in primitive-cell units (multiples of
        ``mp_grid``); the effective R vector of image ``d`` of pair (i, j)
        at mesh vector ``R`` is ``R + shifts[R, i, j, d]``.
    ndeg : (nR, n1, n2) int array
        Number of degenerate images for each (R, i, j).
    """
    Rlist = np.asarray(Rlist, dtype=int)
    centers_i = np.asarray(centers, dtype=float)
    centers_j = (centers_i if centers_j is None
                 else np.asarray(centers_j, dtype=float))
    cell = np.asarray(cell, dtype=float)
    N = np.asarray(mp_grid, dtype=int)
    s = np.asarray(search_size, dtype=int)

    ranges = [range(-s[axis] - 1, s[axis] + 2) for axis in range(3)]
    ncand = np.array(list(itertools.product(*ranges)), dtype=int)  # (C, 3)
    shifts_prim = ncand * N  # (C, 3) primitive-cell image shifts
    cand_cart = shifts_prim @ cell  # (C, 3)

    nR = Rlist.shape[0]
    n1 = centers_i.shape[0]
    n2 = centers_j.shape[0]
    # base[R, i, j, :] = R . a + (r_j - r_i) . a
    Rcart = Rlist @ cell  # (nR, 3)
    cji = centers_j[None, None, :, :] - centers_i[None, :, None, :]  # (1,n1,n2,3)
    base = Rcart[:, None, None, :] + cji @ cell  # (nR, n1, n2, 3)

    d2 = np.sum((base[..., None, :] + cand_cart) ** 2, axis=-1)  # (nR,n1,n2,C)
    dist = np.sqrt(d2)
    within = dist <= dist.min(axis=-1, keepdims=True) + tol
    ndeg = within.sum(axis=-1)  # (nR, n1, n2)
    if ndeg.max() > NDEGX:
        raise ValueError(
            f"ws_translate_dist: degeneracy {ndeg.max()} exceeds NDEGX={NDEGX}")

    # stable argsort on the negated mask: True-first in candidate order
    flat = within.reshape(-1, within.shape[-1])  # (M, C)
    order = np.argsort(~flat, axis=1, kind="stable")[:, :NDEGX]  # (M, NDEGX)
    shifts = shifts_prim[order].reshape(nR, n1, n2, NDEGX, 3)
    return shifts, ndeg


def apply_ws_distance(HR, Rlist, centers, cell, mp_grid,
                      centers_j=None, **ws_kwargs):
    """Materialize the W90 Wigner-Seitz assignment on an R-space tensor.

    ``Rlist`` must be unique modulo ``mp_grid`` (see :func:`fold_R_to_mesh`
    for converting lawaf's legacy extended grid). For each (i, j, R) and
    each of its ``ndeg`` degenerate images, the element ``HR[R, i, j] /
    ndeg`` is added to ``HR_ws[R + n_d*N, i, j]``. The unweighted Fourier
    sum of ``HR_ws`` equals wannier90's per-image phase average exactly
    (all degenerate images share the phase at mesh k-points).

    Works for any (nR, n1, n2) tensor (Hamiltonian, overlap, wannR, IFC);
    ``centers`` gives the n1 row-function centers and ``centers_j`` the n2
    column-function centers (defaults to ``centers``).

    Returns
    -------
    HR_ws : (U, n1, n2) array
        Materialized tensor on the union R grid.
    Rlist_ws : (U, 3) int array
        Sorted unique union of the image R vectors.
    Rdeg : (U,) float array
        All ones (kept for interface symmetry with the grid modes).
    """
    HR = np.asarray(HR)
    Rlist = np.asarray(Rlist, dtype=int)
    nR, n1, n2 = HR.shape
    shifts, ndeg = ws_translate_dist(Rlist, centers, cell, mp_grid,
                                     centers_j=centers_j, **ws_kwargs)

    d = np.arange(NDEGX)
    valid = d[None, None, None, :] < ndeg[..., None]  # (nR, n1, n2, NDEGX)
    targets = Rlist[:, None, None, None, :] + shifts  # (nR, n1, n2, NDEGX, 3)
    vals = np.broadcast_to((HR / ndeg)[..., None], (nR, n1, n2, NDEGX))

    t_all = targets[valid]  # (K, 3)
    v_all = vals[valid]  # (K,)
    Rlist_ws, inv = np.unique(t_all, axis=0, return_inverse=True)
    ii = np.broadcast_to(np.arange(n1)[None, :, None, None], valid.shape)[valid]
    jj = np.broadcast_to(np.arange(n2)[None, None, :, None], valid.shape)[valid]
    HR_ws = np.zeros((len(Rlist_ws), n1, n2), dtype=HR.dtype)
    np.add.at(HR_ws, (inv, ii, jj), v_all)
    return HR_ws, Rlist_ws, np.ones(len(Rlist_ws), dtype=float)


def apply_ws_distance_tensors(tensors, Rlist, centers_list, cell, mp_grid,
                              centers_j_list=None, **ws_kwargs):
    """Materialize several (nR, n1, n2) tensors onto a COMMON union R grid.

    Each tensor gets its own per-pair image assignment (its ``centers_list``
    entry, optionally paired with ``centers_j_list`` for rectangular
    tensors); the results are re-scattered onto the union of all target R
    vectors so that one Rlist/Rdeg pair serves the whole LWF/EWF object.

    Returns (list_of_tensors, Rlist_union, Rdeg_ones).
    """
    Rlist = np.asarray(Rlist, dtype=int)
    if centers_j_list is None:
        centers_j_list = [None] * len(tensors)
    results = [apply_ws_distance(T, Rlist, c, cell, mp_grid,
                                 centers_j=cj, **ws_kwargs)
               for T, c, cj in zip(tensors, centers_list, centers_j_list)]
    merged = np.unique(np.concatenate([r[1] for r in results]), axis=0)
    index = {tuple(R): i for i, R in enumerate(merged)}
    out = []
    for T, (T_ws, Rl_ws, _) in zip(tensors, results):
        U = np.zeros((len(merged),) + T.shape[1:], dtype=T.dtype)
        pos = np.array([index[tuple(R)] for R in Rl_ws])
        U[pos] = T_ws
        out.append(U)
    return out, merged, np.ones(len(merged), dtype=float)


def fold_R_to_mesh(Rlist, tensors, mp_grid, Rdeg=None):
    """Fold an extended (degeneracy-weighted) R grid onto the plain mesh grid.

    lawaf's legacy R grid spans the Wannier90-style extended cell
    [-N/2, N/2]^3 where equivalent R vectors (equal mod mp_grid) each carry
    the full DFT coefficient and a degeneracy weight. WS materialization
    needs one full coefficient per equivalence class: sum w * T over each
    class (all copies are equal, so this is exact). For odd meshes the
    extended grid already has unique representatives and this is a no-op
    up to ordering.

    Returns (tensors_folded, Rlist_folded) with Rlist_folded unique mod N.
    """
    Rlist = np.asarray(Rlist, dtype=int)
    N = np.asarray(mp_grid, dtype=int)
    if Rdeg is None:
        Rdeg = np.ones(len(Rlist))
    Rdeg = np.asarray(Rdeg, dtype=float)
    keys = [tuple(R % N) for R in Rlist]
    uniq = sorted(set(keys))
    index = {k: i for i, k in enumerate(uniq)}
    pos = np.array([index[k] for k in keys])
    Rlist_folded = np.array(uniq, dtype=int)
    out = []
    for T in tensors:
        T = np.asarray(T)
        F = np.zeros((len(uniq),) + T.shape[1:], dtype=T.dtype)
        np.add.at(F, pos, T * Rdeg[(...,) + (None,) * (T.ndim - 1)])
        out.append(F)
    return out, Rlist_folded
