# SPDX-License-Identifier: BSD-2-Clause
"""Wannier90 file format writers and helpers.

``kmesh_nnlist`` builds the nearest-neighbour k-point list wannier90 needs
to validate every ``.mmn`` block. It is an original implementation of the
published, documented neighbour construction for regular k meshes (the
B1 condition of Marzari & Vanderbilt, PRB 56, 12847 (1997), Appendix B,
and the tolerance semantics documented in the wannier90 user guide):
it derives shells and weights from that mathematical contract and is
behaviourally validated against wannier90-produced ``.nnkp`` files. No
wannier90 source code was translated, incorporated, or vendored.
Writers for ``.amn``/``.mmn``/``.eig``/``.win`` are built on top of it.

Conventions (see ADRs in the W90-export architecture note):
- reciprocal lattice rows are b1, b2, b3 including 2*pi (w90 convention);
- nnlist/nncell use 0-based k indices and shape (nkpt, nntot) / (nkpt, nntot, 3).
"""

import numpy as np

_NUM_NNMAX = 12  # w90-compatible hard cap on neighbours per k-point
_MAX_SUPERSHELL = 8  # adaptive completeness bound on the image search range


def _integer_images(nsup):
    """All integer triples (l, m, n) with |l|,|m|,|n| <= nsup, row-stacked."""
    r = np.arange(-nsup, nsup + 1)
    return np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T


def _smallest_excluded_image(recip_lattice, nsup):
    """Minimum |G| over images with some |component| > nsup.

    Used to certify shell completeness: any lattice image outside the
    searched box is at least this far from the origin.
    """
    cand = _integer_images(nsup + 1)
    inside = np.all(np.abs(cand) <= nsup, axis=1)
    outside = cand[~inside]
    return float(np.min(np.linalg.norm(outside @ recip_lattice, axis=1)))


def _cluster_shells(dists, kmesh_tol):
    """Group ascending distances into shells with a relative tolerance band.

    A distance belongs to the current shell while it is within
    ``representative * kmesh_tol`` of it; the representative is the first
    (smallest) distance of the shell.
    """
    shells = []  # list of (representative, [displacement indices])
    for i, d in enumerate(dists):
        if shells and d <= shells[-1][0] * (1.0 + kmesh_tol):
            shells[-1][1].append(i)
        else:
            shells.append((d, [i]))
    return shells


def _select_b1_shells(shell_bvecs, kmesh_tol):
    """Choose the minimal ascending set of shells solving Eq. (B1).

    B1 (PRB 56, 12847 (1997), App. B): find weights w_s with
    ``sum_s w_s * sum_{b in shell s} |b><b| = I``. Shells are tried in
    ascending distance order; a shell whose vectors span no direction
    beyond the already accepted set (all parallel to previous shells) is
    skipped, and selection stops at the first prefix whose least-squares
    weights reproduce the identity within the tolerance.
    """
    target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    accepted = []  # list of (n, 3) b-vector arrays
    amat_cols = []
    for bvec in shell_bvecs:
        norms = np.linalg.norm(bvec, axis=1)
        # skip shells adding no new direction (all vectors parallel to
        # some already-accepted vector)
        parallel = False
        for b, n in zip(bvec, norms):
            for old in np.concatenate(accepted) if accepted else []:
                cos = abs(float(np.dot(b, old))) / (
                    n * np.linalg.norm(old)
                )
                if 1.0 - cos < 1e-6:
                    parallel = True
                    break
            if parallel:
                break
        if parallel:
            continue
        col = np.zeros(6)
        for b in bvec:
            for c, (i, j) in enumerate(comp):
                col[c] += b[i] * b[j]
        trial = amat_cols + [col]
        amat = np.column_stack(trial)
        w, *_ = np.linalg.lstsq(amat, target, rcond=None)
        b1 = sum(
            w[s] * sum(np.outer(b, b) for b in bvs)
            for s, bvs in enumerate(accepted + [bvec])
        )
        if np.max(np.abs(b1 - np.eye(3))) <= kmesh_tol:
            return accepted + [bvec], w
        accepted.append(bvec)
        amat_cols.append(col)
    raise ValueError("kmesh: unable to satisfy B1 with searched shells")


def kmesh_nnlist(kpts, recip_lattice, kmesh_tol=1e-6):
    """Build wannier90-compatible nearest-neighbour k-point lists.

    Original implementation of the documented regular-mesh neighbour
    contract (Eq. B1 of PRB 56, 12847 (1997), Appendix B): shells of
    connecting vectors b are identified on the Cartesian metric, weights
    solve ``sum_b w_b |b><b| = I``, and each k-point's neighbour in
    direction b is the unique mesh point congruent to ``k + b`` modulo
    the reciprocal lattice. Behaviour is pinned to wannier90's ``.nnkp``
    output by the test suite.

    :param kpts: (nkpt, 3) fractional k-points (Gamma-centered mesh).
    :param recip_lattice: (3, 3) rows b1..b3, 2*pi included, same units as
        kmesh_tol (default 1e-6, w90 default, in 1/Angstrom).
    :returns: tuple ``(nnlist, nncell, bvecs, wb)``
        - nnlist (nkpt, nntot) int: 0-based neighbour k indices;
        - nncell (nkpt, nntot, 3) int: lattice image triple G of each pair
          (write 1-based k indices + G in the .mmn header);
        - bvecs (nntot, 3): b-vectors of k-point 1;
        - wb (nntot,): shell weights satisfying Eq. B1.
    """
    kpts = np.asarray(kpts, dtype=float)
    recip_lattice = np.asarray(recip_lattice, dtype=float)
    nkpt = len(kpts)
    kcart = kpts @ recip_lattice

    shells = None
    for nsup in range(2, _MAX_SUPERSHELL + 1):
        lmn = _integer_images(nsup)
        images = lmn @ recip_lattice
        dists_all = np.linalg.norm(
            kcart[None, :, :] + images[:, None, :] - kcart[0], axis=-1
        ).ravel()
        disp_all = (kcart[None, :, :] + images[:, None, :] - kcart[0]).reshape(
            -1, 3
        )
        jk_idx = np.tile(np.arange(nkpt), len(images))
        keep = dists_all > kmesh_tol  # drop the zero self-displacement
        disp, dists, jks = disp_all[keep], dists_all[keep], jk_idx[keep]
        # shell candidates in wannier90's enumeration: distance ascending,
        # lattice image lexicographically descending, k-index ascending
        pair_lmn = np.repeat(lmn, nkpt, axis=0)[keep]
        order = np.lexsort(
            (jks, -pair_lmn[:, 2], -pair_lmn[:, 1], -pair_lmn[:, 0], dists)
        )
        disp, dists = disp[order], dists[order]
        clustered = _cluster_shells(dists, kmesh_tol)
        try:
            shell_bvecs = [disp[idx] for _, idx in clustered]
            accepted, wshell = _select_b1_shells(shell_bvecs, kmesh_tol)
        except ValueError:
            continue
        # completeness: longest accepted shell must be inside the
        # provably-exhausted search radius
        longest = max(np.max(np.linalg.norm(bv, axis=1)) for bv in accepted)
        if longest * (1.0 + kmesh_tol) < _smallest_excluded_image(
            recip_lattice, nsup
        ):
            shells = accepted
            break
    if shells is None:
        raise ValueError(
            "kmesh: unable to find a complete shell set solving B1"
        )

    # column layout: accepted shells in ascending distance order; weights
    # aligned per slot below
    nntot = sum(len(bv) for bv in shells)
    if nntot > _NUM_NNMAX:
        raise ValueError("kmesh: found more than num_nnmax neighbours")
    shell_mult = [len(bv) for bv in shells]
    shell_reps = [
        float(np.median(np.linalg.norm(bv, axis=1))) for bv in shells
    ]
    shell_w = [w for _, w in zip(shells, wshell)]

    # global image iteration order: |G| ascending, ties in reversed
    # integer-image enumeration order (wannier90's observable neighbour
    # ordering in .nnkp output)
    flat = np.arange(len(images))
    img_dist = np.linalg.norm(images, axis=1)
    img_order = np.lexsort((-flat, img_dist))

    nnlist = np.zeros((nkpt, nntot), dtype=int)
    nncell = np.zeros((nkpt, nntot, 3), dtype=int)
    bk = np.zeros((nkpt, nntot, 3))
    wb = np.concatenate(
        [np.full(m, w) for m, w in zip(shell_mult, shell_w)]
    )
    for ik in range(nkpt):
        nnx = 0
        for rep, mult in zip(shell_reps, shell_mult):
            lo, hi = rep * (1.0 - kmesh_tol), rep * (1.0 + kmesh_tol)
            found = 0
            for img in img_order:
                d = np.linalg.norm(
                    kcart + images[img] - kcart[ik], axis=1
                )
                for jk in np.flatnonzero((d >= lo) & (d <= hi)):
                    if found == mult:
                        break
                    nnlist[ik, nnx] = jk
                    nncell[ik, nnx] = lmn[img]
                    bk[ik, nnx] = kcart[jk] + images[img] - kcart[ik]
                    nnx += 1
                    found += 1
                if found == mult:
                    break
            if found < mult:
                raise ValueError(
                    f"kmesh: too few neighbours found for k-point {ik}"
                )
    bvecs = bk[0].copy()

    # neighbour displacement lengths must reproduce the shell b-vectors
    if not np.allclose(
        np.linalg.norm(bk, axis=-1),
        np.linalg.norm(bvecs, axis=-1)[None, :],
        atol=kmesh_tol,
        rtol=0,
    ):
        raise ValueError("Non-symmetric k-point neighbours!")
    # Eq. (B1) must hold at every k-point with the actual displacements
    for ik in range(nkpt):
        b1 = np.einsum("n,ni,nj->ij", wb, bk[ik], bk[ik])
        if np.max(np.abs(b1 - np.eye(3))) > kmesh_tol:
            raise ValueError("Eq. (B1) not satisfied in kmesh_nnlist")
    return nnlist, nncell, bvecs, wb

def write_amn(amn, path, orthogonalize=False):
    """Write a wannier90 ``.amn`` file.

    :param amn: (nkpt, nband, nwann) complex projections
        A[m, n, k] = <g_n,k | psi_m,k>; written raw by default (w90
        orthonormalizes internally).
    :param orthogonalize: Loewdin-orthonormalize the columns per k-point
        (A (A^dag A)^-1/2) before writing.
    """
    amn = np.asarray(amn)
    nk, nb, nw = amn.shape
    if orthogonalize:
        out = np.empty(amn.shape, dtype=np.result_type(amn.dtype, np.float64))
        for ik in range(nk):
            a = amn[ik]
            gram = a.conj().T @ a
            w, v = np.linalg.eigh(gram)
            if w.min() <= 1e-12 * max(abs(w).max(), 1.0):
                raise ValueError(
                    f"write_amn: rank-deficient projection Gram at "
                    f"k-point {ik}; cannot Loewdin-orthogonalize"
                )
            out[ik] = a @ (v @ np.diag(w ** -0.5) @ v.conj().T)
        amn = out
    rows = []
    for ik in range(nk):
        for n in range(nw):
            for m in range(nb):
                rows.append(
                    f"{m + 1:5d} {n + 1:5d} {ik + 1:5d}"
                    f"{amn[ik, m, n].real:18.12f}{amn[ik, m, n].imag:18.12f}"
                )
    with open(path, "w") as fh:
        fh.write(" Created by lawaf\n")
        fh.write(f"{nb:5d} {nk:5d} {nw:5d}\n")
        fh.write("\n".join(rows) + "\n")


def write_eig(evals, path):
    """Write a wannier90 ``.eig`` file: rows ``band kpt e`` in strict
    k-outer / band-inner order (parameters.F90:1646-1648 enforces it)."""
    evals = np.asarray(evals)
    nk, nb = evals.shape
    with open(path, "w") as fh:
        for ik in range(nk):
            for n in range(nb):
                fh.write(f"{n + 1:5d} {ik + 1:5d} {evals[ik, n]:18.12f}\n")


def write_win(path, *, num_wann, num_bands, mp_grid, lattice, symbols,
              frac, kpts, projections=None, kmesh_tol=1e-6, extra=None):
    """Write a minimal wannier90 ``.win`` input.

    :param mp_grid: [nx, ny, nz] of the Gamma-centered mesh;
    :param lattice: (3, 3) rows a1..a3 in Angstrom (unit_cell_cart);
    :param symbols: atomic symbols, len nat;
    :param frac: (nat, 3) fractional positions (atoms_frac block);
    :param kpts: (nkpt, 3) fractional k-points in lawaf's own order —
        written as an explicit ``begin kpoints`` block so wannier90's
        k indices match lawaf's exactly;
    :param projections: None -> ``random`` (w90 picks random trial
        orbitals; use when exporting an external .amn), or a list of
        projection strings like ``["Mn:d"]`` (one per wannier function).
    """
    mp_grid = list(mp_grid)
    if len(kpts) != mp_grid[0] * mp_grid[1] * mp_grid[2]:
        raise ValueError("kpts length does not match mp_grid")
    lines = [f"num_wann = {num_wann}", f"num_bands = {num_bands}",
             f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}"]
    lines.append(f"kmesh_tol = {kmesh_tol}")
    lines += ["begin unit_cell_cart", "ang"]
    for row in np.asarray(lattice):
        lines.append("  {:20.12f}{:20.12f}{:20.12f}".format(*row))
    lines += ["end unit_cell_cart", "begin atoms_frac"]
    frac = np.asarray(frac)
    if frac.ndim != 2 or frac.shape[1] != 3 or len(symbols) != len(frac):
        raise ValueError(
            "write_win: symbols and frac must describe the same atoms "
            "with frac of shape (nat, 3)"
        )
    for sym, pos in zip(symbols, frac):
        lines.append(f"{sym:>3s}  {pos[0]:20.12f}{pos[1]:20.12f}{pos[2]:20.12f}")
    lines += ["end atoms_frac", "begin projections"]
    if projections is None:
        lines.append("random")
    else:
        # projection strings may expand to multiple orbitals (e.g. "Mn:d");
        # wannier90 itself validates the total count at parse time
        lines.extend(projections)
    lines.append("end projections")
    if extra:
        for key, value in extra.items():
            lines.append(f"{key} = {value}")
    lines.append("begin kpoints")
    for k in np.asarray(kpts):
        lines.append(f"  {k[0]:20.12f} {k[1]:20.12f} {k[2]:20.12f}")
    lines += ["end kpoints"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

def compute_Mmn(psi, nnlist, nncell):
    """Overlap matrices M^{k,b}_{mn} = <u_{m,k} | u_{n,k+b}>.

    :param psi: (nkpt, nbasis, nband) periodic-gauge states — COLUMNS are
        the band states (the layout of Wannierizer.get_psi_k, where
        nbasis may differ from nband); rows are the fixed orthonormal
        basis coefficients;
    :param nnlist: (nkpt, nntot) neighbour k indices (from kmesh_nnlist);
    :param nncell: (nkpt, nntot, 3) lattice image triples — with lawaf's
        integer-R (periodic-gauge) convention the image label does not
        enter the VALUE: M = psi_k^dag psi_jk, no extra phase;
    :returns: (nkpt, nntot, nband, nband) complex M[ik, ib, m, n].
    """
    psi = np.asarray(psi)
    nkpt, nntot = nnlist.shape
    nb = psi.shape[2]
    mmn = np.empty((nkpt, nntot, nb, nb), dtype=complex)
    for ik in range(nkpt):
        pk = psi[ik].conj().T
        for ib in range(nntot):
            mmn[ik, ib] = pk @ psi[nnlist[ik, ib]]
    return mmn


def write_mmn(mmn, nnlist, nncell, path):
    """Write a wannier90 ``.mmn`` file.

    Layout per overlap.F90:133-198: comment; ``nb nk nntot``; per (k,
    neighbour) block a header ``ik jk G1 G2 G3`` (1-based) followed by the
    matrix written n outer, m inner (M(m, n) row-major per column n).
    ``nnlist``/``nncell`` MUST be the arrays wannier90 will reconstruct
    itself, i.e. the output of :func:`kmesh_nnlist` for the same mesh —
    wannier90 matches every block by (jk, G) against its own kmesh_get.
    """
    mmn = np.asarray(mmn)
    nkpt, nntot, nb1, nb2 = mmn.shape
    if nb1 != nb2:
        raise ValueError(
            f"write_mmn: blocks must be square, got ({nb1}, {nb2})"
        )
    nb = nb1
    with open(path, "w") as fh:
        fh.write(" Created by lawaf\n")
        fh.write(f"{nb:5d} {nkpt:5d} {nntot:5d}\n")
        for ik in range(nkpt):
            for ib in range(nntot):
                jk = nnlist[ik, ib]
                g = nncell[ik, ib]
                # explicit delimiters: wannier90 reads list-directed
                fh.write(
                    f"{ik + 1:5d} {jk + 1:5d} "
                    f"{g[0]:5d} {g[1]:5d} {g[2]:5d}\n"
                )
                m = mmn[ik, ib]
                for n in range(nb):
                    for mi in range(nb):
                        fh.write(
                            f"{m[mi, n].real:18.12f}"
                            f"{m[mi, n].imag:18.12f}\n"
                        )

def read_mmn(path):
    """Read a wannier90 ``.mmn`` file (inverse of :func:`write_mmn`).

    Accepts the standard layout: one comment line, ``nb nk nntot``,
    then per (k, neighbour) block a header ``ik jk G1 G2 G3`` (1-based)
    followed by the matrix written n outer, m inner. Returns
    ``(mmn, nnlist, nncell)`` with lawaf's 0-based conventions.

    This is the explicit cross-k provider entry point for electron
    models: a caller holding a genuine ``.mmn`` from a wannier90 run
    can feed ``covariant_mmn_links``/MV diagnostics without LaWaF
    synthesizing links from HR/SR data.
    """
    with open(path) as fh:
        lines = [ln for ln in (l.strip() for l in fh) if ln]
    if len(lines) < 2:
        raise ValueError(f"read_mmn: {path} is empty")
    nb, nkpt, nntot = (int(t) for t in lines[1].split()[:3])
    if nb <= 0 or nkpt <= 0 or nntot <= 0:
        raise ValueError(f"read_mmn: invalid header {lines[1]!r}")
    expected = 2 + nkpt * nntot * (1 + nb * nb)
    if len(lines) != expected:
        raise ValueError(
            f"read_mmn: expected {expected} lines for nb={nb}, nk={nkpt}, "
            f"nntot={nntot}, got {len(lines)}"
        )
    mmn = np.empty((nkpt, nntot, nb, nb), dtype=complex)
    nnlist = np.empty((nkpt, nntot), dtype=int)
    nncell = np.empty((nkpt, nntot, 3), dtype=int)
    pos = 2
    for ik in range(nkpt):
        for ib in range(nntot):
            head = [int(t) for t in lines[pos].split()[:5]]
            pos += 1
            ik_f, jk_f, g1, g2, g3 = head
            if ik_f != ik + 1:
                raise ValueError(
                    f"read_mmn: block k index {ik_f} != {ik + 1} at "
 f"block ({ik}, {ib})"
                )
            nnlist[ik, ib] = jk_f - 1
            nncell[ik, ib] = (g1, g2, g3)
            block = np.empty((nb, nb), dtype=complex)
            for n in range(nb):
                for mi in range(nb):
                    re, im = (float(t) for t in lines[pos].split()[:2])
                    block[mi, n] = complex(re, im)
                    pos += 1
            mmn[ik, ib] = block
    return mmn, nnlist, nncell


# ---------------------------------------------------------------------------
# Streaming (bounded-memory) writers.
#
# The dense write_amn/write_mmn above stay for their existing callers and
# small reference smokes.  Producers that never hold a full mesh/pair tensor
# stream deterministic complete records through the *_blocks entry points
# below; they share this module's exact record grammar (" Created by lawaf"
# header, %5d counts, %18.12f fields, m-fastest rows).
# ---------------------------------------------------------------------------


def write_mmn_blocks(path, band_count, k_count, neighbor_table, blocks):
    """Stream a wannier90 ``.mmn`` file from bounded column slabs.

    :param band_count: number of bands per block (the file's ``nb``);
    :param k_count: number of k-points (the file's ``nk``);
    :param neighbor_table: (k_count, nntot, 4) integer array, one row per
        (point, neighbour slot) holding ``(k_prime, g1, g2, g3)`` with
        0-based ``k_prime`` — wannier90 matches every block header
        ``k k' G1 G2 G3`` (1-based as written) against its own kmesh
        reconstruction, so the consumer's table is authoritative and is
        emitted verbatim;
    :param blocks: iterable of ``(point, neighbor_slot, first_column,
        values)`` where ``values`` is a ``(band_count, n_columns)`` complex
        array holding a contiguous, ordered column slab of that pair's
        matrix (rows are bands m, columns are the n indices
        ``first_column .. first_column + n_columns - 1``).

    Coverage validation (fail-closed, no global-M gather): for every
    (point, slot) the slabs must arrive exactly once each, in
    point-major/slot/column order, tile the full ``0..band_count-1``
    column range contiguously without gaps, overlaps, duplicates or
    reordering.  Only one slab is live at a time.

    Layout per overlap.F90:133-198: comment; ``nb nk nntot``; per (point,
    slot) a header row then the matrix written n outer, m inner
    (m-fastest records).
    """
    neighbor_table = np.asarray(neighbor_table, dtype=int)
    if neighbor_table.ndim != 3 or neighbor_table.shape[0] != k_count \
            or neighbor_table.shape[2] != 4:
        raise ValueError(
            f"write_mmn_blocks: neighbor_table must have shape "
            f"({k_count}, nntot, 4), got {neighbor_table.shape}"
        )
    nntot = neighbor_table.shape[1]
    # coverage state: strictly point-major/slot-ordered traversal, each pair
    # tiled contiguously from column 0 to band_count
    cur_key = None
    cur_next = 0
    npairs = 0
    open_key = None
    with open(path, "w") as fh:
        fh.write(" Created by lawaf\n")
        fh.write(f"{band_count:5d} {k_count:5d} {nntot:5d}\n")

        def fail(block_key, detail):
            raise ValueError(
                f"write_mmn_blocks: bad block coverage at "
                f"(point, slot) = {block_key}: {detail}"
            )

        def successor(key):
            return (key[0], key[1] + 1) if key[1] + 1 < nntot \
                else (key[0] + 1, 0)

        for block in blocks:
            point, slot, first_column, values = block
            key = (int(point), int(slot))
            if not (0 <= key[0] < k_count and 0 <= key[1] < nntot):
                fail(key, f"outside the {k_count} x {nntot} neighbor table")
            if cur_key is None:
                if key != (0, 0):
                    fail(key, "first block must be (point, slot) = (0, 0)")
                cur_key, cur_next = key, 0
            elif key == cur_key:
                pass
            elif key == successor(cur_key):
                if cur_next != band_count:
                    fail(cur_key, f"incomplete column coverage: {cur_next} "
                                  f"of {band_count} columns written")
                npairs += 1
                cur_key, cur_next = key, 0
            else:
                fail(key, "duplicate, out-of-order or non-successive pair "
                          f"(expected {successor(cur_key)})")
            if int(first_column) != cur_next:
                fail(key, f"expected next column slab at {cur_next}, "
                          f"got first_column={int(first_column)} "
                          f"(gap or overlap)")
            values = np.asarray(values)
            if values.ndim != 2 or values.shape[0] != band_count:
                fail(key, f"values must be ({band_count}, n_columns), "
                          f"got {values.shape}")
            ncol = values.shape[1]
            if ncol <= 0:
                fail(key, "empty column slab")
            if open_key != key:
                # block header once per (point, slot) pair, not per slab
                jk, g1, g2, g3 = neighbor_table[key[0], key[1]]
                fh.write(
                    f"{key[0] + 1:5d} {int(jk) + 1:5d} "
                    f"{int(g1):5d} {int(g2):5d} {int(g3):5d}\n"
                )
                open_key = key
            for n in range(ncol):
                col = values[:, n]
                for mi in range(band_count):
                    fh.write(
                        f"{col[mi].real:18.12f}{col[mi].imag:18.12f}\n"
                    )
            cur_next = int(first_column) + ncol
        if cur_key is not None:
            if cur_next != band_count:
                fail(cur_key, f"incomplete column coverage: {cur_next} of "
                              f"{band_count} columns written")
            npairs += 1
        if npairs != k_count * nntot:
            fail(cur_key if cur_key is not None else (0, 0),
                 f"missing blocks: covered {npairs} of "
                 f"{k_count * nntot} pairs "
                 f"({k_count * nntot - npairs} missing)")


def write_amn_blocks(path, band_count, k_count, proj_count, blocks):
    """Stream a wannier90 ``.amn`` file from bounded per-k tiles.

    :param blocks: iterable of ``(point, values)`` where ``values`` is a
        ``(band_count, proj_count)`` complex projection tile
        A[m, n] = <psi_m | g_n> for that k-point; exactly one tile per
        k-point, in k order (one projection tile live at a time).

    Rows are the m-fastest records ``m n k re im`` per overlap.F90.
    """
    seen = 0
    with open(path, "w") as fh:
        fh.write(" Created by lawaf\n")
        fh.write(f"{band_count:5d} {k_count:5d} {proj_count:5d}\n")
        for point, values in blocks:
            if int(point) != seen:
                raise ValueError(
                    f"write_amn_blocks: expected tile for k-point {seen}, "
                    f"got {int(point)}"
                )
            values = np.asarray(values)
            if values.shape != (band_count, proj_count):
                raise ValueError(
                    f"write_amn_blocks: k-point {seen} tile must have shape "
                    f"({band_count}, {proj_count}), got {values.shape}"
                )
            for n in range(proj_count):
                for m in range(band_count):
                    fh.write(
                        f"{m + 1:5d} {n + 1:5d} {point + 1:5d}"
                        f"{values[m, n].real:18.12f}"
                        f"{values[m, n].imag:18.12f}\n"
                    )
            seen += 1
        if seen != k_count:
            raise ValueError(
                f"write_amn_blocks: {seen} of {k_count} k-point tiles "
                f"provided"
            )


def write_eig_blocks(path, k_count, blocks):
    """Stream a wannier90 ``.eig`` file from bounded per-k slabs.

    :param blocks: iterable of ``(point, evals)`` with ``evals`` the
        eigenvalues (in eV) of that k-point's retained bands, in strict
        k-outer / band-inner record order (parameters.F90:1646-1648).
    """
    nb = None
    seen = 0
    with open(path, "w") as fh:
        for point, evals in blocks:
            if int(point) != seen:
                raise ValueError(
                    f"write_eig_blocks: expected slab for k-point {seen}, "
                    f"got {int(point)}"
                )
            evals = np.asarray(evals, dtype=float)
            if nb is None:
                nb = evals.shape[0]
            elif evals.shape[0] != nb:
                raise ValueError(
                    f"write_eig_blocks: k-point {seen} has {evals.shape[0]} "
                    f"eigenvalues, expected {nb}"
                )
            for n in range(evals.shape[0]):
                fh.write(f"{n + 1:5d} {point + 1:5d} {evals[n]:18.12f}\n")
            seen += 1
        if seen != k_count:
            raise ValueError(
                f"write_eig_blocks: {seen} of {k_count} k-point slabs provided"
            )
    return seen, (nb if nb is not None else 0)
