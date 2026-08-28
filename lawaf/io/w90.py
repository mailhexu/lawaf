"""Wannier90 file format writers and helpers.

kmesh_nnlist is a faithful Python port of wannier90's ``kmesh_get``
(src/kmesh.F90, w90 v3.1.0): the neighbor construction wannier90 uses to
validate every ``.mmn`` block. Writers for ``.amn``/``.mmn``/``.eig``/``.win``
are built on top of it.

Conventions (see ADRs in the W90-export architecture note):
- reciprocal lattice rows are b1, b2, b3 including 2*pi (w90 convention);
- nnlist/nncell use 0-based k indices and shape (nkpt, nntot) / (nkpt, nntot, 3).

Port fidelity notes (verified against the Fortran):
- ``internal_maxloc`` groups values within eps8 of the maximum and extracts
  the lowest index first, so the ascending image list carries REVERSED
  enumeration order inside eps8-degenerate groups;
- the shell-distance scan is sequential with hysteresis (a representative is
  replaced only by a later value smaller by more than kmesh_tol) and counts
  with a STRICT (dnn-tol, dnn+tol) band;
- per-k neighbor search iterates images in distance order and stops each
  shell at its multiplicity, exactly like kmesh_get lines 283-315;
- the post-construction symmetry and per-k B1 validations of kmesh_get
  (lines 361-409) are ported as hard errors.
"""

import numpy as np

# w90 constants (src/constants.F90, src/parameters.F90, src/kmesh.F90)
_EPS5 = 1.0e-5
_EPS6 = 1.0e-6
_EPS8 = 1.0e-8
_ETA = 99999999.0
_NUM_NNMAX = 12
_NSUPCELL = 5
_SEARCH_SHELLS = 36


def _supercell_images(recip_lattice, nsupcell=_NSUPCELL):
    """Port of kmesh_supercell_sort + internal_maxloc.

    Lattice images sorted ascending by |l @ B|. The Fortran selection sort
    extracts the maximum repeatedly (placing it at the end of the final
    array), and internal_maxloc treats every value within eps8 of the
    maximum as degenerate, extracting the LOWEST such index first — so
    eps8-degenerate groups appear in reversed enumeration order and are
    not split by tiny distance differences.
    """
    lmn = np.array(
        [
            (l, m, n)
            for l in range(-nsupcell, nsupcell + 1)
            for m in range(-nsupcell, nsupcell + 1)
            for n in range(-nsupcell, nsupcell + 1)
        ],
        dtype=int,
    )
    dist = np.linalg.norm(lmn @ recip_lattice, axis=1)
    n = len(dist)
    work = dist.copy()
    order = np.empty(n, dtype=int)
    for pos in range(n - 1, -1, -1):
        m = work.max()
        group = np.flatnonzero(np.abs(work - m) < _EPS8)
        i = int(group.min())  # internal_maxloc: lowest index of the group
        order[pos] = i
        work[i] = -1.0
    return lmn[order], dist[order]


def _distances_from_k1(kcart, images):
    """(M, nkpt) distances from k-point 1 to every (image, kpt) candidate."""
    out = np.empty((len(images), len(kcart)))
    for i, img in enumerate(images):
        out[i] = np.linalg.norm(kcart + img - kcart[0], axis=1)
    return out


def _distance_shells(dist_k1, kmesh_tol, search_shells=_SEARCH_SHELLS):
    """Port of the sequential shell-distance scan in kmesh_get (lines
    101-129): the representative is the first candidate in (image, kpt)
    scan order, replaced only when a later value is smaller by more than
    kmesh_tol (hysteresis), and the counter restarts at each replacement,
    counting a STRICT (dnn-tol, dnn+tol) band.

    Literal chain (vectorized per replacement): from the first candidate,
    jump repeatedly to the first later value smaller by more than kmesh_tol;
    the representative is the last jump target and the counter strict-counts
    the band from that position (the representative itself included). A
    suffix-min threshold is NOT equivalent: intermediate non-replacements
    leave the representative unchanged, so chained near-ties collapse
    differently (e.g. [10, 9.5, 9.0] with tol=0.6 -> 9.0, multiplicity 1).
    """
    d0 = dist_k1.ravel(order="F")  # image-major, kpt-minor scan order
    dnn = []
    multi = []
    dnn0 = 0.0
    for _ in range(search_shells):
        valid = d0[(d0 > kmesh_tol) & (d0 > dnn0 + kmesh_tol)]
        if valid.size == 0:
            dnn.append(_ETA)
            multi.append(0)
            dnn0 = _ETA
            continue
        p = 0
        d1 = valid[0]
        while True:
            hits = np.flatnonzero(valid[p + 1:] < d1 - kmesh_tol)
            if hits.size == 0:
                break
            p = p + 1 + int(hits[0])
            d1 = valid[p]
        d1 = float(d1)
        tail = valid[p:]
        counter = int(
            np.count_nonzero(
                (tail > d1 - kmesh_tol) & (tail < d1 + kmesh_tol)
            )
        )
        dnn.append(d1)
        multi.append(counter)
        dnn0 = d1
    return dnn, multi


def _get_bvectors(dist_k1, kcart, images, shell_dist, multiplicity, kmesh_tol):
    """Port of kmesh_get_bvectors from k-point 1: the first `multiplicity`
    candidates within the (relative) shell band, in (image, kpt) order."""
    mask = (dist_k1 >= shell_dist * (1.0 - kmesh_tol)) & (
        dist_k1 <= shell_dist * (1.0 + kmesh_tol)
    )
    idx = np.argwhere(mask)  # C-order: image-major, kpt-minor
    if len(idx) < multiplicity:
        raise ValueError("kmesh: not enough b-vectors found in shell")
    sel = idx[:multiplicity]
    return kcart[sel[:, 1]] + images[sel[:, 0]] - kcart[0]


def _shell_automatic(dist_k1, kcart, images, dnn, multi, kmesh_tol,
                     search_shells=_SEARCH_SHELLS):
    """Port of kmesh_shell_automatic: pick shells until B1 is satisfied.

    Returns (shell_list, bweight) where bweight solves
    sum_s bweight_s sum_{b in shell s} b_i b_j = delta_ij (least squares
    via the padded SVD in the Fortran).
    """
    target = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    comp = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    shell_list = []
    bvec_of_shell = []
    for shell in range(search_shells):
        if multi[shell] == 0:
            continue
        bvec = _get_bvectors(
            dist_k1, kcart, images, dnn[shell], multi[shell], kmesh_tol
        )
        # reject shells whose vectors are parallel to existing ones
        lpar = False
        for bnew in bvec:
            for bvs in bvec_of_shell:
                for old in bvs:
                    cos = np.dot(bnew, old) / (
                        np.linalg.norm(bnew) * np.linalg.norm(old)
                    )
                    if abs(abs(cos) - 1.0) < _EPS6:
                        lpar = True
                        break
                if lpar:
                    break
            if lpar:
                break
        if lpar:
            continue
        shell_list.append(shell)
        bvec_of_shell.append(bvec)
        nshell = len(shell_list)
        if nshell > 6:
            # more shells than the 6 independent B1 components: the Fortran
            # padded SVD would report a (near-)zero singular value
            shell_list.pop()
            bvec_of_shell.pop()
            continue
        amat = np.zeros((6, nshell))
        for ish, bvs in enumerate(bvec_of_shell):
            for b in bvs:
                for c, (i, j) in enumerate(comp):
                    amat[c, ish] += b[i] * b[j]
        u, s, vt = np.linalg.svd(amat, full_matrices=False)
        if np.any(np.abs(s) < _EPS5):
            if nshell == 1:
                raise ValueError(
                    "kmesh: singular shell system (small singular value)"
                )
            shell_list.pop()
            bvec_of_shell.pop()
            continue
        bweight = vt.T @ np.diag(1.0 / s) @ u.T @ target
        b1 = np.zeros((3, 3))
        for ish, bvs in enumerate(bvec_of_shell):
            for b in bvs:
                b1 += bweight[ish] * np.outer(b, b)
        offdiag = b1 - np.diag(np.diag(b1))
        if (
            np.allclose(np.diag(b1), 1.0, atol=kmesh_tol, rtol=0)
            and np.allclose(offdiag, 0.0, atol=kmesh_tol, rtol=0)
        ):
            return shell_list, bweight
    raise ValueError("kmesh: unable to satisfy B1 with searched shells")


def kmesh_nnlist(kpts, recip_lattice, kmesh_tol=1e-6):
    """Build wannier90's nearest-neighbour k-point list.

    Faithful port of wannier90 ``kmesh_get``: Cartesian-metric shells,
    automatic B1 shell selection, per-k folding with lattice-image labels,
    and the post-construction symmetry/B1 validations.

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
    lmn, _ = _supercell_images(recip_lattice)
    images = lmn @ recip_lattice  # (M, 3), ascending distance
    kcart = kpts @ recip_lattice  # rows = cartesian k-points
    dist_k1 = _distances_from_k1(kcart, images)
    dnn, multi = _distance_shells(dist_k1, kmesh_tol)
    shell_list, bweight = _shell_automatic(
        dist_k1, kcart, images, dnn, multi, kmesh_tol
    )
    nntot = sum(multi[s] for s in shell_list)
    if nntot > _NUM_NNMAX:
        raise ValueError("kmesh: found more than num_nnmax neighbours")
    wb = np.concatenate(
        [np.full(multi[s], bweight[i]) for i, s in enumerate(shell_list)]
    )
    bvecs = np.concatenate(
        [
            _get_bvectors(dist_k1, kcart, images, dnn[s], multi[s], kmesh_tol)
            for s in shell_list
        ]
    )
    nnlist = np.zeros((nkpt, nntot), dtype=int)
    nncell = np.zeros((nkpt, nntot, 3), dtype=int)
    bk = np.zeros((nkpt, nntot, 3))
    for ik in range(nkpt):
        nnx = 0
        for s in shell_list:
            lo = dnn[s] * (1.0 - kmesh_tol)
            hi = dnn[s] * (1.0 + kmesh_tol)
            found = 0
            for img in range(len(images)):  # distance-sorted image order
                d = np.linalg.norm(kcart + images[img] - kcart[ik], axis=1)
                hits = np.flatnonzero((d >= lo) & (d <= hi))
                for jk in hits:  # ascending kpt order within the image
                    if found == multi[s]:
                        break
                    nnlist[ik, nnx] = jk
                    nncell[ik, nnx] = lmn[img]
                    bk[ik, nnx] = kcart[jk] + images[img] - kcart[ik]
                    nnx += 1
                    found += 1
                if found == multi[s]:
                    break
            if found < multi[s]:
                raise ValueError(
                    f"kmesh: too few neighbours found for k-point {ik}"
                )
    # validation, kmesh_get lines 361-379: neighbour lengths must match k=1
    if not np.allclose(
        np.linalg.norm(bk, axis=-1),
        np.linalg.norm(bk[0], axis=-1)[None, :],
        atol=kmesh_tol,
        rtol=0,
    ):
        raise ValueError("Non-symmetric k-point neighbours!")
    # validation, kmesh_get lines 385-409: Eq. (B1) for every k-point
    for ik in range(nkpt):
        b1 = np.einsum("n,ni,nj->ij", wb, bk[ik], bk[ik])
        offdiag = b1 - np.diag(np.diag(b1))
        if not (
            np.allclose(np.diag(b1), 1.0, atol=kmesh_tol, rtol=0)
            and np.allclose(offdiag, 0.0, atol=kmesh_tol, rtol=0)
        ):
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
