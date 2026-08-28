"""Tests for lawaf.io.w90.kmesh_nnlist against wannier90 .nnkp oracle files.

Fixtures were generated with ``wannier90.x -pp`` (QE 7.4 bundle, w90 v3.1.0)
using ``mp_grid`` + explicit Gamma-centered kpoints + ``auto_projections``.
"""

from pathlib import Path

import numpy as np
import pytest

from lawaf.io.w90 import kmesh_nnlist

FIXTURES = Path(__file__).parent / "fixtures" / "w90"


def parse_nnkp(path):
    """Parse real/recip lattice, kpoints and nnkpts blocks of a .nnkp file."""
    lines = Path(path).read_text().splitlines()
    blocks = {}
    name = None
    for line in lines:
        s = line.strip()
        if s.startswith("begin "):
            name = s.split()[1]
            blocks[name] = []
        elif s.startswith("end "):
            name = None
        elif name is not None:
            blocks[name].append(s)
    real = np.array(
        [[float(x) for x in row.split()] for row in blocks["real_lattice"]]
    )
    recip = np.array(
        [[float(x) for x in row.split()] for row in blocks["recip_lattice"]]
    )
    kpts = np.array(
        [[float(x) for x in row.split()] for row in blocks["kpoints"][1:]]
    )
    nntot = int(blocks["nnkpts"][0])
    nnk = np.array(
        [[int(x) for x in row.split()] for row in blocks["nnkpts"][1:]]
    )
    assert nnk.shape == (len(kpts) * nntot, 5)
    nnlist = (nnk[:, 1] - 1).reshape(len(kpts), nntot)
    nncell = nnk[:, 2:5].reshape(len(kpts), nntot, 3)
    # the first column repeats the k index in order; verify
    assert np.array_equal((nnk[:, 0] - 1).reshape(len(kpts), nntot),
                          np.tile(np.arange(len(kpts))[:, None], (1, nntot)))
    return {"real_lattice": real, "recip_lattice": recip, "kpts": kpts,
            "nntot": nntot, "nnlist": nnlist, "nncell": nncell}


CASES = ["cubic444", "skewed333", "even222", "gamma"]


@pytest.mark.parametrize("case", CASES)
def test_kmesh_nnlist_matches_w90(case):
    ref = parse_nnkp(FIXTURES / f"{case}.nnkp")
    nnlist, nncell, bvecs, wb = kmesh_nnlist(
        ref["kpts"], ref["recip_lattice"]
    )
    assert nnlist.shape == (len(ref["kpts"]), ref["nntot"])
    assert nncell.shape == (len(ref["kpts"]), ref["nntot"], 3)
    np.testing.assert_array_equal(nnlist, ref["nnlist"])
    np.testing.assert_array_equal(nncell, ref["nncell"])


@pytest.mark.parametrize("case", CASES)
def test_b1_condition(case):
    """sum_b w_b b b^T == I (Eq. B1, PRB 56, 12847 (1997))."""
    ref = parse_nnkp(FIXTURES / f"{case}.nnkp")
    _, _, bvecs, wb = kmesh_nnlist(ref["kpts"], ref["recip_lattice"])
    # bvecs is (nntot, 3) for the first k-point (matches w90 bk)
    b1 = np.einsum("n,ni,nj->ij", wb, bvecs, bvecs)
    np.testing.assert_allclose(b1, np.eye(3), atol=1e-7)


def test_neighbor_count_invariants():
    for case in CASES:
        ref = parse_nnkp(FIXTURES / f"{case}.nnkp")
        nnlist, nncell, bvecs, wb = kmesh_nnlist(
            ref["kpts"], ref["recip_lattice"]
        )
        # per k-point: nntot distinct (jk, G) neighbour pairs
        for ik in range(len(ref["kpts"])):
            pairs = set(map(tuple, np.column_stack([nnlist[ik], nncell[ik]])))
            assert len(pairs) == nnlist.shape[1], (case, ik)

def test_distance_shells_ordered_scan_semantics():
    """The Fortran scan keeps the FIRST representative (hysteresis) and
    counts a STRICT band: global-min + closed band diverge on near-ties."""
    from lawaf.io.w90 import _distance_shells

    # representative = first candidate, later smaller-by->tol does not
    # replace it: Fortran gives dnn=2.0000005, multi=2
    d0 = np.array([[2.0000005], [2.0]])  # shape (M, nkpt=1)
    dnn, multi = _distance_shells(d0, 1e-6, search_shells=2)
    assert dnn[0] == 2.0000005
    assert multi[0] == 2

    # strict band: a value exactly on the upper boundary is excluded
    d0 = np.array([[2.0], [3.0]])
    dnn, multi = _distance_shells(d0, 1.0, search_shells=2)
    assert multi[0] == 1

    # clustered near-ties: all within tol of the FIRST candidate -> one shell
    d0 = np.array([[1.0], [1.0000009], [0.9999991], [1.00000005]])
    dnn, multi = _distance_shells(d0, 1e-6, search_shells=3)
    assert dnn[0] == 1.0
    assert multi[0] == 4


def test_distance_shells_chained_hysteresis():
    """Greedy replacement chain: a value that is NOT below the current
    representative minus tol does not update it, even when it is smaller
    than the representative; a later value below (current rep - tol) DOES
    reset. A suffix-min formula gets these wrong."""
    from lawaf.io.w90 import _distance_shells

    # [10, 9.5, 9.0], tol=0.6: 9.5 stays in band of 10; 9.0 < 10-0.6=9.4
    # resets -> dnn=9.0, and only the tail after the reset counts
    dnn, multi = _distance_shells(
        np.array([[10.0], [9.5], [9.0]]), 0.6, search_shells=3
    )
    assert dnn[0] == 9.0
    assert multi[0] == 1

    # reversed order: no reset after 9.0; 9.5 in band, 10 out (strict)
    dnn, multi = _distance_shells(
        np.array([[9.0], [9.5], [10.0]]), 0.6, search_shells=3
    )
    assert dnn[0] == 9.0
    assert multi[0] == 2

    # scaled to the default tolerance
    dnn, multi = _distance_shells(
        np.array([[2.0000015], [2.0000010], [2.0]]), 1e-6, search_shells=3
    )
    assert dnn[0] == 2.0
    assert multi[0] == 1


def test_supercell_images_eps8_ties():
    """internal_maxloc groups eps8-degenerate distances and extracts the
    lowest enumeration index first (-> placed LAST in ascending order),
    so near-equal distances must NOT be split by their numerical value."""
    from lawaf.io.w90 import _supercell_images

    # |(1,0,0)| = 1+5e-9 > |(0,0,1)| = 1, but |diff| < eps8 and (0,0,1)
    # is enumerated first -> extracted first -> placed last:
    # (1,0,0) must come BEFORE (0,0,1) in the ascending image list.
    recip = np.diag([1.0 + 5e-9, 1.0, 1.0])
    lmn, dist = _supercell_images(recip, nsupcell=1)
    assert tuple(lmn[0]) == (0, 0, 0)  # global minimum first
    i100 = int(np.flatnonzero((lmn == (1, 0, 0)).all(axis=1))[0])
    i001 = int(np.flatnonzero((lmn == (0, 0, 1)).all(axis=1))[0])
    assert i100 < i001

    # exact-tie check: symmetric lattice keeps reversed enumeration order
    lmn, _ = _supercell_images(np.eye(3), nsupcell=1)
    i_neg = [int(np.flatnonzero((lmn == g).all(axis=1))[0])
             for g in [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]]
    assert i_neg == sorted(i_neg, reverse=True)
