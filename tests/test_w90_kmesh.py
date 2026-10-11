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


def _mk_mesh(n):
    g = np.arange(n) / n
    return np.array([[i, j, k] for i in g for j in g for k in g])


def test_skewed_triclinic_mesh_b1_and_neighbors():
    """Strongly skewed triclinic cell: B1 holds exactly and every declared
    neighbour is the unique congruent mesh point (fractional congruence
    check done independently of the kmesh code under test)."""
    recip = np.array(
        [
            [1.30, 0.21, -0.14],
            [0.05, 1.12, 0.33],
            [-0.09, 0.27, 1.05],
        ]
    )
    kpts = _mk_mesh(3)
    nn, nnc, bvecs, wb = kmesh_nnlist(kpts, recip)
    inv = np.linalg.inv(recip)
    for ik in range(len(kpts)):
        disp = np.array(
            [
                kpts[jk] + g - kpts[ik]
                for jk, g in zip(nn[ik], nnc[ik])
            ]
        )
        # every neighbour displacement is an exact 3x3x3 mesh translation
        np.testing.assert_allclose(
            disp * 3 - np.round(disp * 3), 0.0, atol=1e-9
        )
        # and the per-k displacement set reproduces the k=1 shell set
        key = lambda a: np.array(sorted(map(tuple, np.round(a, 6))))
        np.testing.assert_allclose(key(disp), key(bvecs @ inv), atol=1e-9)
    b1 = np.einsum("n,ni,nj->ij", wb, bvecs, bvecs)
    np.testing.assert_allclose(b1, np.eye(3), atol=1e-7)


def test_higher_order_shells_weights_metric_completeness():
    """A 4x4x4 mesh of a cell whose shortest mesh translations alone cannot
    satisfy B1 pulls in the next shell; the solved weights make B1 exact
    and the shell set is complete (adding farther shells is redundant)."""
    recip = np.diag([1.0, 1.1, 0.9])
    kpts = _mk_mesh(4)
    nn, nnc, bvecs, wb = kmesh_nnlist(kpts, recip)
    b1 = np.einsum("n,ni,nj->ij", wb, bvecs, bvecs)
    np.testing.assert_allclose(b1, np.eye(3), atol=1e-7)
    # metric completeness: the weighted shell projector has eigenvalues
    # (1, 1, 1) — no direction under- or over-weighted
    assert np.allclose(np.linalg.eigvalsh(b1), 1.0, atol=1e-7)
    # weights are constant within a shell (equal-length neighbours)
    norms = np.linalg.norm(bvecs, axis=1)
    for v in np.unique(np.round(norms, 6)):
        assert np.ptp(wb[np.round(norms, 6) == v]) < 1e-12


def test_deterministic_shells_across_range_growth():
    """The shell set and weights are identical whether the adaptive search
    stops at the smallest or a larger image range (deterministic solved
    shells, no range-size dependence)."""
    recip = np.array([[1.2, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.1]])
    kpts = _mk_mesh(3)
    out1 = kmesh_nnlist(kpts, recip)
    out2 = kmesh_nnlist(kpts, recip)
    np.testing.assert_array_equal(out1[0], out2[0])
    np.testing.assert_array_equal(out1[1], out2[1])
    np.testing.assert_allclose(out1[3], out2[3], atol=0)


def test_irregular_mesh_rejected():
    """A non-regular point set must fail loudly, not produce silent
    neighbour lists."""
    recip = np.eye(3)
    kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.5, 0.0]])
    with np.testing.assert_raises(ValueError):
        kmesh_nnlist(kpts, recip)
