"""Story 7: SupercellMaker.sc_Rlist_HR accumulate mode conserves spectra."""
import numpy as np

from lawaf.mathutils.kR_convert import R_to_onek
from lawaf.utils.supercell import SupercellMaker


def make_hr(kmesh=(2, 2, 2), nwann=2, seed=3):
    rng = np.random.default_rng(seed)
    kpts = np.array([[i / kmesh[0], j / kmesh[1], k / kmesh[2]]
                     for i in range(kmesh[0])
                     for j in range(kmesh[1])
                     for k in range(kmesh[2])])
    Rlist = np.array([[i, j, k] for i in range(kmesh[0])
                      for j in range(kmesh[1]) for k in range(kmesh[2])],
                     dtype=int)
    HR = np.zeros((len(Rlist), nwann, nwann), dtype=complex)
    for R in Rlist:
        H = rng.random((nwann, nwann)) + 1j * rng.random((nwann, nwann))
        HR[list(map(tuple, Rlist)).index(tuple(R))] = 0.5 * (H + H.conj().T)
    return Rlist, HR, kpts


def test_accumulate_conserves_spectrum():
    sc_matrix = np.diag([2, 2, 2])
    kmesh = (2, 2, 2)
    Rlist, HR, kpts = make_hr(kmesh)
    scm = SupercellMaker(sc_matrix)
    sc_Rlist, sc_HR = scm.sc_Rlist_HR(Rlist, HR, n_basis=HR.shape[1],
                                      accumulate=True)
    # supercell R list must be unique
    assert len(sc_Rlist) == len(np.unique(sc_Rlist, axis=0))
    # folded k-points of the Gamma supercell point
    N = sc_matrix.diagonal()
    folded_k = np.array([[i / N[0], j / N[1], k / N[2]]
                         for i in range(N[0]) for j in range(N[1])
                         for k in range(N[2])])
    ref_evals = np.sort(np.array(
        [np.linalg.eigvalsh(R_to_onek(k, Rlist, HR)) for k in folded_k]
    ).flatten())
    sc_evals = np.sort(np.linalg.eigvalsh(
        R_to_onek(np.zeros(3), sc_Rlist, sc_HR))).flatten()
    assert np.allclose(ref_evals, sc_evals, atol=1e-10)


def test_accumulate_sums_colliding_blocks():
    """R vectors equal mod the supercell fold to ONE supercell R, keeping
    all contributions (legacy mode emits duplicate rows)."""
    Rlist = np.array([[1, 0, 0], [3, 0, 0]])  # equal mod sc 2
    HR = np.zeros((2, 1, 1), dtype=float)
    HR[0, 0, 0] = 1.0
    HR[1, 0, 0] = 2.0
    scm = SupercellMaker(np.diag([2, 2, 2]))
    sc_Rlist, sc_HR = scm.sc_Rlist_HR(Rlist, HR, n_basis=1, accumulate=True)
    assert len(sc_Rlist) == len(np.unique(sc_Rlist, axis=0))
    # legacy mode emits duplicate supercell R rows; totals must match
    lg_Rlist, lg_HR = scm.sc_Rlist_HR(Rlist, HR, n_basis=1)
    assert len(lg_Rlist) > len(np.unique(lg_Rlist, axis=0))
    assert np.isclose(sc_HR.sum(), lg_HR.sum())
