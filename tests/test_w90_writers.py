"""Tests for lawaf.io.w90 writers: .amn, .eig, .win.

Formats pinned from wannier90 v3.1.0 source:
- .amn  (overlap.F90:219-252): comment; ``nb nk nwann``; rows ``m n ik re im``
        (1-based; reader is order-agnostic; we write the pw2wannier90
        convention ik outer, n middle, m inner);
- .eig  (parameters.F90:1644-1663): no header; STRICTLY ordered rows
        ``band kpt e`` with k outer, band inner (index match enforced);
- .win  (parameters.F90): mp_grid + explicit kpoints block, unit_cell_cart
        (Ang), atoms_frac, projections (or ``random``).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from lawaf.io.w90 import write_amn, write_eig, write_win

W90 = Path.home() / ".local/qe74_release/bin/wannier90.x"


def parse_amn(path):
    """Independent .amn parser returning (nb, nk, nw, array[nk, nb, nw])."""
    lines = Path(path).read_text().splitlines()
    nb, nk, nw = (int(x) for x in lines[1].split())
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[2:]])
    assert rows.shape == (nb * nk * nw, 5)
    m, n, k = (rows[:, i].astype(int) for i in range(3))
    vals = rows[:, 3] + 1j * rows[:, 4]
    out = np.zeros((nk, nb, nw), dtype=complex)
    out[k - 1, m - 1, n - 1] = vals
    return nb, nk, nw, out, lines


def test_amn_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    amn = rng.standard_normal((3, 5, 2)) + 1j * rng.standard_normal((3, 5, 2))
    f = tmp_path / "seed.amn"
    write_amn(amn, f)
    nb, nk, nw, back, lines = parse_amn(f)
    assert (nb, nk, nw) == (5, 3, 2)
    np.testing.assert_allclose(back, amn, atol=1e-12)
    # pw2wannier90 row order: ik outer, n middle, m inner (fastest)
    assert lines[2].split()[:3] == ["1", "1", "1"]   # m=1 n=1 k=1
    assert lines[3].split()[:3] == ["2", "1", "1"]   # m=2 n=1 k=1
    assert lines[7].split()[:3] == ["1", "2", "1"]   # m=1 n=2 k=1
    assert lines[12].split()[:3] == ["1", "1", "2"]  # m=1 n=1 k=2


def test_amn_roundtrip_wannier90io(tmp_path):
    w90io = pytest.importorskip("wannier90io")
    rng = np.random.default_rng(2)
    amn = rng.standard_normal((2, 4, 3)) + 1j * rng.standard_normal((2, 4, 3))
    f = tmp_path / "seed.amn"
    write_amn(amn, f)
    with open(f) as fh:
        back = w90io.read_amn(fh)
    np.testing.assert_allclose(back, amn, atol=1e-12)


def test_amn_projection_norm(tmp_path):
    """Projections onto orthonormal states: each column has norm <= 1."""
    rng = np.random.default_rng(3)
    nb, nw, nk = 6, 2, 3
    q, _ = np.linalg.qr(rng.standard_normal((nb, nw)) + 0j)
    amn = np.tile(q, (nk, 1, 1))
    f = tmp_path / "seed.amn"
    write_amn(amn, f)
    _, _, _, back, _ = parse_amn(f)
    norms2 = (np.abs(back) ** 2).sum(axis=1)  # (nk, nw)
    np.testing.assert_allclose(norms2, 1.0, atol=1e-10)


def test_amn_orthogonalize(tmp_path):
    """orthogonalize=True: the Loewdin transform A (A^dag A)^-1/2 — the
    unitary polar factor, independently computed via SVD as U @ Vh."""
    rng = np.random.default_rng(4)
    nb, nw, nk = 6, 3, 2
    amn = rng.standard_normal((nk, nb, nw)) + 1j * rng.standard_normal((nk, nb, nw))
    f = tmp_path / "seed.amn"
    write_amn(amn, f, orthogonalize=True)
    _, _, _, back, _ = parse_amn(f)
    for ik in range(nk):
        gram = back[ik].conj().T @ back[ik]
        np.testing.assert_allclose(gram, np.eye(nw), atol=1e-10)
        u, _, vh = np.linalg.svd(amn[ik], full_matrices=False)
        np.testing.assert_allclose(back[ik], u @ vh, atol=1e-10)


def test_eig_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    eig = rng.standard_normal((3, 4))
    f = tmp_path / "seed.eig"
    write_eig(eig, f)
    rows = np.array([[float(x) for x in ln.split()]
                     for ln in Path(f).read_text().splitlines()])
    assert rows.shape == (12, 3)
    # strict order: k outer, band inner (parameters.F90:1646-1648)
    np.testing.assert_array_equal(rows[:, 0].astype(int),
                                  np.tile(np.arange(1, 5), 3))
    np.testing.assert_array_equal(rows[:, 1].astype(int),
                                  np.repeat(np.arange(1, 4), 4))
    np.testing.assert_allclose(rows[:, 2].reshape(3, 4), eig, atol=1e-12)


def _win_inputs():
    from lawaf.utils.kpoints import monkhorst_pack
    a = 3.9
    lattice = a * np.eye(3)
    symbols = ["Sr", "Mn"]
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    kpts = monkhorst_pack([2, 2, 2])
    return dict(num_wann=3, num_bands=4, mp_grid=[2, 2, 2], lattice=lattice,
                symbols=symbols, frac=frac, kpts=kpts)


def test_win_kpoints_block(tmp_path):
    inp = _win_inputs()
    f = tmp_path / "seed.win"
    write_win(f, **inp)
    text = f.read_text()
    assert "mp_grid" in text and "num_wann" in text and "num_bands" in text
    body = text.split("begin kpoints")[1].split("end kpoints")[0]
    lines = [ln for ln in body.splitlines()[1:] if ln.strip()]
    assert len(lines) == 8
    kpts = np.array([[float(x) for x in ln.split()[:3]] for ln in lines])
    np.testing.assert_allclose(kpts, inp["kpts"], atol=1e-12)
    # fractional atoms block
    body = text.split("begin atoms_frac")[1].split("end atoms_frac")[0]
    lines = [ln for ln in body.splitlines() if ln.strip()]
    assert lines[0].split()[0] == "Sr" and lines[1].split()[0] == "Mn"


def test_win_random_projections_accepted_by_w90(tmp_path):
    if not W90.exists():
        pytest.skip("wannier90.x not available")
    inp = _win_inputs()
    inp["projections"] = None  # -> random
    f = tmp_path / "seed.win"
    write_win(f, **inp)
    r = subprocess.run([str(W90), "-pp", "seed"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (tmp_path / "seed.nnkp").exists()


def test_win_explicit_projections_accepted_by_w90(tmp_path):
    if not W90.exists():
        pytest.skip("wannier90.x not available")
    inp = _win_inputs()
    inp["projections"] = ["Mn:d"]
    f = tmp_path / "seed.win"
    write_win(f, **inp)
    r = subprocess.run([str(W90), "-pp", "seed"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (tmp_path / "seed.nnkp").exists()
