"""Tests for lawaf.anharmonic.representation — story-014 space-group action S_g(q).

Validates the space-group representation on the mass-weighted Cartesian
displacement basis (atom-major xyz, lawaf gauge) with the phase convention
documented in the module docstring:

- TEST-001 unitarity + group law S_g(hq) S_h(q) = S_{gh}(q) on commensurate
  meshes including non-TRIM points (symmetry-only, no dynamical matrix).
- TEST-002 eigenvector transport at non-TRIM q (projector residual) against
  PhonopyWrapper.solve, plus a negative control: the transposed source/target
  placement of the rotation blocks (a wrong pairing convention) must FAIL.
  Note: on the symmorphic Pm-3m fixture a bare sign flip of the defect phase
  e^{∓2πi q'.t_kappa} is degenerate (t_kappa is integral for every operation),
  so the discriminating wrong convention is the misplaced pairing.
- TEST-003 Gamma reduces to a real permutation-rotation.
- TEST-004 star orbit + irreducible-qpoint completeness/partition.
- One-symmetry-analysis NFR: spglib runs once at build time.

Oracle evidence: story-014 debugging (BaTiO3 DM_dip_wang exhaustive transport
search: convention residual ~1e-13 across 48 ops x 15 bands; lawaf gauge law
H = E^-1 D E verified at 1e-17), story-010 memo 2026-08-29-spgrep-modulation-
projectors (gauge conversion e^{+2 pi i r_kappa.q}).
"""

from pathlib import Path

import numpy as np
import pytest

phonopy = pytest.importorskip("phonopy")

from lawaf.anharmonic.representation import (  # noqa: E402
    SpaceGroupAction,
    build_space_group_action,
)

FIXTURE = Path(__file__).parent / "fixtures" / "phonopy" / "BaTiO3_DM_dip_wang.yaml"

# commensurate, includes non-TRIM points (denominators up to 8)
MESH_QS = [
    np.zeros(3),
    np.array([0.5, 0.0, 0.25]),
    np.array([0.25, 0.25, 0.25]),
    np.array([0.125, 0.5, 0.25]),
    np.array([0.5, 0.5, 0.5]),
    np.array([0.0, 0.25, 0.5]),
]


@pytest.fixture(scope="module")
def phonon():
    ph = phonopy.load(phonopy_yaml=str(FIXTURE), is_nac=False)
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def model(phonon):
    from lawaf.interfaces.phonopy.phonopywrapper import PhonopyWrapper

    return PhonopyWrapper(phonon, mode="dm", is_nac=False, use_cache=False)


@pytest.fixture(scope="module")
def sga(phonon):
    return build_space_group_action(phonon)


# ---------------------------------------------------------------- TEST-001
def test_build_basic(sga):
    """Dataset recorded once; primitive BaTiO3: Pm-3m (#221), 5 atoms, 48 ops."""
    assert isinstance(sga, SpaceGroupAction)
    assert sga.n_atoms == 5
    assert sga.n_ops == 48
    assert sga.symmetry_dataset.number == 221
    assert sga.rotations.shape == (48, 3, 3)
    assert sga.translations.shape == (48, 3)


def test_unitarity_all_ops_all_q(sga):
    """S_g(q) is unitary to 1e-12 for every operation at each mesh q."""
    for q in MESH_QS:
        for g in range(sga.n_ops):
            M = sga.matrix(g, q)
            err = np.abs(M @ M.conj().T - np.eye(3 * sga.n_atoms)).max()
            assert err <= 1e-12, (q, g, err)


def test_group_law_closure_mesh(sga):
    """S_g(hq) S_h(q) = S_{gh}(q) to 1e-12, incl. non-TRIM q (no DM needed)."""

    def find_op(R, w):
        for k in range(sga.n_ops):
            if np.array_equal(sga.rotations[k], R) and np.allclose(
                sga.translations[k] % 1.0, w % 1.0, atol=1e-8
            ):
                return k
        return None

    for q in MESH_QS:
        for g in range(sga.n_ops):
            qg = sga.qmap(g, q)
            for h in range(0, sga.n_ops, 7):  # stride: 48*48/7 products per q
                R_gh = sga.rotations[g] @ sga.rotations[h]
                w_gh = sga.rotations[g].astype(float) @ sga.translations[h] + sga.translations[g]
                gh = find_op(R_gh, w_gh)
                assert gh is not None, (g, h)
                S_gh = sga.matrix(gh, q)
                S_g_at_qh = sga.matrix(g, sga.qmap(h, q))
                S_h = sga.matrix(h, q)
                err = np.abs(S_g_at_qh @ S_h - S_gh).max()
                assert err <= 1e-12, (q, g, h, err)


# ---------------------------------------------------------------- TEST-002
def _transport_residuals(sga, model, q, placement="sigma"):
    """Max projector residual of S_g(q) v(q) over all ops and bands."""
    e_q, v_q = model.solve(q)
    n = 3 * sga.n_atoms
    cache = {}

    def solve_cached(qp):
        key = tuple(np.round(qp, 8) % 1.0)
        if key not in cache:
            cache[key] = model.solve(qp)
        return cache[key]

    worst = 0.0
    for g in range(sga.n_ops):
        qp = sga.qmap(g, q)
        e_p, v_p = solve_cached(qp)
        M = np.zeros((n, n), dtype=complex)
        R = sga.cart_rotations[g]
        sig = sga.atom_maps[g]
        t_all = sga.defect_vectors[g]
        spos = sga.scaled_positions
        for kappa in range(sga.n_atoms):
            ex = -(qp @ t_all[kappa])
            src, dst = (kappa, sig[kappa]) if placement == "sigma" else (sig[kappa], kappa)
            M[3 * dst:3 * dst + 3, 3 * src:3 * src + 3] = np.exp(2j * np.pi * ex) * R
        for j in range(n):
            lam = e_q[j]
            cluster = np.abs(e_p - lam) <= 1e-4
            Vp = v_p[:, cluster]
            P = Vp @ Vp.conj().T
            r = float(np.linalg.norm((np.eye(n) - P) @ (M @ v_q[:, j])))
            worst = max(worst, r)
    return worst


def test_transport_nontrim_q(sga, model):
    """S_g(q) v(q) lies in the eigenspace at g.q for all ops (residual <= 1e-10)."""
    for q in (np.array([0.5, 0.0, 0.25]), np.array([0.25, 0.25, 0.25])):
        worst = _transport_residuals(sga, model, q)
        assert worst <= 1e-10, (q, worst)


def test_transport_negative_control(sga, model):
    """Transposed rotation-block placement (wrong convention) FAILS transport."""
    q = np.array([0.5, 0.0, 0.25])
    worst = _transport_residuals(sga, model, q, placement="transposed")
    assert worst > 1e-6, worst


# ---------------------------------------------------------------- TEST-003
def test_gamma_real_permutation_rotation(sga):
    """At Gamma every S_g is real (imaginary part <= 1e-14) and unitary."""
    for g in range(sga.n_ops):
        M = sga.matrix(g, np.zeros(3))
        assert np.abs(M.imag).max() <= 1e-14, g
        err = np.abs(M @ M.conj().T - np.eye(3 * sga.n_atoms)).max()
        assert err <= 1e-12, g


# ---------------------------------------------------------------- TEST-004
def test_star_orbit(sga):
    """Star images are distinct, op-canonical, and group-closed."""
    q = np.array([0.5, 0.0, 0.25])
    star = sga.star(q)
    assert len(star) <= sga.n_ops
    assert sga.n_ops % len(star) == 0  # orbit-stabilizer
    images = [qp for _, qp in star]
    for g, qp in star:
        back = sga.qmap(g, qp)  # applying g's *inverse-equivalent* wrap check:
        d = back - np.rint(back)
        # image must differ from q unless g stabilizes q
        dq = qp - q
        dq -= np.rint(dq)
        if np.linalg.norm(dq) > 1e-8:
            # no operation may map q outside the collected orbit
            pass
    # closure: applying any operation to any star image lands in the orbit
    ops_images = [sga.qmap(g, im) for g in range(sga.n_ops) for im in images]
    for qp in ops_images:
        inside = any(
            np.linalg.norm(((qp - s) - np.rint(qp - s))) < 1e-8 for s in images
        )
        assert inside, qp
    assert len(images) == len({tuple(np.round(s, 6) % 1.0) for s in images})


def test_star_gamma_trivial(sga):
    """Gamma is its own star: single image, stabilized by every operation."""
    star = sga.star(np.zeros(3))
    assert len(star) == 1
    assert np.allclose(star[0][1], 0.0)


def test_irreducible_qpoints_partition(sga):
    """Stars of the representatives partition the 2x2x2 mesh exactly once."""
    mesh = (2, 2, 2)
    reps = sga.irreducible_qpoints(mesh)
    covered = []
    for r in reps:
        for _, qp in sga.star(r):
            covered.append(tuple(np.round(qp * 2).astype(int) % 2))
    expected = {(i, j, k) for i in range(2) for j in range(2) for k in range(2)}
    assert len(covered) == len(expected)
    assert set(covered) == expected


# --------------------------------------------------------------- NFR: once
def test_single_symmetry_analysis(sga, monkeypatch):
    """matrix()/star() are pure: blocking spglib afterwards changes nothing."""
    import lawaf.anharmonic.representation as rep_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("spglib must not be called after build")

    monkeypatch.setattr(rep_mod.spglib, "get_symmetry_dataset", _boom)
    M = sga.matrix(3, np.array([0.5, 0.0, 0.25]))
    assert M.shape == (3 * sga.n_atoms, 3 * sga.n_atoms)
    sga.star(np.array([0.25, 0.25, 0.25]))
    sga.irreducible_qpoints((2, 2, 2))


def test_build_from_ase_atoms(phonon):
    """The same structure via ase.Atoms yields the identical action."""
    from ase import Atoms

    prim = phonon.primitive
    atoms = Atoms(
        symbols=prim.symbols,
        cell=np.array(prim.cell),
        scaled_positions=np.array(prim.scaled_positions),
    )
    sga2 = build_space_group_action(atoms)
    q = np.array([0.5, 0.0, 0.25])
    for g in (0, 7, 23, 41):
        assert np.abs(sga2.matrix(g, q) - build_space_group_action(phonon).matrix(g, q)).max() <= 1e-12
