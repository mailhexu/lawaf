"""Story 016: get_distorted_atoms regression + Q-space sampling / force projection."""
import numpy as np
from ase import Atoms

from lawaf.anharmonic.sampling import (
    FrameSpec,
    SamplingPlan,
    make_atoms,
    project_forces,
    sample_frames,
)
from lawaf.lwf.lwf import LWF
from lawaf.lwf.lwf_supercell import MyLWFSC
from lawaf.utils.supercell import SupercellMaker


def make_toy_mylwfsc(seed=7):
    """Tiny synthetic LWF: 2-atom primitive cell, 2 lwf branches, 4 R vectors,
    sc_matrix diag(2,1,1). Deterministic under seed."""
    rng = np.random.default_rng(seed)
    natom = 2
    nlwf = 2
    natom3 = 3 * natom
    Rlist = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    wannR = rng.normal(0.0, 0.5, size=(len(Rlist), natom3, nlwf))
    # keep entries clear of the 1e-4 mapping-matrix cutoff
    wannR = np.where(np.abs(wannR) < 5e-3, 5e-3, wannR)
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.2, 1.2, 1.2]],
                  cell=np.eye(3) * 5.0, pbc=True)
    lwf = LWF(wannR=wannR, Rlist=Rlist, atoms=atoms)
    return MyLWFSC(lwf, SupercellMaker(np.diag([2, 1, 1])))


def test_get_distorted_atoms_returns_shifted_positions():
    mylwfsc = make_toy_mylwfsc()
    rng = np.random.default_rng(11)
    nlwf_sc = mylwfsc.mapping_mat.shape[1]
    amp = rng.normal(0.0, 0.1, size=nlwf_sc)
    atoms, disp = mylwfsc.get_distorted_atoms(amp)
    pristine = mylwfsc.sc_atoms
    assert disp.shape == (mylwfsc.natom_sc, 3)
    # the old bug returned the pristine sc_atoms object
    assert atoms is not pristine
    assert not np.allclose(atoms.get_positions(), pristine.get_positions())
    np.testing.assert_allclose(
        atoms.get_positions(), pristine.get_positions() + disp, atol=1e-12
    )


def test_project_forces_matches_fd_gradient_with_sign():
    mylwfsc = make_toy_mylwfsc()
    M = mylwfsc.mapping_mat
    natom3_sc = M.shape[0]
    # harmonic model on the supercell atoms: E(u) = 1/2 u' Phi u, SPD Phi
    rng = np.random.default_rng(3)
    A = rng.normal(size=(natom3_sc, natom3_sc))
    Phi = A @ A.T + natom3_sc * np.eye(natom3_sc)

    plan = SamplingPlan(
        single_modes=[(0, np.array([0.1]))],
        n_random=1,
        random_amp=0.05,
        seed=42,
    )
    frames = sample_frames(mylwfsc.lwf, mylwfsc.scmaker, plan)
    assert len(frames) == 2
    h = 1e-5
    for frame in frames:
        u = M @ frame.Q
        forces = -(Phi @ u)  # ASE convention: F = -dE/du
        g_Q = project_forces(M, forces)
        fd = np.empty_like(frame.Q)
        # central differences of E(Q) = 1/2 (MQ)' Phi (MQ)
        for i in range(frame.Q.size):
            dQ = np.zeros_like(frame.Q)
            dQ[i] = h
            eplus = 0.5 * (M @ (frame.Q + dQ)).dot(Phi @ (M @ (frame.Q + dQ)))
            eminus = 0.5 * (M @ (frame.Q - dQ)).dot(Phi @ (M @ (frame.Q - dQ)))
            fd[i] = (eplus - eminus) / (2 * h)
        np.testing.assert_allclose(fd, -g_Q, rtol=1e-6, atol=1e-8)
        # explicit sign assertion: projection carries the ASE minus sign
        assert np.linalg.norm(g_Q) > 1e-8
        assert not np.allclose(fd, g_Q, rtol=1e-3)


def test_sample_frames_deterministic_given_seed():
    mylwfsc = make_toy_mylwfsc()
    plan = SamplingPlan(
        single_modes=[(1, np.linspace(-0.2, 0.2, 3))],
        coupled_modes=[((0, 1), (np.array([-0.1, 0.1]), np.array([0.05, -0.05])))],
        n_random=4,
        random_amp=0.1,
        seed=123,
    )
    f1 = sample_frames(mylwfsc.lwf, mylwfsc.scmaker, plan)
    f2 = sample_frames(mylwfsc.lwf, mylwfsc.scmaker, plan)
    assert len(f1) == len(f2) == 3 + 4 + 4
    for a, b in zip(f1, f2):
        np.testing.assert_array_equal(a.Q, b.Q)
        assert (a.strain_voigt is None) == (b.strain_voigt is None)
        if a.strain_voigt is not None:
            np.testing.assert_array_equal(a.strain_voigt, b.strain_voigt)
        assert a.provenance == b.provenance
        assert a.split == b.split


def test_sample_frames_layout_and_provenance():
    mylwfsc = make_toy_mylwfsc()
    nlwf = mylwfsc.nlwf
    ncell = mylwfsc.scmaker.ncell
    assert (nlwf, ncell) == (2, 2)
    strain = np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0])
    plan = SamplingPlan(
        single_modes=[(0, np.array([0.1, 0.2]))],
        coupled_modes=[((0, 1), (np.array([0.1]), np.array([0.2, 0.3])))],
        n_random=2,
        random_amp=0.1,
        seed=0,
        strains=[None, strain],
    )
    frames = sample_frames(mylwfsc.lwf, mylwfsc.scmaker, plan)
    # displacement frames: 2 single + 2 coupled + 2 random = 6, crossed with
    # 2 strain entries -> 12
    assert len(frames) == 12
    provs = [f.provenance for f in frames]
    assert provs.count("single") == 4
    assert provs.count("coupled") == 4
    assert provs.count("random") == 4
    assert all(f.split == "train" for f in frames)
    # single mode: branch 0 excited uniformly in every cell (c = icell*nlwf + b)
    singles = [f for f in frames if f.provenance == "single"]
    for amp in (0.1, 0.2):
        matches = [f for f in singles if np.isclose(f.Q[0], amp)]
        assert len(matches) == 2
        for f in matches:
            np.testing.assert_allclose(f.Q, [amp, 0.0, amp, 0.0], atol=1e-14)
    # every displacement frame appears once per strain entry
    q0 = frames[0].Q
    same_disp = [f for f in frames if np.array_equal(f.Q, q0)]
    assert len(same_disp) == 2
    kinds = [f.strain_voigt is None for f in same_disp]
    assert sorted(kinds) == [False, True]
    strained = [f for f in same_disp if f.strain_voigt is not None][0]
    np.testing.assert_allclose(strained.strain_voigt, strain)


def test_make_atoms_applies_strain_to_distorted_structure():
    mylwfsc = make_toy_mylwfsc()
    nlwf_sc = mylwfsc.mapping_mat.shape[1]
    Q = np.zeros(nlwf_sc)
    Q[0] = 0.08
    eps_v = np.array([0.02, -0.01, 0.015, 0.004, -0.003, 0.002])
    eps = np.zeros((3, 3))
    eps[0, 0], eps[1, 1], eps[2, 2] = eps_v[0], eps_v[1], eps_v[2]
    eps[1, 2] = eps[2, 1] = eps_v[3]
    eps[0, 2] = eps[2, 0] = eps_v[4]
    eps[0, 1] = eps[1, 0] = eps_v[5]

    frame = FrameSpec(Q=Q, strain_voigt=eps_v, provenance="test", split="val")
    atoms = make_atoms(mylwfsc, frame)
    ref_atoms, disp = mylwfsc.get_distorted_atoms(Q)
    cell0 = ref_atoms.cell.array
    p0 = ref_atoms.get_positions()
    eye = np.eye(3)
    # deformation gradient F = I + eps right-multiplied on the cell rows
    np.testing.assert_allclose(atoms.cell.array, cell0 @ (eye + eps), atol=1e-12)
    # positions scale with the cell: p' = p @ (I + eps)
    np.testing.assert_allclose(atoms.get_positions(), p0 @ (eye + eps), atol=1e-12)
    # FrameSpec round-trip
    assert np.allclose(frame.strain_voigt, eps_v)
    # without strain: distorted but undeformed cell / positions
    atoms_ns = make_atoms(
        mylwfsc, FrameSpec(Q=Q, strain_voigt=None, provenance="t", split="t")
    )
    np.testing.assert_allclose(atoms_ns.cell.array, cell0, atol=1e-12)
    np.testing.assert_allclose(atoms_ns.get_positions(), p0, atol=1e-12)


def test_sign_chain_symbolic():
    """sympy: for E(u) = 1/2 u' Phi u, u = M Q, ASE F = -dE/du, the projection
    g_Q = M' F equals -dE/dQ (the sign chain documented in sampling.py)."""
    import sympy as sp

    M = sp.Matrix(2, 3, lambda i, j: sp.Symbol(f"m{i}{j}"))
    P = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"p{i}{j}"))
    Phi = P + P.T  # symmetric harmonic force constants
    q = sp.Matrix(3, 1, lambda i, _: sp.Symbol(f"q{i}"))
    u = M * q
    E = sp.Rational(1, 2) * (u.T * Phi * u)[0, 0]
    dEdQ = sp.Matrix([sp.diff(E, qi) for qi in q])
    F_atom = -Phi * u  # ASE convention
    g_Q = M.T * F_atom
    assert sp.expand(dEdQ + g_Q) == sp.zeros(3, 1)


def test_strain_convention_symbolic():
    """sympy: the documented cell @ (I + eps) deformation realizes the
    deformation gradient F = I + eps, hence Green-Lagrangian strain
    E_GL = (F'F - I)/2 = eps + eps^2/2 (C = I + 2 E_GL = (I + eps)^2)."""
    import sympy as sp

    e00, e11, e22, e12, e02, e01 = sp.symbols("e00 e11 e22 e12 e02 e01")
    eps = sp.Matrix([[e00, e01, e02], [e01, e11, e12], [e02, e12, e22]])
    F = sp.eye(3) + eps
    E_GL = (F.T * F - sp.eye(3)) / 2
    assert sp.expand(E_GL - (eps + eps**2 / 2)) == sp.zeros(3, 3)
    # explicit diagonal case: E_GL_ii = e_ii + e_ii^2 / 2
    d0, d1, d2 = sp.symbols("d0 d1 d2")
    Fd = sp.diag(1 + d0, 1 + d1, 1 + d2)
    E_d = (Fd.T * Fd - sp.eye(3)) / 2
    ref = sp.diag(d0 + d0**2 / 2, d1 + d1**2 / 2, d2 + d2**2 / 2)
    assert sp.expand(E_d - ref) == sp.zeros(3, 3)
