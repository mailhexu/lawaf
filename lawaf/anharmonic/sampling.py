"""Q-space sampling and force projection for the LWF anharmonic effective model.

Conventions
-----------
Q ordering
    A frame amplitude vector ``Q`` is the flat vector over the columns of
    ``build_lwf_lattice_mapping_matrix(mylwf, scmaker)`` (a sparse CSR matrix
    ``M`` of shape ``(3 * natom_sc, ncell * nlwf)``).  Column ``c`` addresses
    supercell translation ``icell = c // nlwf`` (in ``scmaker.sc_vec`` order)
    and branch ``iwann = c % nlwf`` of the primitive cell, i.e.
    ``c = icell * nlwf + iwann``.  The real-space displacement field is
    ``u = lwf_to_disp(M, Q) = M @ Q``, flattened atom-major xyz (atom index
    slow, x/y/z fast) — the same ordering as the rows of ``M``.

Sign chain
    ASE forces follow ``F = -dE/du``.  The mode-space projected force is
    ``g_Q = project_forces(M, F) = M.T @ F``.  For ``u = M Q`` the chain rule
    gives ``dE/dQ = M.T @ dE/du = -M.T @ F = -g_Q``; asserted symbolically
    (sympy) in ``tests/test_anharmonic_sampling.py`` and verified numerically
    there by finite differences.

Strain convention
    ``strain_voigt`` uses the order ``(xx, yy, zz, yz, xz, xy)`` and defines
    the symmetric small-strain matrix ``eps``.  The supercell deformation is
    ``F = I + eps`` applied uniformly (lattice vectors and atomic positions
    transform as ``v -> F v``), implemented as
    ``atoms.set_cell(cell @ F, scale_atoms=True)``.  The exact finite strain
    measure realized is therefore the Green-Lagrangian strain
    ``E_GL = (F.T F - I) / 2 = eps + eps**2 / 2`` (right Cauchy-Green
    ``C = I + 2 E_GL = (I + eps)**2``), i.e. ``eps`` is the small-strain
    parameter and equals ``E_GL`` to first order.  This identity is asserted
    symbolically (sympy, diagonal and general symmetric cases) in
    ``tests/test_anharmonic_sampling.py``.  Strain is applied on top of the
    displacement ``u`` of the same frame.

Sampling semantics
    ``single_modes`` excites one branch with the SAME amplitude in every
    supercell cell (a Gamma-like uniform modulation).  ``coupled_modes`` takes
    the Cartesian product of one amplitude array per branch.  ``n_random``
    frames draw each Q component uniformly from
    ``[-random_amp, +random_amp]`` (broadcast over a per-component array
    ``random_amp`` too) from ``np.random.default_rng(seed)``.  Each
    displacement frame is repeated once per entry of ``strains`` (None entries
    included), displacement-major / strain-minor.
"""
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FrameSpec:
    """One sampled structure: mode amplitudes Q plus an optional strain.

    Q: flat amplitudes over M's columns, ``c = icell * nlwf + iwann``.
    strain_voigt: (6,) in order (xx, yy, zz, yz, xz, xy), or None.
    provenance / split: free-form lineage tags (e.g. "single"/"coupled"/
    "random" and "train"/"val").
    """

    Q: np.ndarray
    strain_voigt: Optional[np.ndarray] = None
    provenance: str = ""
    split: str = "train"


@dataclass
class SamplingPlan:
    """Declarative description of a Q-space sampling set.

    ``single_modes`` and ``coupled_modes`` excite primitive-cell branch
    indices uniformly over the supercell and retain their story-017 API.
    ``vector_modes`` and ``coupled_vectors`` are their finite-supercell
    counterparts: each vector is a normalized Q-space character direction
    (for example a folded X or M harmonic eigenvector), so zone-boundary
    ladders remain in the exact LWF subspace.
    """
    single_modes: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    coupled_modes: List[Tuple[Tuple[int, ...], Tuple[np.ndarray, ...]]] = field(
        default_factory=list
    )
    vector_modes: List[Tuple[str, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    coupled_vectors: List[
        Tuple[Tuple[str, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]
    ] = field(default_factory=list)
    n_random: int = 0
    random_amp: float = 0.1
    seed: int = 0
    strains: List[Optional[np.ndarray]] = field(default_factory=lambda: [None])
    split: str = "train"


def sample_frames(lwf, scmaker, plan: SamplingPlan) -> List[FrameSpec]:
    """Materialize a SamplingPlan into a list of FrameSpec."""
    nlwf = lwf.wannR.shape[2]
    ncell = scmaker.ncell
    nQ = ncell * nlwf

    def branch_line(branches, amps) -> np.ndarray:
        Q = np.zeros(nQ)
        for b, a in zip(branches, amps):
            if not 0 <= b < nlwf:
                raise ValueError(f"branch {b} out of range [0, {nlwf})")
            Q[b::nlwf] = a
        return Q

    disp_frames = []  # (Q, provenance)
    for branch, amps in plan.single_modes:
        for a in np.atleast_1d(amps):
            disp_frames.append((branch_line([branch], [a]), "single"))
    for branches, amp_arrays in plan.coupled_modes:
        if len(branches) != len(amp_arrays):
            raise ValueError("coupled_modes needs one amplitude array per branch")
        for combo in product(*[np.atleast_1d(a) for a in amp_arrays]):
            disp_frames.append((branch_line(branches, combo), "coupled"))
    for name, vector, amps in plan.vector_modes:
        v = np.asarray(vector, dtype=float)
        if v.shape != (nQ,):
            raise ValueError(f"vector mode {name!r} must have shape ({nQ},)")
        if not np.isfinite(v).all() or np.linalg.norm(v) == 0.0:
            raise ValueError(f"vector mode {name!r} must be finite and nonzero")
        v = v / np.linalg.norm(v)
        for a in np.atleast_1d(amps):
            disp_frames.append((float(a) * v, f"ladder:{name}"))
    for names, vectors, amp_arrays in plan.coupled_vectors:
        if len(names) != len(vectors) or len(names) != len(amp_arrays):
            raise ValueError(
                "coupled_vectors needs one name, vector, and amplitude array "
                "for every coupled direction"
            )
        vs = []
        for name, vector in zip(names, vectors):
            v = np.asarray(vector, dtype=float)
            if v.shape != (nQ,):
                raise ValueError(
                    f"coupled vector {name!r} must have shape ({nQ},)"
                )
            if not np.isfinite(v).all() or np.linalg.norm(v) == 0.0:
                raise ValueError(
                    f"coupled vector {name!r} must be finite and nonzero"
                )
            vs.append(v / np.linalg.norm(v))
        for combo in product(*[np.atleast_1d(a) for a in amp_arrays]):
            Q = sum(float(a) * v for a, v in zip(combo, vs))
            disp_frames.append((Q, "coupled:" + "+".join(names)))
    if plan.n_random:
        rng = np.random.default_rng(plan.seed)
        for _ in range(plan.n_random):
            Q = rng.uniform(-plan.random_amp, plan.random_amp, size=nQ)
            disp_frames.append((Q, "random"))

    frames = []
    for Q, prov in disp_frames:
        for strain in plan.strains:
            sv = None if strain is None else np.asarray(strain, dtype=float)
            frames.append(
                FrameSpec(Q=Q.copy(), strain_voigt=sv, provenance=prov,
                          split=plan.split)
            )
    return frames


def voigt_to_matrix(strain_voigt) -> np.ndarray:
    """Voigt (xx, yy, zz, yz, xz, xy) -> symmetric 3x3 strain matrix."""
    v = np.asarray(strain_voigt, dtype=float)
    eps = np.zeros((3, 3))
    eps[0, 0], eps[1, 1], eps[2, 2] = v[0], v[1], v[2]
    eps[1, 2] = eps[2, 1] = v[3]
    eps[0, 2] = eps[2, 0] = v[4]
    eps[0, 1] = eps[1, 0] = v[5]
    return eps


def make_atoms(mylwfsc, frame: FrameSpec):
    """Build the distorted (+ optionally strained) supercell Atoms of a frame.

    Displacement from the fixed ``MyLWFSC.get_distorted_atoms``; if the frame
    carries a strain, the cell is deformed as ``cell @ (I + eps)`` with
    positions scaled (see module docstring for the exact strain measure).
    """
    atoms, _ = mylwfsc.get_distorted_atoms(frame.Q)
    if frame.strain_voigt is not None:
        eps = voigt_to_matrix(frame.strain_voigt)
        atoms.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return atoms


def project_forces(M, forces) -> np.ndarray:
    """Project supercell forces onto the lwf amplitude coordinates.

    forces: (3 * natom_sc,) atom-major xyz, matching the rows of M.
    Returns ``g_Q = M.T @ forces``; note dE/dQ = -g_Q under the ASE sign
    convention F = -dE/du.
    """
    return np.asarray(M.T @ np.asarray(forces, dtype=float).ravel())
