"""Space-group action S_g(q) on the mass-weighted Cartesian displacement basis.

Story-014 core: the (3N, 3N) unitary representation matrices of the space
group acting on phonon eigenvectors at commensurate wavevectors, in lawaf's
gauge, with the phase convention validated by the eigenvector-transport test
(story-014 oracle evidence) and consistent with the story-010 seed convention
``v_lawaf(kappa) = e^{+2*pi*i r_kappa.q} v_spgrep(kappa)``.

Conventions (all validated, see tests/test_anharmonic_representation.py)
-----------------------------------------------------------------------
Ordering
    Atoms of the PRIMITIVE cell in phonopy ``primitive`` order; displacement
    components atom-major xyz (atom 0 x,y,z; atom 1 x,y,z; ...), matching
    ``PhonopyWrapper.solve`` / dynamical-matrix layout.

Space-group operations
    spglib dataset operations ``(W | w)``: integer ``W`` acting on reduced
    (fractional) coordinates as column-vector maps ``x' = W x + w``; ``w`` a
    fractional translation of the primitive lattice.  ``W`` maps the crystal
    onto itself: atom ``kappa`` at reduced ``r_kappa`` maps to atom
    ``sigma_g(kappa)`` with the lattice-defect vector
    ``t_kappa = W r_kappa + w - r_{sigma_g(kappa)}`` (integer).

Wavevector image (star convention)
    ``q --g--> q' = W^{-T} q  (mod 1)`` -- the transpose-inverse action used
    by phonopy/spgrep; the dynamical matrices obey ``D(q') S_g(q) = S_g(q)
    D(q)``.

Cartesian rotation
    ``R_g = A W_g A^{-1}`` with ``A`` the cell matrix whose ROWS are the
    primitive lattice vectors; Cartesian displacement components transform as
    ``u -> R_g u``.  For reduced coords ``W`` itself is the fractional-image
    rotation; only ``R_g`` acts on displacement components.

Phase law (validated)
    The implemented, transport-validated block is

        [S_g(q)]_{sigma(kappa), kappa} = R_g * exp(-2*pi*i * q' . t_kappa)

    with ``q' = W^{-T} q``.  This is the physical transporter derived from
    ``u'(x) = R u(g^{-1} x)``; the nonsymmorphic defect enters through the
    SOURCE-atom phase evaluated at the IMAGE point.  Ground truth
    (story-014 oracle): on the BaTiO3 ``DM_dip_wang`` fixture this form
    transports every lawaf-gauge eigenvector into the eigenspace at ``g.q``
    to ~1e-13 across all 48 operations x 15 bands, while transposed
    source/target placement of the rotation blocks fails at ~1e-1; unitarity
    and the group law hold to machine precision on commensurate meshes.
    Lawaf dynamical matrices relate to phonopy's by
    ``H_lawaf(q) = E(q)^{-1} D(q) E(q)`` with
    ``E(q) = diag(e^{-2*pi*i r_kappa.q} (x) I_3)`` (verified to 1e-17); on
    symmorphic fixtures such as BaTiO3 the gauge conjugation
    ``E(q')^{-1} S E(q)`` of this matrix is numerically identical to ``S``
    itself, which is why the bare form is the validated lawaf-gauge law.
    At Gamma all phases vanish: ``S_g(0)`` is the real permutation-rotation
    ``R_g`` block-permuted by ``sigma_g``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import spglib

from lawaf.utils.kpoints import reciprocal_point_key, reciprocal_point_name

__all__ = [
    "SpaceGroupAction",
    "WindowBands",
    "WindowLegalityBlock",
    "build_space_group_action",
    "check_window_legality",
    "SeedsLabels",
]


@dataclass(frozen=True)
class SpaceGroupAction:
    """Immutable space-group action on primitive-cell displacements.

    Attributes
    ----------
    rotations:
        ``(nsym, 3, 3)`` integer spglib rotations (reduced coordinates).
    translations:
        ``(nsym, 3)`` fractional translations of the primitive lattice.
    symmetry_dataset:
        The spglib dataset object from the single symmetry analysis performed
        at build time (also exposes ``.number``, ``.international``, ...).
    scaled_positions:
        ``(natom, 3)`` reduced coordinates of the primitive basis atoms.
    """

    rotations: np.ndarray
    translations: np.ndarray
    symmetry_dataset: object
    scaled_positions: np.ndarray
    symprec: float = 1e-5
    _aux: dict = field(default_factory=dict, repr=False, compare=False)

    # -- derived (computed once; matrix() itself is pure per-(g, q) algebra) --

    @cached_property
    def n_ops(self) -> int:
        return len(self.rotations)

    @cached_property
    def n_atoms(self) -> int:
        return len(self.scaled_positions)

    @cached_property
    def cart_rotations(self) -> np.ndarray:
        """Real-space rotations acting on Cartesian displacement components."""
        cell = self._aux["cell"]
        inv = np.linalg.inv(cell)
        return np.array([cell @ W.astype(float) @ inv for W in self.rotations])

    @cached_property
    def atom_maps(self) -> np.ndarray:
        """``(nsym, natom)`` int: sigma_g(kappa), image atom of each source atom."""
        return self._aux["sigma"]

    @cached_property
    def defect_vectors(self) -> np.ndarray:
        """``(nsym, natom, 3)``: lattice-defect vectors t_kappa (integers)."""
        return self._aux["defects"]

    def qmap(self, op_index: int, qfrac) -> np.ndarray:
        """Image wavevector ``W^{-T} q`` wrapped into [0, 1)."""
        q = np.asarray(qfrac, dtype=float).reshape(3)
        W = self.rotations[op_index].astype(float)
        qp = np.linalg.solve(W.T, q)
        return qp - np.floor(qp + 1e-8)

    def matrix(self, op_index: int, qfrac) -> np.ndarray:
        """Lawaf-gauge representation matrix ``S_g(q)`` (complex, (3N, 3N)).

        Pure per-(g, q) computation: no symmetry analysis occurs here.
        """
        n = self.n_atoms
        q = np.asarray(qfrac, dtype=float).reshape(3)
        qp = self.qmap(op_index, q)
        R = self.cart_rotations[op_index]
        t_all = self.defect_vectors[op_index]
        sig = self.atom_maps[op_index]

        M = np.zeros((3 * n, 3 * n), dtype=complex)
        for kappa in range(n):
            s = sig[kappa]
            ex = -(qp @ t_all[kappa])
            M[3 * s:3 * s + 3, 3 * kappa:3 * kappa + 3] = np.exp(2j * np.pi * ex) * R
        return M

    def star(self, qfrac, tol: float = 1e-8) -> list:
        """Star of ``q``: ``[(op_index, q_image), ...]``.

        One entry per DISTINCT image point (wrapped into [0, 1)), carrying the
        lowest-indexed operation that maps ``q`` onto it.  The images are the
        orbit under the whole group; ``len(star)`` divides ``n_ops``.
        """
        q = np.asarray(qfrac, dtype=float).reshape(3)
        images = []
        reps = []
        for g in range(self.n_ops):
            qp = self.qmap(g, q)
            dup = False
            for seen in reps:
                d = qp - seen
                d -= np.rint(d)
                if np.linalg.norm(d) < tol:
                    dup = True
                    break
            if not dup:
                reps.append(qp)
                images.append((g, qp))
        return images

    def irreducible_qpoints(self, mesh, tol: float = 1e-8) -> np.ndarray:
        """One representative per star on the ``mesh = (n1, n2, n3)`` grid.

        Returns ``(n_ir, 3)`` wrapped fractional q-points in first-seen mesh
        order (standard IBZ enumeration).  The stars of the representatives
        partition the full mesh.
        """
        mesh = np.asarray(mesh, dtype=int).reshape(3)
        reps = []
        star_members = []
        for i in range(mesh[0]):
            for j in range(mesh[1]):
                for k in range(mesh[2]):
                    q = np.array([i / mesh[0], j / mesh[1], k / mesh[2]])
                    if any(
                        np.linalg.norm((q - r) - np.rint(q - r)) < tol
                        for r in star_members
                    ):
                        continue
                    reps.append(q)
                    star_members.extend(qp for _, qp in self.star(q, tol=tol))
        return np.array(reps)



@dataclass(frozen=True)
class WindowLegalityBlock:
    """Little-group legality result for one selected frequency block."""

    frequency: float
    dimension: int
    irreps: tuple[str, ...]
    residual: float


@dataclass(frozen=True)
class WindowBands:
    """Validated representatives, expanded bands, and per-arm legality."""

    representatives: dict[tuple[float, float, float], tuple[int, ...]]
    bands: dict[tuple[float, float, float], tuple[int, ...]]
    legality: dict[
        tuple[float, float, float],
        tuple[WindowLegalityBlock, ...],
    ]


def _window_qkey(qpoint) -> tuple[float, float, float]:
    """Wrapped, stable fractional q-point key for a window specification."""
    return reciprocal_point_key(qpoint)


def _window_qname(qpoint: tuple[float, float, float]) -> str:
    """Conventional cubic label where available, otherwise a q-point tuple."""
    return reciprocal_point_name(qpoint)


def _window_eigensystem(phonon, qpoint):
    """Return sorted eigenpairs and fixture-unit frequencies at ``qpoint``."""
    obj = phonon.phonon if hasattr(phonon, "phonon") else phonon
    solved_in_lawaf_gauge = hasattr(phonon, "solve") and not hasattr(
        phonon, "get_dynamical_matrix_at_q"
    )
    if solved_in_lawaf_gauge:
        evals, evecs = phonon.solve(qpoint)
    elif hasattr(obj, "run_qpoints"):
        # Use phonopy's native q-point path: it includes the full NAC term
        # when the supplied phonon object has ``nac_params``.  Only the
        # eigenvectors need lawaf's atom-Bloch rephasing; altering the
        # dynamical matrix itself changes its spectrum at zone boundaries.
        obj.run_qpoints([qpoint], with_eigenvectors=True)
        evals = np.asarray(obj.qpoints.eigenvalues[0], dtype=float)
        evecs = np.asarray(obj.qpoints.eigenvectors[0], dtype=complex)
        phases = np.repeat(
            np.exp(
                2j * np.pi
                * (np.asarray(obj.primitive.scaled_positions) @ np.asarray(qpoint))
            ),
            3,
        )
        evecs = phases[:, None] * evecs
    else:
        raise TypeError(
            "check_window_legality needs a phonopy object or wrapper exposing "
            "solve(q)"
        )
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs, dtype=complex)
    factor = float(getattr(obj, "unit_conversion_factor", 1.0))
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * factor
    return evals, evecs, freqs


def _relative_degeneracy_blocks(freqs, tol: float) -> tuple[tuple[int, int], ...]:
    """Consecutive sorted-frequency blocks separated by relative eigengaps."""
    freqs = np.asarray(freqs, dtype=float)
    if len(freqs) == 0:
        return ()
    scale = max(float(np.max(np.abs(freqs))), np.finfo(float).tiny)
    blocks = []
    start = 0
    for i in range(len(freqs) - 1):
        if abs(freqs[i + 1] - freqs[i]) / scale > tol:
            blocks.append((start, i + 1))
            start = i + 1
    blocks.append((start, len(freqs)))
    return tuple(blocks)



def _block_irreps(sga, qpoint, vectors) -> tuple[tuple[str, ...], int]:
    """Character-project one degenerate eigenspace into little-group irreps."""
    from lawaf.anharmonic.compatibility import character_table, little_group

    ops = little_group(sga, qpoint)
    table = character_table(sga, ops)
    chars = {
        op: np.trace(vectors.conj().T @ sga.matrix(op, qpoint) @ vectors)
        for op in ops
    }
    mult = table.multiplicities(chars)
    labels = []
    dimensions = []
    for name, dim, value in zip(table.names, table.dims, mult):
        count = int(round(value.real))
        if abs(value.real - count) > 1e-4 or abs(value.imag) > 1e-4:
            raise ValueError(
                f"non-integer irrep multiplicity {value} for {name} at "
                f"q={_window_qname(_window_qkey(qpoint))}"
            )
        if count:
            labels.append(name)
            dimensions.append(dim)
    if not labels:
        raise ValueError(
            f"no little-group irreps found at q={_window_qname(_window_qkey(qpoint))}"
        )
    return tuple(labels), max(dimensions)


def _subspace_invariance_residual(sga, qpoint, vectors) -> float:
    """Maximum little-group leakage of an orthonormal selected subspace."""
    from lawaf.anharmonic.compatibility import little_group

    vectors = np.asarray(vectors, dtype=complex)
    projector = vectors @ vectors.conj().T
    scale = np.sqrt(vectors.shape[1])
    return max(
        float(
            np.linalg.norm(
                (np.eye(len(projector)) - projector)
                @ sga.matrix(op, qpoint)
                @ vectors,
                ord="fro",
            )
            / scale
        )
        for op in little_group(sga, qpoint)
    )


def _validate_window_at_q(sga, phonon, qkey, bands, degeneracy_tol):
    """Validate complete eigenspace selection and return its decomposition."""
    _evals, evecs, freqs = _window_eigensystem(phonon, qkey)
    if evecs.shape[0] != 3 * sga.n_atoms:
        raise ValueError(
            "phonon eigenspace dimension does not match the space-group action"
        )
    nband = len(freqs)
    if not bands:
        raise ValueError(f"window at q={_window_qname(qkey)} selects no bands")
    if bands[0] < 0 or bands[-1] >= nband:
        raise ValueError(
            f"window at q={_window_qname(qkey)} contains band outside "
            f"[0, {nband - 1}]"
        )

    selected = set(bands)
    decomposition = []
    for start, stop in _relative_degeneracy_blocks(freqs, degeneracy_tol):
        block = set(range(start, stop))
        overlap = selected & block
        if not overlap:
            continue
        selected_indices = sorted(overlap)
        selected_vectors = evecs[:, selected_indices]
        residual = _subspace_invariance_residual(
            sga, qkey, selected_vectors
        )
        if residual > 1e-6:
            labels, irrep_dim = _block_irreps(
                sga, qkey, evecs[:, start:stop]
            )
            raise ValueError(
                f"illegal window at q={_window_qname(qkey)}: straddles "
                f"degenerate block at frequency {freqs[start]:.3f} "
                f"(dim {stop - start}; irrep dimension {irrep_dim}: "
                f"{', '.join(labels)}; residual {residual:.3e})"
            )
        labels, _irrep_dim = _block_irreps(
            sga, qkey, selected_vectors
        )
        decomposition.append(
            WindowLegalityBlock(
                frequency=float(freqs[start]),
                dimension=len(selected_indices),
                irreps=labels,
                residual=residual,
            )
        )
    return tuple(decomposition), _relative_degeneracy_blocks(freqs, degeneracy_tol)


def check_window_legality(
    sga: SpaceGroupAction,
    phonon,
    window_bands,
    degeneracy_tol: float | None = None,
) -> WindowBands:
    """Validate explicit per-q windows before any Wannier gauge work.

    Every selected band set must be a union of complete sorted-frequency
    eigenspaces.  The eigengap threshold is relative to the q-point spectrum,
    so the BaTiO3 M-point A/E split is not merged merely because it is small
    in cm-1.  Each listed representative is expanded over its star; the
    sorted-frequency block structure of every arm is checked before sharing
    the representative's band indices.
    """
    if not isinstance(window_bands, dict):
        raise TypeError("window_bands must be a dict mapping q-points to band indices")
    if not window_bands:
        raise ValueError("window_bands must contain at least one representative")
    tol = 1e-5 if degeneracy_tol is None else float(degeneracy_tol)
    if tol <= 0.0:
        raise ValueError("degeneracy_tol must be positive")

    requested = {}
    for qpoint, ibands in window_bands.items():
        qkey = _window_qkey(qpoint)
        try:
            raw_bands = tuple(ibands)
        except TypeError as exc:
            raise TypeError(
                f"window bands at q={_window_qname(qkey)} must be iterable"
            ) from exc
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            for index in raw_bands
        ):
            raise ValueError(
                f"window bands at q={_window_qname(qkey)} must be integer indices"
            )
        bands = tuple(sorted(int(index) for index in raw_bands))
        if len(set(bands)) != len(bands):
            raise ValueError(
                f"window at q={_window_qname(qkey)} repeats a band index"
            )
        if qkey in requested and requested[qkey] != bands:
            raise ValueError(
                f"same wrapped q={_window_qname(qkey)} has conflicting "
                "band selections"
            )
        requested[qkey] = bands

    bands_out = {}
    legality = {}
    for qkey, bands in requested.items():
        decomposition, block_structure = _validate_window_at_q(
            sga, phonon, qkey, bands, tol
        )
        for _op, arm in sga.star(qkey):
            armkey = _window_qkey(arm)
            if armkey in requested and requested[armkey] != bands:
                raise ValueError(
                    f"star arm q={_window_qname(armkey)} must use the same "
                    f"band indices as q={_window_qname(qkey)}"
                )
            arm_decomposition, arm_structure = _validate_window_at_q(
                sga, phonon, armkey, bands, tol
            )
            if arm_structure != block_structure:
                raise ValueError(
                    f"star arm q={_window_qname(armkey)} does not share the "
                    f"sorted-frequency block structure of "
                    f"q={_window_qname(qkey)}"
                )
            bands_out[armkey] = bands
            legality[armkey] = arm_decomposition
    return WindowBands(
        representatives=dict(requested),
        bands=bands_out,
        legality=legality,
    )


def build_space_group_action(phonon_or_atoms, symprec: float = 1e-5) -> SpaceGroupAction:
    """Build :class:`SpaceGroupAction` from an ase.Atoms or phonopy object.

    Accepts ``ase.Atoms``, ``phonopy.Phonopy`` (uses its ``primitive``), or any
    wrapper exposing a ``.phonon`` attribute (e.g.
    ``lawaf.interfaces.phonopy.phonopywrapper.PhonopyWrapper``).  The spglib
    symmetry analysis runs ONCE here; every later ``matrix()``/``star()`` call
    is pure algebra.
    """
    obj = phonon_or_atoms
    if hasattr(obj, "phonon"):
        obj = obj.phonon
    if hasattr(obj, "primitive"):  # phonopy.Phonopy
        prim = obj.primitive
        cell = np.array(prim.cell)
        spos = np.array(prim.scaled_positions)
        numbers = np.array(prim.numbers)
    else:  # ase.Atoms
        cell = np.array(obj.cell[:])
        spos = obj.get_scaled_positions(wrap=True)
        numbers = np.array(obj.get_atomic_numbers())
    dataset = spglib.get_symmetry_dataset((cell, spos, numbers), symprec=symprec)
    rotations = np.array(dataset.rotations, dtype=int)
    translations = np.array(dataset.translations, dtype=float)
    nsym = len(rotations)
    natom = len(spos)
    sigma = np.empty((nsym, natom), dtype=int)
    defects = np.empty((nsym, natom, 3), dtype=float)
    for g in range(nsym):
        W = rotations[g].astype(float)
        w = translations[g]
        tg = spos @ W.T + w
        for kappa in range(natom):
            d = tg[kappa][None, :] - spos
            d -= np.rint(d)
            dists = np.linalg.norm(d, axis=1)
            j = int(np.argmin(dists))
            if dists[j] > 1e-5:
                raise ValueError(
                    f"operation {g} does not map atom {kappa} onto a basis atom"
                )
            sigma[g, kappa] = j
            t_raw = tg[kappa] - spos[j]
            t_int = np.rint(t_raw)
            if np.abs(t_raw - t_int).max() > 1e-6:
                raise ValueError(f"non-lattice defect vector for op {g}, atom {kappa}")
            defects[g, kappa] = t_int

    return SpaceGroupAction(
        rotations=rotations,
        translations=translations,
        symmetry_dataset=dataset,
        scaled_positions=spos,
        symprec=symprec,
        _aux={"cell": cell, "sigma": sigma, "defects": defects},
    )


# ----------------------------------------------------------------------
# story-024: seeds label provider (ADR-011 adapter)
# ----------------------------------------------------------------------
class SeedsLabels:
    """Found-irrep labels from symmetry seeds (story-024, ADR-011 adapter).

    Implements the
    :class:`~lawaf.anharmonic.compatibility.RepresentationLabels` protocol
    on top of ``lawaf.interfaces.phonopy.symmetry_seeds``: every retained
    band touched by a seeded eigenspace is labeled by the daughter space
    group of its selected OPD family plus the direction character of its
    own seed column (the ADR-005 axis taxonomy, reused from the seeds
    module as its single owner), e.g. ``"P4mm axis-pure (z)"`` for one
    branch of the BaTiO3 soft triplet at Gamma.

    ``bands`` are lawaf band indices of the retained window (as in
    ``params.anchors``); ``opd``/``opd_index`` select OPD families as in
    ``get_symmetry_anchor_wfn``.  ``symprec=1e-5`` mirrors the seeds
    module's ``DEFAULT_SYMPREC`` (kept a literal: a module-level import is
    banned here).  A guard fallback at an anchor raises :class:`ValueError`
    naming the reason — an unlabeled window is never silently reported.

    The seeds module is imported lazily inside :meth:`irreps_at`: no
    ``anharmonic`` module may import it at module import time (story-024
    boundary test).
    """

    def __init__(
        self,
        phonon,
        bands,
        opd=None,
        opd_index=None,
        symprec: float = 1e-5,
        degeneracy_tol: float | None = None,
    ):
        self.phonon = phonon
        self.bands = tuple(bands)
        self.opd = opd
        self.opd_index = opd_index
        self.symprec = symprec
        self.degeneracy_tol = degeneracy_tol

    def irreps_at(self, qpoint) -> list:
        # ADR-011 import boundary: lazy, adapter-internal only.
        from lawaf.interfaces.phonopy import symmetry_seeds

        report = symmetry_seeds.get_symmetry_anchor_wfn(
            self.phonon,
            qpoint,
            self.bands,
            symprec=self.symprec,
            degeneracy_tol=self.degeneracy_tol,
            opd=self.opd,
            opd_index=self.opd_index,
        )
        if report.fell_back:
            raise ValueError(
                "symmetry seed labels unavailable at q="
                f"{np.round(np.asarray(qpoint, dtype=float), 6).tolist()}: "
                + "; ".join(report.warnings)
            )
        labels = []
        for rec in report.per_band:
            direction = symmetry_seeds._axis_summary(report.psi[:, rec.band])
            labels.append(f"{rec.sg_symbol} {direction}")
        return labels
