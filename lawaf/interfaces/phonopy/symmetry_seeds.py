"""Symmetry-adapted phonon projector seeds via spgrep-modulation.

Story-010 seed core: band→eigenspace mapping, whole-eigenspace + svmin guard
with per-anchor fallback, lawaf-gauge conversion, commensurability
denominators, lazy optional import. Story-011: OPD-family listing
(``list_opd_families``), selection policy (default / SG number / index,
scalar or per-eigenspace mapping), canonical seed sets (OPD line +
little-group images, QR, deterministic ordering) and per-band records.
Downfolder wiring arrives with story-012.

Gauge law (research memo, finding 4; re-verified against
``PhonopyWrapper.solve``'s phase code)::

    v_lawaf(κ) = e^{+2πi r_κ·q} v_spgrep(κ)

spgrep-modulation is imported lazily: environments without the ``symmetry``
extra keep working (FR-005). Its 0.3.0 "Inconsistent eigenvalue" UserWarning
is benign (symphon precedent) and filtered here (NFR-003).
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

# NFR-003: spgrep-modulation 0.3.x emits a benign "Inconsistent eigenvalue"
# UserWarning during eigenspace decomposition (symphon precedent). Filter ONLY
# that message — other spgrep_modulation warnings (non-commensurate supercell,
# non-Hermitian matrix) must still propagate (review SPEC-003).
warnings.filterwarnings(
    "ignore",
    message=r"Inconsistent eigenvalue.*",
    category=UserWarning,
    module=r"spgrep_modulation",
)

MAX_SUPERCELL_DENOMINATOR = 12
DEFAULT_SYMPREC = 1e-5
# Eigenvalue-matching tolerance between lawaf's eigh output and
# Modulation.eigenspaces eigenvalues (units: lawaf solve() eigenvalues,
# cm^-2-ish for the validated BaTiO3 fixture; distinct eigenspaces there are
# separated by ~2e-1, degenerate partners agree to ~1e-12).
DEFAULT_DEGENERACY_TOL = 1e-3
SVMIN_THRESHOLD = 1 - 1e-6

@dataclass
class SeedBandRecord:
    """Per-band seed metadata (FR-006; review SPECQ-004 typed contract)."""

    band: int
    eigenspace_index: int
    eigenspace_dim: int
    irrep_chars: dict
    family_index: int | None
    sg_number: int | None
    sg_symbol: str | None
    direction_summary: str | None
    frequency: float


@dataclass
class OPDFamily:
    """One order-parameter-direction family of one eigenspace (FR-010).

    ``index`` is globally unique across the listing but unstable across
    spgrep-modulation versions; ``sg_number`` is the stable identifier.
    ``ndir == 1`` families are selectable as seed lines.
    """

    index: int
    sg_number: int
    sg_symbol: str
    subgroup_order: int
    ndir: int
    eigenspace_index: int
    frequency: float
    direction_summary: str

    def __str__(self):
        return (
            f"OPD {self.index:>3}: {self.sg_symbol:<11s} (#{self.sg_number:<3d}) "
            f"order={self.subgroup_order:<3d} ndir={self.ndir} "
            f"eigenspace={self.eigenspace_index} "
            f"freq={self.frequency:9.4f} {self.direction_summary}"
        )


@dataclass
class SymmetrySeedReport:
    """Result of a symmetry-seed construction at one anchor q.

    ``psi`` is the full (3N, 3N) eigenvector matrix in lawaf's gauge with the
    touched degenerate blocks replaced by symmetry vectors (story-010:
    eigenspace basis; story-011: OPD seeds); untouched columns are lawaf's
    own. ``per_band`` is populated from story-011 on (SeedBandRecord).
    """

    qpoint: tuple
    psi: np.ndarray
    per_band: list = field(default_factory=list)
    fell_back: bool = False
    warnings: list = field(default_factory=list)


class _GuardError(Exception):
    """Internal: guard violation at an anchor (mapped to fallback + warning)."""


def _import_modulation():
    try:
        from spgrep_modulation.modulation import Modulation
    except ImportError as exc:  # FR-005: optional dependency
        raise RuntimeError(
            "symmetry_seed requires spgrep-modulation: "
            "pip install lawaf[symmetry]"
        ) from exc
    return Modulation


def smallest_denominators(qpoint, max_denominator=MAX_SUPERCELL_DENOMINATOR):
    """Smallest per-component d ≤ max_denominator with d·q integral.

    Returns an int array (d1, d2, d3) or None when some component has no such
    denominator (→ the anchor q is not commensurate, FR-008 fallback).
    """
    q = np.asarray(qpoint, dtype=float)
    denoms = []
    for qi in q:
        for d in range(1, max_denominator + 1):
            if np.isclose(d * qi, round(d * qi), atol=1e-6):
                denoms.append(d)
                break
        else:
            return None
    return np.array(denoms, dtype=int)


def build_modulation(phonon, qpoint, symprec=DEFAULT_SYMPREC, degeneracy_tol=None):
    """``Modulation`` for the smallest commensurate supercell at ``qpoint``.

    Raises ValueError when q is not commensurate (no denominator ≤ 12).
    """
    Modulation = _import_modulation()
    q = np.asarray(qpoint, dtype=float)
    denoms = smallest_denominators(q)
    if denoms is None:
        raise ValueError(
            f"q-point {np.round(q, 6).tolist()} is not commensurate with any "
            f"supercell with denominators <= {MAX_SUPERCELL_DENOMINATOR}"
        )
    kwargs = dict(
        dynamical_matrix=phonon.dynamical_matrix,
        supercell_matrix=np.diag(denoms),
        qpoint=q,
        factor=phonon.unit_conversion_factor,
        symprec=symprec,
    )
    if degeneracy_tol is not None:
        kwargs["degeneracy_tolerance"] = degeneracy_tol
    return Modulation.with_supercell_and_symmetry_search(**kwargs)


def _reference_solve(phonon, qpoint):
    """Cache-free lawaf reference solve at an anchor q (ADR-003).

    ``use_cache=False`` bypasses the on-disk ``./phon_cache`` pickle, whose
    key carries no model identity (feasibility finding FEAS-002). The
    wrapper's ``_prepare`` symmetrizes force constants IN PLACE, and
    phonopy's symmetrizer is not bitwise idempotent (~1e-14 drift per
    application) — that drift re-gauges exactly-degenerate eigenspaces and
    breaks bitwise determinism of the OPD seeds (NFR-001). Snapshot and
    restore the force constants so the phonon object leaves this function
    pristine.
    """
    from .phonopywrapper import PhonopyWrapper

    snapshot = phonon.force_constants.copy()
    try:
        model = PhonopyWrapper(
            phonon, mode="dm", is_nac=False, use_cache=False
        )
        return model.solve(np.asarray(qpoint, dtype=float))
    finally:
        phonon.force_constants[...] = snapshot


def _match_bands(evals_ref, eigenspaces, tol):
    """Map lawaf band indices → eigenspace indices, matched by eigenvalue.

    Both spectra are sorted ascending and share units (validated on BaTiO3:
    lawaf ``solve()`` eigenvalues and ``Modulation`` eigenspace eigenvalues
    agree to ~1e-8). Eigenspaces are sorted explicitly, bands consumed in
    order (each eigenspace owns one contiguous block), and every block's
    eigenvalues are verified against the eigenspace's eigenvalue within
    ``tol`` (FR-002/ADR-003). Total coverage must be complete (Σ dims = 3N).
    """
    ordered = sorted(eigenspaces, key=lambda es: es[0])
    # pass 1: dimension bookkeeping and completeness — checked BEFORE any
    # indexing so over-complete decompositions cannot raise IndexError
    dims = [es[1].shape[0] for es in ordered]
    total = sum(dims)
    if total != len(evals_ref):
        raise _GuardError(
            f"eigenspaces cover {total} modes but lawaf solved "
            f"{len(evals_ref)}: incomplete eigenspace decomposition"
        )
    # pass 2: contiguous block assignment + eigenvalue verification
    groups: dict = {}
    start = 0
    for new_index, (eigval, _eigvecs, _irrep) in enumerate(ordered):
        dim = dims[new_index]
        block = list(range(start, start + dim))
        deviation = float(
            np.max(np.abs(np.asarray(evals_ref)[block] - eigval))
        )
        if deviation > tol:
            raise _GuardError(
                f"eigenspace {new_index} (eigenvalue {eigval:.6f}) does not "
                f"match lawaf eigenvalues of bands {block} within tol {tol} "
                f"(max deviation {deviation:.3e})"
        )
        groups[new_index] = block
        start += dim
    return ordered, groups


def _distinct_images(coeffs, irrep):
    """Distinct little-group image LINES of an OPD coefficient vector.

    Each image is canonicalized in phase (largest-magnitude component made
    real positive) so ±c and other phase-variant copies of one line collapse
    together; distinct lines are returned in a deterministic sort order.
    """
    seen = {}
    for matrix in irrep:
        image = matrix @ coeffs
        pivot = int(np.argmax(np.abs(image)))
        factor = image[pivot]
        if abs(factor) < 1e-12:
            continue
        canonical = image / factor
        key = tuple(np.round(canonical.real, 8)) + tuple(
            np.round(canonical.imag, 8)
        )
        seen.setdefault(key, canonical)

    def sort_key(vec):
        rounded = np.round(vec, 8)
        mag = np.abs(rounded)
        return (int(np.argmax(mag)), tuple(mag), tuple(rounded.real))

    return sorted(seen.values(), key=sort_key)


def _axis_summary(displacement):
    """Direction character of a seed displacement (ADR-005 taxonomy).

    Axis weights w_a = sqrt(sum_kappa |u_a(kappa)|^2), classified at
    tolerance 1e-8: axis-pure / face-diagonal / body-diagonal / generic,
    dominant axes appended.
    """
    w = np.sqrt((np.abs(displacement).reshape(-1, 3) ** 2).sum(axis=0))
    total = w.sum()
    if total == 0:
        return "generic"
    w = w / total
    order = np.argsort(-w)
    tol = 1e-8
    axes = "xyz"
    if w[order[1]] <= tol:
        return f"axis-pure ({axes[order[0]]})"
    if w[order[2]] <= tol and abs(w[order[0]] - w[order[1]]) <= tol:
        pair = sorted(axes[a] for a in order[:2])
        return f"face-diagonal ({pair[0]},{pair[1]})"
    if (
        abs(w[order[0]] - w[order[1]]) <= tol
        and abs(w[order[1]] - w[order[2]]) <= tol
    ):
        return "body-diagonal"
    return "generic"



def _daughter_sg(modulation, frequency_index, coeffs):
    """Daughter space group of an OPD line, via modulation + spglib.

    spgrep-modulation returns no SG label itself (FEAS-003); apply the
    modulation and classify the distorted supercell.
    """
    import spglib

    amplitudes = list(np.abs(coeffs) * 0.1)
    arguments = list(np.angle(coeffs))
    cell, _mod = modulation.get_modulated_supercell_and_modulation(
        frequency_index, amplitudes, arguments, return_cell=True
    )
    dataset = spglib.get_symmetry_dataset(
        (cell.cell, cell.scaled_positions, cell.numbers), symprec=1e-4
    )
    if dataset is None:
        return None, None
    return int(dataset.number), str(dataset.international)




def _irrep_chars(irrep):
    return {
        "little_group_order": int(len(irrep)),
        "characters": [round(float(np.trace(m).real), 6) for m in irrep],
    }


def _families_for(modulation, eigval, eigvecs, irrep, vecs_gauge, orig, start):
    """OPD families of ONE eigenspace, numbered from ``start``.

    Returns ``(rows, coeffs, next_start)``: rows are ``OPDFamily`` records
    and ``coeffs`` maps family index → first OPD coefficient vector.
    """
    from spgrep_modulation.isotropy import IsotropyEnumerator

    ie = IsotropyEnumerator(
        modulation.little_rotations,
        modulation.little_translations,
        modulation.qpoint,
        irrep,
    )
    freq = float(modulation.eigvals_to_frequencies(eigval))
    dim = eigvecs.shape[0]
    rows = []
    coeffs = {}
    for opd, subgroups in zip(
        ie.order_parameter_directions, ie.maximal_isotropy_subgroups
    ):
        first = np.asarray(opd[0])
        coeffs[start] = first
        displacement = vecs_gauge @ first
        sg_number, sg_symbol = _daughter_sg(modulation, orig, first)
        rows.append(
            OPDFamily(
                index=start,
                sg_number=sg_number if sg_number is not None else -1,
                sg_symbol=sg_symbol if sg_symbol is not None else "?",
                subgroup_order=len(subgroups),
                ndir=len(opd),
                eigenspace_index=None,  # set by the caller (sorted index)
                frequency=freq,
                direction_summary=_axis_summary(displacement),
            )
        )
        start += 1
    return rows, coeffs, start, dim


def _enumerate_families(modulation, ordered_with_orig, eigvecs_gauge, touched):
    """All OPD families per touched (sorted) eigenspace, globally unique.

    ``ordered_with_orig``: list of (orig_index, eigval, eigvecs, irrep) in
    sorted-by-eigenvalue order. ``eigvecs_gauge``: per sorted index j, the
    (3N, dim) lawaf-gauge eigenspace vectors. Returns ({j: [OPDFamily,...]},
    {family_index: first OPD coefficient vector}).
    """
    families: dict = {}
    coeffs_of: dict = {}
    start = 0
    for j, (orig, eigval, eigvecs, irrep) in enumerate(ordered_with_orig):
        if j not in touched:
            continue
        rows, coeffs, start, _dim = _families_for(
            modulation, eigval, eigvecs, irrep, eigvecs_gauge[j], orig, start
        )
        for row in rows:
            row.eigenspace_index = j
        families[j] = rows
        coeffs_of.update(coeffs)
    return families, coeffs_of


def _touched(groups, bands):
    """Eigenspace indices whose band blocks intersect the anchor bands."""
    anchor = set(bands)
    return {j for j, group in groups.items() if anchor.intersection(group)}
def _resolve_family(rows, eigenspace_index, selector, is_index, all_rows):
    """Choose one ndir==1 family for an eigenspace (ADR-005 semantics).

    ``selector=None`` → the default rule: highest subgroup order, ties to
    the lowest global index. Integer selectors match the daughter SG
    number (``is_index=False``) or the global family index
    (``is_index=True``). No match raises ValueError naming valid families.
    """
    candidates = [f for f in rows if f.ndir == 1]
    if selector is None:
        if not candidates:
            raise ValueError(
                f"eigenspace {eigenspace_index} has no 1D OPD family"
            )
        return max(candidates, key=lambda f: (f.subgroup_order, -f.index))
    if is_index:
        for f in candidates:
            if f.index == selector:
                return f
    else:
        matches = [f for f in candidates if f.sg_number == selector]
        if matches:
            return max(matches, key=lambda f: (f.subgroup_order, -f.index))
    valid = ", ".join(f"#{f.index}->{f.sg_symbol}(#{f.sg_number})" for f in candidates)
    raise ValueError(
        f"selector {selector!r} matches no 1D OPD family in eigenspace "
        f"{eigenspace_index}; valid families: {valid or 'none'}; "
        f"see list_opd_families: {[str(f) for f in all_rows]}"
    )



def _resolve_selection(families, j, opd, opd_index):
    """Resolve the family for eigenspace j given opd / opd_index API args.

    Semantics (ADR-005 + refinement for globally-unique indices):
    - mapping {eigenspace_index: selector}: per-block control; unmentioned
      blocks use the default rule;
    - scalar ``opd`` (SG number): must match in EVERY touched eigenspace,
      else ValueError naming the failing block;
    - scalar ``opd_index`` (global family index): applies to its owning
      eigenspace; other touched eigenspaces use the default rule; a
      nonexistent index raises ValueError.
    """
    rows = families[j]
    for mapping, is_index in ((opd, False), (opd_index, True)):
        if isinstance(mapping, dict):
            if j in mapping:
                return _resolve_family(rows, j, mapping[j], is_index, rows)
            return _resolve_family(rows, j, None, False, rows)
    if opd is not None:  # scalar SG number
        return _resolve_family(rows, j, opd, False, rows)
    if opd_index is not None:  # scalar global index
        if any(f.index == opd_index for f in rows):
            return _resolve_family(rows, j, opd_index, True, rows)
        exists = any(
            f.index == opd_index for fs in families.values() for f in fs
        )
        if not exists:
            raise ValueError(
                f"opd_index={opd_index} matches no family; "
                "see list_opd_families"
            )
        return _resolve_family(rows, j, None, False, rows)
    return _resolve_family(rows, j, None, False, rows)


def list_opd_families(
    phonon,
    qpoint,
    bands,
    symprec=DEFAULT_SYMPREC,
    degeneracy_tol=None,
):
    """List OPD families of the eigenspaces touched by ``bands`` (FR-010).

    Returns ``list[OPDFamily]`` (globally unique indices) and prints the
    same as a table.
    """
    q = np.asarray(qpoint, dtype=float)
    bands = tuple(bands)
    n_modes = 3 * len(phonon.primitive)
    if not bands or any(not (0 <= b < n_modes) for b in bands):
        raise ValueError(f"invalid anchor bands {bands} for {n_modes} modes")
    match_tol = (
        DEFAULT_DEGENERACY_TOL if degeneracy_tol is None else degeneracy_tol
    )
    evals_ref, _ = _reference_solve(phonon, q)
    modulation = build_modulation(
        phonon, q, symprec=symprec, degeneracy_tol=degeneracy_tol
    )
    ordered, groups = _match_bands(evals_ref, modulation.eigenspaces, match_tol)
    _check_whole_eigenspaces(groups, bands)
    ordered_with_orig, eigvecs_gauge = _prepare_eigenspaces(
        modulation, ordered, phonon.primitive.scaled_positions, q
    )
    families, _coeffs = _enumerate_families(
        modulation, ordered_with_orig, eigvecs_gauge, _touched(groups, bands)
    )
    rows = [f for j in sorted(families) for f in families[j]]
    print("\n".join(str(f) for f in rows))
    return rows


def _prepare_eigenspaces(modulation, ordered, scaled_positions, q):
    """Sorted eigenspaces with original indices + lawaf-gauge vectors."""
    orig_of = {id(es): i for i, es in enumerate(modulation.eigenspaces)}
    ordered_with_orig = [
        (orig_of[id(es)], es[0], es[1], es[2]) for es in ordered
    ]
    eigvecs_gauge = {}
    for j, (_orig, _eigval, eigvecs, _irrep) in enumerate(ordered_with_orig):
        dim = eigvecs.shape[0]
        eigvecs_gauge[j] = gauge_convert(
            eigvecs.reshape(dim, -1).conj().T, scaled_positions, q
        )
    return ordered_with_orig, eigvecs_gauge



def _check_whole_eigenspaces(groups, bands):
    """Whole-eigenspace policy (ADR-003): every eigenspace touched by the
    anchor bands is covered completely; untouched eigenspaces pass through
    lawaf's own columns."""
    anchor = set(bands)
    for j, group in groups.items():
        touched = anchor.intersection(group)
        if not touched:
            continue
        missing = set(group) - anchor
        if missing:
            raise _GuardError(
                f"anchor bands {sorted(anchor)} cover only part of "
                f"eigenspace {j} (missing bands {sorted(missing)}): a "
                "partial selection cannot be symmetry-canonical"
            )


def gauge_convert(converted, scaled_positions, qpoint):
    """Convert (3N, dim) spgrep-gauge vectors to lawaf gauge.

    Per-atom factor e^{+2πi r_κ·q}, repeated over the 3 Cartesian components
    (atom-major flattening, matching lawaf's solve()).
    """
    q = np.asarray(qpoint, dtype=float)
    phase = np.exp(2j * np.pi * np.asarray(scaled_positions) @ q)
    return np.repeat(phase, 3)[:, None] * converted


def get_symmetry_anchor_wfn(
    phonon,
    qpoint,
    bands,
    symprec=DEFAULT_SYMPREC,
    degeneracy_tol=None,
    opd=None,
    opd_index=None,
):
    """Symmetry seed report for one anchor q (story-010 core semantics).

    ``bands`` are lawaf band indices (0-based, as in ``params.anchors``).
    Degenerate eigenspaces touched by ``bands`` are replaced by their
    spgrep-modulation eigenspace basis in lawaf gauge (validated by the
    svmin guard); any guard or commensurability failure falls back to lawaf's
    own eigenvectors with a warning (FR-002).
    """
    q = np.asarray(qpoint, dtype=float)
    bands = tuple(bands)
    n_modes = 3 * len(phonon.primitive)
    if not bands or any(not (0 <= b < n_modes) for b in bands):
        raise ValueError(f"invalid anchor bands {bands} for {n_modes} modes")

    def fallback(evecs_ref, reason):
        warnings.warn(
            f"symmetry_seed at q={np.round(q, 6).tolist()}: {reason}; "
            "falling back to raw anchor eigenvectors",
            stacklevel=2,
        )
        return SymmetrySeedReport(
            qpoint=tuple(q),
            psi=evecs_ref.copy(),
            fell_back=True,
            warnings=[reason],
        )

    evals_ref, evecs_ref = _reference_solve(phonon, q)
    # degeneracy_tol=None means "spgrep's default eigenspace grouping"; the
    # eigenvalue-matching guard always needs a numeric tolerance
    match_tol = (
        DEFAULT_DEGENERACY_TOL if degeneracy_tol is None else degeneracy_tol
    )
    try:
        modulation = build_modulation(
            phonon, q, symprec=symprec, degeneracy_tol=degeneracy_tol
        )
    except ValueError as exc:
        return fallback(evecs_ref, str(exc))

    try:
        ordered, groups = _match_bands(
            evals_ref, modulation.eigenspaces, match_tol
        )
        _check_whole_eigenspaces(groups, bands)
    except _GuardError as exc:
        return fallback(evecs_ref, str(exc))

    psi = evecs_ref.copy()
    spos = phonon.primitive.scaled_positions
    ordered_with_orig, eigvecs_gauge = _prepare_eigenspaces(
        modulation, ordered, spos, q
    )
    touched = [
        j
        for j, group in groups.items()
        if set(group).intersection(bands)
    ]
    # pass 1: svmin guard on every touched eigenspace BEFORE any OPD
    # enumeration — guard failures must fall back without touching the
    # enumeration API (story-010 fallback contract)
    for j in touched:
        cols = groups[j]
        sv = np.linalg.svd(
            evecs_ref[:, cols].conj().T @ eigvecs_gauge[j],
            compute_uv=False,
        )
        if sv.min() < SVMIN_THRESHOLD:
            return fallback(
                evecs_ref,
                f"subspace guard failed for eigenspace {j} "
                f"(svmin={sv.min():.3e} < {SVMIN_THRESHOLD})",
            )
    # pass 2: enumerate ALL touched eigenspaces first (global indices must
    # span every touched block before any selection, so the scalar
    # opd_index existence check sees later blocks too — SPEC-003), then
    # resolve selections and build seeds per eigenspace
    families, coeffs_of = _enumerate_families(
        modulation, ordered_with_orig, eigvecs_gauge, set(touched)
    )
    per_band = []
    for j in touched:
        orig, eigval, eigvecs, irrep = ordered_with_orig[j]
        cols = groups[j]
        dim = eigvecs.shape[0]
        converted = eigvecs_gauge[j]
        family = _resolve_selection(families, j, opd, opd_index)
        coeffs = coeffs_of[family.index]
        images = _distinct_images(coeffs, irrep)
        if len(images) < dim:
            return fallback(
                evecs_ref,
                f"little-group images undergenerate for eigenspace {j} "
                f"({len(images)} < {dim})",
            )
        seeds = np.column_stack(
            [eigvecs_gauge[j] @ c for c in images[:dim]]
        )
        q_mat, r_mat = np.linalg.qr(seeds)
        q_mat = q_mat * np.sign(np.diag(r_mat))
        sv_seeds = np.linalg.svd(
            evecs_ref[:, cols].conj().T @ q_mat, compute_uv=False
        )
        if sv_seeds.min() < SVMIN_THRESHOLD:
            return fallback(
                evecs_ref,
                f"seed subspace guard failed for eigenspace {j} "
                f"(svmin={sv_seeds.min():.3e})",
            )
        psi[:, cols] = q_mat
        freq = float(modulation.eigvals_to_frequencies(eigval))
        chars = _irrep_chars(irrep)
        for band in cols:
            per_band.append(
                SeedBandRecord(
                    band=band,
                    eigenspace_index=j,
                    eigenspace_dim=dim,
                    irrep_chars=chars,
                    family_index=family.index,
                    sg_number=family.sg_number,
                    sg_symbol=family.sg_symbol,
                    direction_summary=family.direction_summary,
                    frequency=freq,
                )
            )
    per_band.sort(key=lambda r: r.band)
    return SymmetrySeedReport(
        qpoint=tuple(q), psi=psi, per_band=per_band
    )
