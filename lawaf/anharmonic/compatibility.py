"""Declared-representation compatibility and subspace covariance (story-015).

Two checks over the story-014 space-group action :class:`SpaceGroupAction`
(``lawaf.anharmonic.representation``):

Subspace covariance (FR-002)
    A retained window with projector ``P(q) = V V^dagger`` (``V`` = retained
    eigenvector columns of the window at ``q``) is *covariant* iff

        ``P(g.q) = S_g(q) P(q) S_g(q)^dagger``   for every operation ``g``.

    This holds exactly when the window is a union of complete degenerate
    eigen-blocks at every point of the (star-closed) point set: symmetry
    operations commute with the dynamical matrix, hence preserve its
    eigenspaces, and transport projectors along the star (story-014
    transport oracle).  :func:`check_subspace_covariance` measures the
    per-(q, op) residual ``||P(g.q) - S_g P(q) S_g^dagger||_F / sqrt(rank)``
    and :func:`assert_covariance` raises naming the offending op and q.

Declared representation (FR-003)
    A :class:`RepresentationDeclaration` names a Wyckoff orbit of the
    primitive cell and the site irreps its displacements carry.  At an
    anchor q the little group ``G_q`` acts; inducing the site irrep from the
    site-symmetry subgroup ``H`` (stabilizer of an orbit atom) to ``G_q``
    (Sakuma induction) produces the *expected* little-group content

        ``chi_ind(g) = (1/|H|) sum_{x in G_q} chi_site(x^-1 g x)``

    (``chi_site`` extended by zero off ``H``).  The *found* content is the
    decomposition of the window characters ``chi(g) = tr S_g(q) P(q)``.
    :func:`check_compatibility` compares expected vs found labels per anchor;
    :func:`assert_compatible` raises naming the point and the irreps.

Irreducible characters
    Both decompositions use the character table of the relevant group
    extracted numerically from its LEFT REGULAR representation: class sums
    of the regular rep commute; their joint eigenspaces carry eigenvalues
    ``lambda_{i,c} = |c| chi_i(c) / d_i`` from which

        ``d_i = sqrt( |G| / sum_c |lambda_{i,c}|^2 / |c| )``,
        ``chi_i(c) = lambda_{i,c} d_i / |c|``.

    (These formulas are verified symbolically in
    ``docs/derivations/story015_induced_characters_sympy.py``; the extracted
    O_h table is checked against the published table in
    ``tests/test_anharmonic_compatibility.py``.)

Naming convention (deterministic; Mulliken-exact for the cubic groups met
here): reference characters are derived from the Cartesian rotation part --
trivial, polar ``tr R`` ("T1"), axial ``det R * tr R`` ("T1"), z-component
``R_zz`` ("A2"), xy-block ``R_xx + R_yy`` ("E"), quadratic forms ``R00^2 -
R01^2`` ("B1") and ``R00 R11 + R01 R10`` ("B2") -- each accepted only when
class-constant.  Parity suffixes g/u from the inversion character; the
determinant partner of a named irrep gets the flipped-parity name (this is
Mulliken's parity flip ``X x A1u``).  Remaining irreps are named by
dimension rule (2-dim "E", 3-dim "T2") and a deterministic leftover cycle
("A2", "B1", ...).  For O_h and D4h the result coincides with the published
Mulliken tables (asserted in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DEFAULT_COVARIANCE_MESH",
    "CharacterLabels",
    "CharacterTable",
    "CompatCheck",
    "RepresentationDeclaration",
    "RepresentationLabels",
    "RepresentationReport",
    "assert_compatible",
    "assert_covariance",
    "character_table",
    "check_compatibility",
    "check_subspace_covariance",
    "induced_characters",
    "little_group",
    "resolve_site_irrep",
]
# default mesh for the covariance check: star closure of its irreducible set
DEFAULT_COVARIANCE_MESH = (2, 2, 2)

_CLASS_CONSTANCY_TOL = 1e-8
_MATCH_TOL = 1e-6


# ----------------------------------------------------------------------
# group algebra on spglib operation indices
# ----------------------------------------------------------------------
def _op_key(sga, g: int) -> tuple:
    """Hashable key of operation ``g``: (W bytes, w mod 1 bytes)."""
    return (
        np.ascontiguousarray(sga.rotations[g], dtype=np.int64).tobytes(),
        np.round(np.mod(sga.translations[g], 1.0), 6).tobytes(),
    )


class _GroupAlgebra:
    """Composition/inverse lookups on a closed set of operation indices."""

    def __init__(self, sga, ops):
        self.sga = sga
        self.ops = tuple(int(g) for g in ops)
        self._lookup = {
            _op_key(sga, g): int(g) for g in range(sga.n_ops)
        }

    def compose(self, g: int, h: int) -> int:
        """``(g o h) x = W_g (W_h x + w_h) + w_g`` (apply h first)."""
        sga = self.sga
        W = sga.rotations[g] @ sga.rotations[h]
        w = sga.translations[g] + sga.rotations[g] @ sga.translations[h]
        key = (
            np.ascontiguousarray(W, dtype=np.int64).tobytes(),
            np.round(np.mod(w, 1.0), 6).tobytes(),
        )
        try:
            return self._lookup[key]
        except KeyError as exc:  # pragma: no cover - requires non-closed input
            raise ValueError("operation product not in the stored op set") from exc

    def inverse(self, g: int) -> int:
        sga = self.sga
        Wi = np.round(np.linalg.inv(sga.rotations[g].astype(float))).astype(np.int64)
        w = -(Wi @ sga.translations[g])
        key = (np.ascontiguousarray(Wi).tobytes(), np.round(np.mod(w, 1.0), 6).tobytes())
        try:
            return self._lookup[key]
        except KeyError as exc:  # pragma: no cover
            raise ValueError("operation inverse not in the stored op set") from exc


def little_group(sga, qpoint, tol: float = 1e-8) -> list:
    """Operations ``g`` with ``g.q = q`` (mod 1): the little group of ``q``."""
    q = np.asarray(qpoint, dtype=float).reshape(3)
    out = []
    for g in range(sga.n_ops):
        d = sga.qmap(g, q) - q
        if np.linalg.norm(d - np.rint(d)) < tol:
            out.append(g)
    return out


def _star_closure(sga, qpoints, tol: float = 1e-8) -> list:
    """Union of the stars of ``qpoints`` as wrapped fractional tuples."""
    seen = {}
    for q in np.asarray(qpoints, dtype=float).reshape(-1, 3):
        for _, qi in sga.star(q, tol=tol):
            key = tuple(np.round(qi, 6) % 1.0)
            seen.setdefault(key, key)
    return [np.array(k) for k in seen.values()]


# ----------------------------------------------------------------------
# character table from the regular representation
# ----------------------------------------------------------------------
class CharacterTable:
    """Numerical irreducible characters of a group of space-group operations.

    Attributes
    ----------
    ops:
        The group elements (``SpaceGroupAction`` op indices), closed under
        composition.
    classes:
        Conjugacy classes as tuples of op indices.
    class_sizes:
        ``(n_classes,)`` int array.
    dims:
        Irrep dimensions (descending).
    chars:
        ``(n_irreps, n_classes)`` complex array; ``chars[i, c] = chi_i(c)``.
    names:
        Deterministic Mulliken-style labels (see module docstring).
    """

    def __init__(self, sga, ops, classes, dims, chars, names):
        self.sga = sga
        self.ops = tuple(int(g) for g in ops)
        self.classes = tuple(tuple(int(g) for g in cl) for cl in classes)
        self.class_sizes = np.array([len(cl) for cl in self.classes], dtype=float)
        self.dims = [int(d) for d in dims]
        self.chars = np.asarray(chars, dtype=complex)
        self.names = list(names)
        self._algebra = _GroupAlgebra(sga, self.ops)
        self._cls_of = {g: c for c, cl in enumerate(self.classes) for g in cl}

    @property
    def order(self) -> int:
        return len(self.ops)

    def _to_class_values(self, chi, per_op: bool) -> np.ndarray:
        """Map a character (per op or per class) to class values, checking
        class constancy."""
        if not per_op:
            vals = np.asarray(chi, dtype=complex)
            if len(vals) != len(self.classes):
                raise ValueError(
                    "per-class character must have one value per class"
                )
            return vals
        vals = np.asarray([chi[g] for g in self.ops], dtype=complex)
        out = np.empty(len(self.classes), dtype=complex)
        for c, cl in enumerate(self.classes):
            block = np.asarray([vals[self.ops.index(g)] for g in cl])
            if block.max() - block.min() > 1e-6:
                raise ValueError("character is not a class function")
            out[c] = block[0]
        return out

    def multiplicities(self, chi, per_op: bool = True) -> np.ndarray:
        """``n_i = <chi, chi_i> = (1/|G|) sum_c |c| chi(c) chi_i(c)*``.

        ``chi`` is a per-op array/dict aligned with ``self.ops`` (or a
        per-class array with ``per_op=False``).
        """
        cv = self._to_class_values(chi, per_op)
        return (self.chars.conj() * cv[None, :] * self.class_sizes[None, :]).sum(
            axis=1
        ) / self.order

    def labels(self, mult, tol: float = 1e-4) -> list:
        """Irrep labels expanded by (near-integer) multiplicity."""
        out = []
        for name, m in zip(self.names, np.asarray(mult)):
            mi = int(round(m.real))
            if abs(m.real - mi) > tol:
                raise ValueError(
                    f"non-integer multiplicity {m.real:.6f} for {name}"
                )
            out.extend([name] * mi)
        return out


def character_table(sga, ops) -> CharacterTable:
    """Extract the irreducible character table of the op-index group ``ops``.

    ``ops`` must be closed under composition (use :func:`little_group` or a
    site-symmetry stabilizer).  Method: class sums of the left regular
    representation commute; their joint eigenspaces are the irrep-isotypic
    components of the regular rep; the class-sum eigenvalue on irrep ``i``
    is ``lambda_{i,c} = |c| chi_i(c) / d_i``, and irreducibility fixes
    ``d_i`` via row orthogonality (formulas sympy-verified in
    ``docs/derivations/story015_induced_characters_sympy.py``).
    """
    alg = _GroupAlgebra(sga, ops)
    ops = list(alg.ops)
    n = len(ops)
    index = {g: i for i, g in enumerate(ops)}

    classes = []
    claimed = set()
    for g in ops:
        if g in claimed:
            continue
        cl = sorted({alg.compose(alg.compose(h, g), alg.inverse(h)) for h in ops})
        classes.append(cl)
        claimed |= set(cl)
    n_cls = len(classes)
    sizes = np.array([len(cl) for cl in classes], dtype=float)

    # class sums of the regular representation (permutation matrices)
    sums = []
    for cl in classes:
        M = np.zeros((n, n))
        for g in cl:
            for j, h in enumerate(ops):
                M[index[alg.compose(g, h)], j] += 1.0
        sums.append(M)

    def _split(matrices, basis):
        """Joint eigenspace refinement over commuting normal matrices."""
        sym = basis.T @ matrices[0] @ basis
        w, U = np.linalg.eigh(sym)
        blocks = []
        for val in np.unique(np.round(w, 7)):
            sel = np.abs(w - val) <= 1e-7
            blocks.append(basis @ U[:, sel])
        if len(matrices) == 1:
            return blocks
        out = []
        for blk in blocks:
            out.extend(_split(matrices[1:], blk))
        return out

    comps = _split(sums, np.eye(n))
    dims, chis = [], []
    for b in comps:
        lam = np.array(
            [np.trace(b.T @ M @ b).real / b.shape[1] for M in sums]
        )
        d = np.sqrt(n / np.sum(np.abs(lam) ** 2 / sizes))
        dims.append(int(round(d)))
        chis.append(lam * d / sizes)

    order = sorted(
        range(len(dims)), key=lambda k: (-dims[k], tuple(np.round(chis[k], 6)))
    )
    dims = [dims[i] for i in order]
    chis = [np.round(chis[i], 8) for i in order]
    names = _name_irreps(sga, ops, classes, dims, chis)
    return CharacterTable(sga, ops, classes, dims, chis, names)


# ----------------------------------------------------------------------
# naming (reference characters from the Cartesian rotation part)
# ----------------------------------------------------------------------
class _NotClassConstant(Exception):
    pass


def _name_irreps(sga, ops, classes, dims, chis) -> list:
    """Deterministic Mulliken-style names; see module docstring."""
    inv_g = None
    eye = np.eye(3, dtype=np.int64)
    for g in ops:
        if np.allclose(sga.rotations[g], -eye) and np.allclose(
            np.mod(sga.translations[g], 1.0), 0.0
        ):
            inv_g = g
            break
    cls_of = {}
    for c, cl in enumerate(classes):
        for g in cl:
            cls_of[g] = c

    rot = sga.cart_rotations

    def cvals(f) -> np.ndarray:
        vals = {}
        for g in ops:
            vals[g] = f(g)
        out = np.empty(len(classes), dtype=float)
        for c, cl in enumerate(classes):
            arr = np.array([vals[g] for g in cl])
            if arr.max() - arr.min() > _CLASS_CONSTANCY_TOL:
                raise _NotClassConstant
            out[c] = arr.mean()
        return out

    def flip_suffix(name: str) -> str:
        if name.endswith("g"):
            return name[:-1] + "u"
        if name.endswith("u"):
            return name[:-1] + "g"
        return name

    names = [None] * len(dims)

    def parity(k: int) -> str:
        if inv_g is None:
            return ""
        return "g" if (chis[k][cls_of[inv_g]].real / dims[k]) > 0 else "u"

    def put(k: int, base: str) -> None:
        name = base + parity(k)
        names[k] = name
        if inv_g is None:
            return
        # determinant partner: irrep x A1u  (Mulliken parity flip)
        try:
            partner = cvals(
                lambda g: chis[k][cls_of[g]].real * float(np.linalg.det(rot[g]))
            )
        except _NotClassConstant:
            return
        for j in range(len(dims)):
            if names[j] is None and dims[j] == dims[k] and np.allclose(
                chis[j].real, partner, atol=_MATCH_TOL
            ):
                names[j] = flip_suffix(name)
                return

    references = [
        ("A1", lambda g: 1.0),
        ("T1", lambda g: float(np.trace(rot[g]))),  # polar vector
        ("T1", lambda g: float(np.linalg.det(rot[g]) * np.trace(rot[g]))),  # axial
        ("A2", lambda g: float(rot[g][2, 2])),  # z component (class-constant
        ("E", lambda g: float(rot[g][0, 0] + rot[g][1, 1])),  # xy block
        ("B1", lambda g: float(rot[g][0, 0] ** 2 - rot[g][0, 1] ** 2)),
        ("B2", lambda g: float(rot[g][0, 0] * rot[g][1, 1] + rot[g][1, 0] * rot[g][0, 1])),
    ]
    for base, f in references:
        try:
            ref = cvals(f)
        except _NotClassConstant:
            continue
        for k in range(len(dims)):
            if names[k] is None and np.allclose(chis[k].real, ref, atol=_MATCH_TOL):
                put(k, base)
                break

    for k in range(len(dims)):
        if names[k] is None and dims[k] == 2:
            put(k, "E")
    for k in range(len(dims)):
        if names[k] is None and dims[k] == 3:
            put(k, "T2")

    cycle = iter(["A2", "B1", "B2", "B3", "B4", "B5"])
    while any(nm is None for nm in names):
        k = min((i for i in range(len(dims)) if names[i] is None),
                key=lambda i: tuple(chis[i]))
        try:
            base = next(cycle)
        except StopIteration:  # pragma: no cover
            base = f"i{k}"
        put(k, base)

    if len(set(names)) != len(names):  # pragma: no cover - defensive
        raise ValueError(f"non-unique irrep names generated: {names}")
    return names


# ----------------------------------------------------------------------
# induction (Sakuma) from a site irrep
# ----------------------------------------------------------------------
def resolve_site_irrep(sga, site_ops, name: str):
    """Resolve a declared site-irrep name against the site character table.

    Returns ``(chi_site, name, dim)`` with ``chi_site`` a per-op dict over
    ``site_ops``.  Raises :class:`ValueError` listing the available names.
    """
    ct = character_table(sga, site_ops)
    if name not in ct.names:
        raise ValueError(
            f"unknown site irrep {name!r}; available: {', '.join(ct.names)}"
        )
    j = ct.names.index(name)
    chi = {g: ct.chars[j, ct._cls_of[g]] for g in ct.ops}
    return chi, name, ct.dims[j]


def induced_characters(sga, little_ops, site_ops, chi_site) -> np.ndarray:
    """Sakuma induction ``Ind_H^G chi`` at a (star-fixed) wavevector.

    ``chi_ind(g) = (1/|H|) sum_{x in G_q} chi_site(x^-1 g x)`` with
    ``chi_site`` extended by zero off ``H``.  ``little_ops`` must be closed
    (a little group), ``site_ops`` the stabilizer of one orbit atom within
    it.  Returns the induced character as an array aligned with
    ``little_ops``.  (Formula sympy-verified on a symbolic group and via
    Frobenius reciprocity; see docs/derivations/story015*.py.)
    """
    alg = _GroupAlgebra(sga, little_ops)
    Hs = set(int(g) for g in site_ops)
    chi = {int(g): complex(chi_site[int(g)]) for g in site_ops}
    out = np.empty(len(little_ops), dtype=complex)
    for i, g in enumerate(little_ops):
        acc = 0.0 + 0.0j
        for x in little_ops:
            y = alg.compose(alg.inverse(x), alg.compose(g, x))
            acc += chi.get(y, 0.0 + 0.0j)
        out[i] = acc / len(Hs)
    return out


# ----------------------------------------------------------------------
# FR-002: subspace covariance
# ----------------------------------------------------------------------
def _default_covariance_points(sga) -> list:
    reps = sga.irreducible_qpoints(DEFAULT_COVARIANCE_MESH)
    return _star_closure(sga, reps)


def _validate_projector(P, tol: float = 1e-8) -> None:
    P = np.asarray(P)
    herm = np.max(np.abs(P - P.conj().T))
    if herm > tol:
        raise ValueError(f"projector_fn must return a Hermitian projector (|P-P^dag|max={herm:.2e})")
    resid = np.max(np.abs(P @ P - P))
    if resid > 1e-6:
        raise ValueError(f"projector_fn must return an idempotent projector (|P^2-P|max={resid:.2e})")


def check_subspace_covariance(sga, projector_fn: Callable, qpoints=None, tol: float = 1e-10) -> dict:
    """Verify ``P(g.q) = S_g(q) P(q) S_g(q)^dag`` on a star-closed point set.

    ``projector_fn(q)`` returns the ``(3N, 3N)`` Hermitian idempotent
    projector of the retained window at ``q`` (e.g. ``V V^dag`` over retained
    wrapper eigenvector columns).  ``qpoints=None`` uses the star closure of
    the irreducible :data:`DEFAULT_COVARIANCE_MESH`.

    Returns a dict with keys ``passed``, ``tol``, ``points``, ``residuals``
    (one entry per (q, op): q, op, q_image, rank, residual), ``worst`` and
    ``worst_residual`` (the maximal entry).
    """
    if qpoints is None:
        points = _default_covariance_points(sga)
    else:
        points = [np.asarray(q, dtype=float).reshape(3) for q in qpoints]

    residuals = []
    for q in points:
        P = np.asarray(projector_fn(q), dtype=complex)
        _validate_projector(P)
        rank = int(round(np.trace(P).real))
        norm = np.sqrt(max(rank, 1))
        for g in range(sga.n_ops):
            qimg = sga.qmap(g, q)
            Pimg = np.asarray(projector_fn(qimg), dtype=complex)
            M = sga.matrix(g, q)
            r = float(np.linalg.norm(Pimg - M @ P @ M.conj().T) / norm)
            residuals.append(
                {
                    "q": tuple(np.round(q, 10)),
                    "op": int(g),
                    "q_image": tuple(np.round(qimg, 10)),
                    "rank": rank,
                    "residual": r,
                }
            )
    worst_entry = max(residuals, key=lambda r: r["residual"])
    return {
        "passed": bool(worst_entry["residual"] <= tol),
        "tol": float(tol),
        "points": [tuple(np.round(q, 10)) for q in points],
        "residuals": residuals,
        "worst": float(worst_entry["residual"]),
        "worst_residual": worst_entry,
    }


def _fmt_q(q) -> str:
    q = np.asarray(q, dtype=float).reshape(3)
    if np.linalg.norm(q) < 1e-8:
        return "Gamma"
    return "(" + ", ".join(f"{v:+.4f}" for v in q) + ")"


def assert_covariance(result: dict, tol: float | None = None) -> None:
    """Raise :class:`ValueError` naming the offending op and q if the
    covariance check failed."""
    tol = result["tol"] if tol is None else tol
    bad = [r for r in result["residuals"] if r["residual"] > tol]
    if not bad:
        return
    worst = max(bad, key=lambda r: r["residual"])
    raise ValueError(
        f"subspace covariance FAILED: {len(bad)} violating (q, op) pairs "
        f"(tol={tol:g}); worst: op {worst['op']} at q={_fmt_q(worst['q'])} "
        f"(image {_fmt_q(worst['q_image'])}): residual {worst['residual']:.3e}"
    )


# ----------------------------------------------------------------------
# FR-003: declared-representation compatibility
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RepresentationDeclaration:
    """Declared representation carried by a Wyckoff orbit.

    Attributes
    ----------
    wyckoff:
        Wyckoff letter of the orbit in the primitive cell, optionally with
        its multiplicity prefix (``"1b"`` or ``"b"``).
    site_irreps:
        Site-irrep names carried by the orbit's displacement sector, with
        repetition denoting multiplicity (e.g. ``["T1u", "T1u"]``).
    strain_sector:
        Whether the declared sector couples to strain (carried for FR-003
        bookkeeping; not used by the character check).
    anchors:
        Optional anchor wavevectors (fractional, wrapped).  ``None`` means
        Gamma only; anchors must be points where the site irreps are defined
        (high-symmetry/commensurate points).
    window_irreps:
        Optional per-anchor retained-window irrep contents.  This is distinct
        from ``site_irreps``: it declares the little-group decomposition of
        an explicitly pinned zone-boundary window.
    """

    wyckoff: str
    site_irreps: list
    strain_sector: bool = True
    anchors: tuple | None = None
    window_irreps: dict[tuple, list[str]] | None = None


@runtime_checkable
class RepresentationLabels(Protocol):
    """Provider of found irrep labels of a retained window at an anchor q."""

    def irreps_at(self, qpoint) -> list: ...


class CharacterLabels:
    """Built-in :class:`RepresentationLabels` from window characters.

    ``irreps_at(q)`` builds ``P(q)`` via ``projector_fn``, forms the
    little-group characters ``chi(g) = tr S_g(q) P(q)`` and decomposes them
    against the numerically extracted little-group character table
    (``label_source='characters'``).
    """

    def __init__(self, sga, projector_fn: Callable):
        self.sga = sga
        self.projector_fn = projector_fn

    def irreps_at(self, qpoint) -> list:
        q = np.asarray(qpoint, dtype=float).reshape(3)
        ops = little_group(self.sga, q)
        ct = character_table(self.sga, ops)
        P = np.asarray(self.projector_fn(q), dtype=complex)
        _validate_projector(P)
        chi = [complex(np.trace(self.sga.matrix(g, q) @ P)) for g in ops]
        return ct.labels(ct.multiplicities(chi, per_op=True))


@dataclass(frozen=True)
class CompatCheck:
    """Per-anchor compatibility result."""

    qpoint: np.ndarray
    expected: list
    found: list
    passed: bool


@dataclass(frozen=True)
class RepresentationReport:
    """Result of :func:`check_compatibility` over all anchors."""

    declaration: RepresentationDeclaration
    checks: tuple
    label_source: str

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def check_compatibility(
    sga,
    declaration: RepresentationDeclaration,
    projector_fn_or_labels,
    qpoints=None,
) -> RepresentationReport:
    """Compare declared (induced) vs found irreps at each anchor point.

    ``projector_fn_or_labels`` is either a :class:`RepresentationLabels`
    provider (used verbatim, ``label_source='provided'``) or a projector
    function, which is wrapped in :class:`CharacterLabels`
    (``label_source='characters'``).  ``qpoints`` overrides
    ``declaration.anchors``; both default to Gamma.  Mismatch is reported
    (not raised); use :func:`assert_compatible`.
    """
    if callable(projector_fn_or_labels) and not isinstance(
        projector_fn_or_labels, CharacterLabels
    ):
        # a raw projector function: wrap it in the character provider
        labels = CharacterLabels(sga, projector_fn_or_labels)
        label_source = "characters"
    else:
        labels = projector_fn_or_labels
        label_source = (
            "characters" if isinstance(labels, CharacterLabels) else "provided"
        )

    if qpoints is None:
        anchors = (
            [np.zeros(3)]
            if declaration.anchors is None
            else [np.asarray(a, dtype=float).reshape(3) for a in declaration.anchors]
        )
    else:
        anchors = [np.asarray(a, dtype=float).reshape(3) for a in qpoints]

    letter = declaration.wyckoff.lstrip("0123456789")
    if not letter:
        raise ValueError(f"declaration wyckoff {declaration.wyckoff!r} has no letter")
    wyckoffs = tuple(sga.symmetry_dataset.wyckoffs)
    orbit = [k for k, w in enumerate(wyckoffs) if w == letter]
    if not orbit:
        raise ValueError(
            f"declaration wyckoff {declaration.wyckoff!r}: no atom with wyckoff "
            f"letter {letter!r} (present: {sorted(set(wyckoffs))})"
        )
    kappa0 = orbit[0]

    checks = []
    for q in anchors:
        lg = little_group(sga, q)
        ct_lg = character_table(sga, lg)
        site_ops = [g for g in lg if sga.atom_maps[g][kappa0] == kappa0]
        chi_ind = np.zeros(len(lg), dtype=complex)
        for name in declaration.site_irreps:
            chi_site, _, _ = resolve_site_irrep(sga, site_ops, name)
            chi_ind = chi_ind + induced_characters(sga, lg, site_ops, chi_site)
        expected = ct_lg.labels(ct_lg.multiplicities(chi_ind, per_op=True))
        found = list(labels.irreps_at(q))
        passed = sorted(expected) == sorted(found)
        checks.append(
            CompatCheck(qpoint=q, expected=expected, found=found, passed=passed)
        )
    return RepresentationReport(
        declaration=declaration, checks=tuple(checks), label_source=label_source
    )


def assert_compatible(report: RepresentationReport) -> None:
    """Raise :class:`ValueError` naming the point and expected-vs-found
    irreps for every failing anchor."""
    bad = [c for c in report.checks if not c.passed]
    if not bad:
        return
    parts = [
        (
            f"at q={_fmt_q(c.qpoint)}: expected {c.expected}, found {c.found} "
            f"(declaration wyckoff={report.declaration.wyckoff!r}, "
            f"site_irreps={list(report.declaration.site_irreps)})"
        )
        for c in bad
    ]
    raise ValueError(
        "representation compatibility FAILED: " + "; ".join(parts)
    )
