"""Symmetry-invariant polynomial basis over LWF cluster indices (story-020).

The anharmonic LWF effective energy is a polynomial in the mode amplitudes
``Q_{(b, R)}`` (branch ``b`` of the primitive-cell LWF basis in supercell
translation ``R``) and optionally in the homogeneous strain ``eps``.  This
module builds a symmetry-adapted polynomial basis for that energy: orbit
averages (Reynolds projections) of the monomials over a finite group action on
the cluster labels.

Coordinate and label conventions
--------------------------------
Q ordering
    Flat coordinate ``c = iR * nlwf + b`` with ``iR`` the index of ``R`` in the
    SORTED ``Rlist`` (lexicographic) -- the same ``c = icell * nlwf + iwann``
    layout as ``lawaf.anharmonic.sampling``.

Cluster labels
    A factor is ``(branch, R)`` with ``R`` an integer 3-tuple (lattice
    translation index space); a strain factor is a Voigt index ``0..5`` in the
    sampling order ``(xx, yy, zz, yz, xz, xy)`` with tensor components stored
    ONCE (``v[3] = eps_yz``, no factor 2 -- matches
    ``sampling.voigt_to_matrix``).

Group action (generic orbit interface)
    The builder works against an abstract finite group action
    (:class:`ClusterAction`): ``n_ops`` (= ``point_group_order`` here),
    ``image_factor((b, R), g) -> ((b', R'), sign)`` with ``sign = +-1`` the
    matrix element of the signed branch transformation, ``point_rep(g)`` the
    natural 3x3 representation (for the Molien series), and
    ``strain_voigt_matrix(g)`` the 6x6 Voigt image of ``eps -> R eps R^T``.
    Two concrete actions are provided:

    * :func:`build_oh_action` -- synthetic cubic Oh (48 signed permutation
      matrices) acting on Cartesian branch triads and on ``R`` through the
      integer rotation; a pure signed-permutation action.
    * :func:`cluster_action_from_space_group` -- placeholder built from a
      story-014 :class:`~lawaf.anharmonic.representation.SpaceGroupAction`:
      ``(b, R) -> (branch_map(b), W R)``.  The real real-space LWF transporter
      ``T_g`` (story-018/019) will replace the identity branch map and the
      plain ``W R`` lattice part; until then this constructor is EXERCISED
      WITH SYNTHETIC CUBIC DATA (cell A = I, so the integer spglib ``W`` IS
      the Cartesian rotation) and must not be used for production physics.

Translation canonicalization
    Translation action: ``(b, R) -> (b, R + t)``.  :meth:`ClusterKey.canonical`
    returns the unique origin-touching ("anchored") member of a cluster's
    translation orbit: subtract the lexicographically minimal ``R`` present
    (a translate has minimal R == origin iff it subtracts exactly that
    minimum, so the anchored member is unique and translation-invariant).
    The BASIS BUILDER groups monomials by their point-group orbit and seeds
    the Reynolds average with raw in-pool clusters: monomial orbits need not
    contain an anchored member (e.g. the ``Q(R-x) Q(R+x)`` pair), and only
    the raw-orbit partition makes the constructed columns exactly
    Molien-countable and pool-closed.  Translation-summed energy coefficients
    (one per translation class, by translational invariance) are assembled
    from these columns downstream when fitting.

Reynolds construction and counting theory
-----------------------------------------
For a seed monomial ``m`` the invariant candidate is the orbit average
``R(m) = (1/|G|) sum_g s_g m(g Q)`` (``s_g`` the product of branch signs; the
strain part expands through the Voigt matrix elements).  Counting facts:

1. ``R(g m) = R(m)``: the Reynolds image is constant on group orbits, so the
   images of one representative per orbit SPAN the fixed space exactly.
2. For the pure monomial sectors (signed branch maps) the monomial supports
   of distinct orbits are disjoint, so the nonzero per-orbit images are
   linearly independent: their number EQUALS the Molien coefficient of the
   coordinate representation (uncut pool).
3. In the strain/coupled sectors the Voigt matrix elements MIX monomial
   supports, so distinct orbits can yield linearly dependent images.  The
   builder therefore runs an exact rational rank selection (greedy Gaussian
   elimination in enumeration order) over the orbit images and keeps a
   maximal independent subset; its cardinality again equals the Molien
   coefficient (uncut pool).  All of this is verified by
   :func:`molien_check` and in ``tests/test_anharmonic_basis.py``.

Pure-strain parity rule
    By DEFAULT, odd pure-strain invariants are INCLUDED (PRD FR-017 requires
    the cubic elastic sector, e.g. ``Tr eps^3`` for Oh; the
    epsilon-representation ``eps -> R eps R^T`` is trivial on the central
    inversion, so genuine odd invariants exist -- verified by the Molien
    series of the strain rep, see
    ``docs/derivations/lwf_invariant_basis_derivation.py``).  Set
    ``even_pure_strain=True`` to restrict pure-strain terms to even total
    power, matching the Fortran MULTIBINIT ``getEvenAnhaStrain`` convention.
    Affected Molien rows carry a note.

Molien calibration (Oh natural rep, derived in
``docs/derivations/lwf_invariant_basis_derivation.py`` and asserted in tests)::

    M(t) = 1/|G| sum_g 1/det(I - t D_g) = 1/((1-t^2)(1-t^4)(1-t^6))
    order:   0  1  2  3  4  5  6
    count:   1  0  1  0  2  0  3

i.e. order-2 invariants = 1 (``x^2+y^2+z^2``), order-3 = 0, order-4 = 2
(``sum x^4``- and ``sum_{i<j} x^2 y^2``-type), order-6 = 3.
"""

from __future__ import annotations

import abc
import hashlib
import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Factor",
    "ClusterKey",
    "ClusterAction",
    "PermutationClusterAction",
    "build_oh_action",
    "cluster_action_from_space_group",
    "ClusterCutoffs",
    "InvariantTerm",
    "InvariantBasis",
    "build_invariant_basis",
    "MolienRow",
    "MolienReport",
    "molien_check",
    "coordinate_matrix",
    "voigt_matrix_from_rotation",
    "voigt_matrix_exact",
    "molien_series_coefficients",
    "molien_series_expr_sympy",
    "molien_series_sympy",
]

Factor = Tuple[int, Tuple[int, int, int]]
MonoKey = Tuple[Tuple[Factor, ...], Tuple[int, ...]]
VOIGT: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _frac(x) -> Fraction:
    """Exact Fraction for integer-valued floats, else a bounded rational."""
    xf = float(x)
    if xf.is_integer():
        return Fraction(int(xf))
    return Fraction(xf).limit_denominator(10**12)


# ---------------------------------------------------------------------------
# strain Voigt representation
# ---------------------------------------------------------------------------
def voigt_matrix_from_rotation(rot) -> np.ndarray:
    """Float 6x6 Voigt matrix ``V`` of ``eps -> R eps R^T``.

    Defined by ``voigt(R eps R^T) = V @ voigt(eps)`` with the sampling Voigt
    order ``(xx, yy, zz, yz, xz, xy)`` and tensor components stored once.
    Derived symbolically against the tensor law in
    ``docs/derivations/lwf_invariant_basis_derivation.py``.
    """
    R = np.asarray(rot, dtype=float)
    V = np.empty((6, 6))
    for u, (i, j) in enumerate(VOIGT):
        E = np.zeros((3, 3))
        E[i, j] = E[j, i] = 1.0
        M = R @ E @ R.T
        V[:, u] = [M[a, b] for (a, b) in VOIGT]
    return V


def voigt_matrix_exact(rot) -> Tuple[Tuple[Fraction, ...], ...]:
    """Exact rational Voigt matrix of the same transformation."""
    R = [[_frac(x) for x in row] for row in np.asarray(rot, dtype=float)]

    def matmul3(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    V = []
    for u, (i, j) in enumerate(VOIGT):
        E = [[Fraction(0)] * 3 for _ in range(3)]
        E[i][j] = E[j][i] = Fraction(1)
        Rf = [[Fraction(x) for x in row] for row in R]
        M = matmul3(matmul3(Rf, E), [[r[k] for r in Rf] for k in range(3)])  # R E R^T
        V.append(tuple(M[a][b] for (a, b) in VOIGT))
    return tuple(tuple(V[v][u] for v in range(6)) for u in range(6))  # V[v][u]


# ---------------------------------------------------------------------------
# ClusterKey
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterKey:
    """Canonical multiset of cluster factors.

    Attributes
    ----------
    factors:
        Sorted multiset of ``(branch, R)`` factors.
    strain_factors:
        Sorted multiset of Voigt indices (0..5).
    order:
        Total polynomial degree (property: len(factors) + len(strain_factors)).
    """

    factors: Tuple[Factor, ...]
    strain_factors: Tuple[int, ...] = ()

    def __post_init__(self):
        fac = tuple(
            sorted(
                (int(b), (int(r[0]), int(r[1]), int(r[2]))) for b, r in self.factors
            )
        )
        st = tuple(sorted(int(v) for v in self.strain_factors))
        for _, r in fac:
            if len(r) != 3:
                raise ValueError(f"factor R must be a 3-tuple, got {r}")
        for v in st:
            if not 0 <= v <= 5:
                raise ValueError(f"strain Voigt index out of range: {v}")
        object.__setattr__(self, "factors", fac)
        object.__setattr__(self, "strain_factors", st)

    @property
    def order(self) -> int:
        return len(self.factors) + len(self.strain_factors)

    @property
    def sector(self) -> str:
        if not self.factors and not self.strain_factors:
            return "const"
        if not self.strain_factors:
            return "q"
        if not self.factors:
            return "strain"
        return "coupled"

    @property
    def sort_key(self):
        return (self.factors, self.strain_factors)

    def __lt__(self, other: "ClusterKey") -> bool:
        return self.sort_key < other.sort_key

    def translated(self, t) -> "ClusterKey":
        t = (int(t[0]), int(t[1]), int(t[2]))
        return ClusterKey(
            tuple((b, (r[0] - t[0], r[1] - t[1], r[2] - t[2])) for b, r in self.factors),
            self.strain_factors,
        )

    def canonical(self) -> "ClusterKey":
        """Translation-anchored canonical representative.

        Subtracts the lexicographically minimal ``R`` present, i.e. returns the
        unique origin-touching member of the translation orbit (a translate has
        minimal ``R == (0,0,0)`` iff it subtracts exactly that minimum).  The
        result is therefore independent of which translate of the cluster was
        supplied and deterministic under the cluster total order.
        """
        if not self.factors:
            return self
        rmin = min(r for _, r in self.factors)
        # subtracting the lexicographically minimal R present yields THE unique
        # origin-touching member of the translation orbit (any translate whose
        # minimal R is the origin is obtained exactly this way), so this is the
        # deterministic canonical representative under the cluster order.
        return self.translated(rmin)


# ---------------------------------------------------------------------------
# cluster action interface
# ---------------------------------------------------------------------------
class ClusterAction(abc.ABC):
    """Generic finite group action on cluster labels ``(branch, R)`` + strain.

    Subclasses provide a signed-permutation-style action on the factor labels,
    the natural 3x3 point representation (Molien input) and the 6x6 Voigt
    strain representation.  ``n_ops`` is the group order (= point_group_order).
    """

    n_ops: int

    @abc.abstractmethod
    def image_factor(self, factor: Factor, op: int) -> Tuple[Factor, int]:
        """Label image ``(b, R) -> (b', W R)`` and its branch sign ``+-1``."""

    @abc.abstractmethod
    def point_rep(self, op: int) -> np.ndarray:
        """Natural 3x3 representation matrix (integer or float)."""

    @abc.abstractmethod
    def strain_voigt_matrix(self, op: int) -> np.ndarray:
        """Float 6x6 Voigt matrix with ``voigt(R eps R^T) = V @ voigt(eps)``."""

    @abc.abstractmethod
    def strain_row_exact(self, v: int, op: int) -> Tuple[Tuple[int, Fraction], ...]:
        """Nonzero entries ``(u, coeff)`` of row ``v`` of the exact Voigt matrix."""

    @abc.abstractmethod
    def op_inverse(self, op: int) -> int:
        """Index of the group inverse of operation ``op``."""

    @property
    def point_group_order(self) -> int:
        """Group order of the acting point group (== ``n_ops`` here)."""
        return self.n_ops

    def strain_image(self, v: int, op: int) -> Tuple[int, int]:
        """Signed-permutation image of a single strain factor: ``(u, sign)``."""
        row = self.strain_row_exact(v, op)
        if len(row) != 1:
            raise NotImplementedError(
                "strain_image requires a signed-permutation Voigt representation; "
                f"row {v} of op {op} has {len(row)} nonzeros"
            )
        (u, c) = row[0]
        return u, int(c)

    def cluster_image(self, cluster: ClusterKey, op: int) -> ClusterKey:
        """Label image of a whole cluster (signs dropped; not re-anchored)."""
        facs = tuple(self.image_factor(f, op)[0] for f in cluster.factors)
        st = tuple(self.strain_image(v, op)[0] for v in cluster.strain_factors)
        return ClusterKey(facs, st)


class PermutationClusterAction(ClusterAction):
    """Action defined by integer rotations plus per-branch signed permutations.

    Parameters
    ----------
    rotations:
        ``(nsym, 3, 3)`` integer signed-permutation matrices acting on the
        ``R`` index space (must be closed as a group).
    branch_perm:
        ``(nsym, nlwf)`` integers; ``branch_perm[g, b]`` is the image of
        branch ``b`` under operation ``g``.
    branch_sign:
        ``(nsym, nlwf)`` entries ``+-1``; the matrix element
        ``D_g[b', b] = branch_sign[g, b]`` with ``b' = branch_perm[g, b]``.
    cart_rotations:
        Optional float/int ``(nsym, 3, 3)`` rotations used for the strain
        Voigt rep and as the Molien point rep; defaults to ``rotations``
        (valid in a cubic frame ``A = I`` where ``W`` is the Cartesian
        rotation).
    name:
        Free-form label.
    """

    def __init__(
        self,
        rotations,
        branch_perm,
        branch_sign,
        cart_rotations=None,
        name: str = "permutation",
    ):
        rotations = np.asarray(rotations, dtype=int)
        branch_perm = np.asarray(branch_perm, dtype=int)
        branch_sign = np.asarray(branch_sign, dtype=int)
        nsym = rotations.shape[0]
        nlwf = branch_perm.shape[1]
        if branch_perm.shape != (nsym, nlwf) or branch_sign.shape != (nsym, nlwf):
            raise ValueError("branch_perm/branch_sign must be (nsym, nlwf)")
        for g in range(nsym):
            W = rotations[g]
            if not np.array_equal(W @ W.T, np.eye(3, dtype=int)):
                raise ValueError(f"rotation {g} is not an integer orthogonal matrix")
            if sorted(branch_perm[g]) != list(range(nlwf)):
                raise ValueError(f"branch_perm[{g}] is not a permutation")
            if set(branch_sign[g].tolist()) - {-1, 1}:
                raise ValueError(f"branch_sign[{g}] has entries outside +-1")
        # group law: rotations closed; branch action a consistent representation
        lut = {W.tobytes(): g for g, W in enumerate(rotations)}
        for g in range(nsym):
            for h in range(nsym):
                key = (rotations[g] @ rotations[h]).tobytes()
                if key not in lut:
                    raise ValueError("rotations are not closed under composition")
                k = lut[key]
                hp = branch_perm[h]
                if not np.array_equal(branch_perm[k], branch_perm[g][hp]):
                    raise ValueError("branch_perm does not satisfy the group law")
                if not np.array_equal(
                    branch_sign[k], branch_sign[g][hp] * branch_sign[h]
                ):
                    raise ValueError("branch_sign does not satisfy the group law")
        self.rotations = rotations
        self.branch_perm = branch_perm
        self.branch_sign = branch_sign
        self.nlwf = nlwf
        self.name = name
        cart = rotations if cart_rotations is None else np.asarray(cart_rotations)
        self.cart_rotations = cart
        # exact Voigt rep + signed-permutation validation
        self._voigt_exact: List[Tuple[Tuple[Fraction, ...], ...]] = []
        self._voigt_rows: List[List[Tuple[int, Fraction]]] = []
        for g in range(nsym):
            V = voigt_matrix_exact(cart[g])
            self._voigt_exact.append(V)
            rows = []
            for v in range(6):
                nz = tuple((u, V[v][u]) for u in range(6) if V[v][u] != 0)
                if len(nz) != 1:
                    raise ValueError(
                        "Voigt rep is not a signed permutation "
                        f"(op {g}, row {v}); use of this action class requires "
                        "integer signed-permutation Cartesian rotations"
                    )
                rows.append(nz)
            self._voigt_rows.append(rows)
        # inverse-operation lookup: for integer orthogonal rotations the
        # inverse is the transpose; the branch action inverts with the same
        # operation (guaranteed by the group-law validation above)
        lut = {W.tobytes(): g for g, W in enumerate(rotations)}
        self._op_inverse = [lut[rotations[g].T.tobytes()] for g in range(nsym)]

    @property
    def n_ops(self) -> int:
        return self.rotations.shape[0]

    def image_factor(self, factor: Factor, op: int) -> Tuple[Factor, int]:
        b, r = factor
        r2 = tuple(int(x) for x in self.rotations[op] @ np.asarray(r, dtype=int))
        return (int(self.branch_perm[op, b]), r2), int(self.branch_sign[op, b])

    def op_inverse(self, op: int) -> int:
        """Index of the inverse operation (valid for rotations AND branches)."""
        return self._op_inverse[op]

    def point_rep(self, op: int) -> np.ndarray:
        return self.rotations[op]

    def strain_voigt_matrix(self, op: int) -> np.ndarray:
        V = self._voigt_exact[op]
        return np.array([[float(V[v][u]) for u in range(6)] for v in range(6)])

    def strain_row_exact(self, v: int, op: int) -> Tuple[Tuple[int, Fraction], ...]:
        """Nonzero entries ``(u, coeff)`` of row ``v`` of the exact Voigt matrix."""
        return self._voigt_rows[op][v]



def build_oh_action(nlwf: int = 3, signed: bool = True) -> PermutationClusterAction:
    """Synthetic cubic Oh action (48 signed permutation matrices).

    Branches are grouped in Cartesian triads ``(3t + j)``; operation ``g``
    maps triad element ``j`` to signed-permutation image ``pi_g(j)`` with sign
    ``s_g(j)`` (or ``+1`` for ``signed=False``, giving the signless
    permutation action used for Burnside cross-checks).  ``nlwf`` must be a
    multiple of 3.
    """
    if nlwf % 3:
        raise ValueError("nlwf must be a multiple of 3 for the Oh triad action")
    rotations, perms, signs = [], [], []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            W = np.zeros((3, 3), dtype=int)
            bp = np.zeros(nlwf, dtype=int)
            bs = np.ones(nlwf, dtype=int)
            for j in range(3):
                W[p[j], j] = s[j]
            for t in range(nlwf // 3):
                for j in range(3):
                    bp[3 * t + j] = 3 * t + p[j]
                    bs[3 * t + j] = s[j] if signed else 1
            rotations.append(W)
            perms.append(bp)
            signs.append(bs)
    action = PermutationClusterAction(
        rotations, perms, signs, name="Oh-signed" if signed else "Oh-signless"
    )
    return action


def cluster_action_from_space_group(
    sga,
    nlwf: int = 3,
    branch_perm=None,
    branch_sign=None,
    cart_rotations=None,
    name: str = "from-space-group",
) -> PermutationClusterAction:
    """Placeholder cluster action from a story-014 ``SpaceGroupAction``.

    Maps ``(b, R) -> (branch_map_g(b), W_g R)`` with ``W_g = sga.rotations[g]``.
    PLACEHOLDER status (documented, exercised with synthetic cubic data where
    ``A = I`` so ``W_g`` is the Cartesian rotation): the real real-space LWF
    transporter ``T_g`` of story-018/019 will supply the branch transformation
    (via ``branch_perm``/``branch_sign`` or a dense generalization) and the
    defect-corrected lattice part.  ``branch_perm``/``branch_sign`` default to
    the identity permutation / ``+1`` on ``nlwf`` branches.
    """
    rotations = np.asarray(sga.rotations, dtype=int)
    nsym = rotations.shape[0]
    if branch_perm is None:
        branch_perm = np.tile(np.arange(nlwf, dtype=int), (nsym, 1))
    if branch_sign is None:
        branch_sign = np.ones((nsym, nlwf), dtype=int)
    return PermutationClusterAction(
        rotations,
        branch_perm,
        branch_sign,
        cart_rotations=cart_rotations,
        name=name,
    )


# ---------------------------------------------------------------------------
# coordinate representation (for invariance tests and Molien)
# ---------------------------------------------------------------------------
def coordinate_matrix(action, op: int, coord_labels, nlwf: int) -> np.ndarray:
    """Signed-permutation matrix of ``op`` on the flat Q coordinates.

    ``coord_labels`` is the ordered list of ``(branch, R)`` labels (position ==
    coordinate index ``c``).  ``D[c', c] = sign`` with ``c'`` the image of
    ``c``, i.e. the pullback used by the Reynolds action: ``(D Q)[c'] = sign *
    Q[c]``.  Raises if the label pool is not closed under the action.
    """
    labels = list(coord_labels)
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    D = np.zeros((n, n))
    for c, lab in enumerate(labels):
        (b2, r2), s = action.image_factor(lab, op)
        key = (b2, r2)
        if key not in index:
            raise ValueError(
                f"coordinate pool not closed under op {op}: {lab} -> {key}"
            )
        D[index[key], c] = s
    return D


# ---------------------------------------------------------------------------
# cutoffs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClusterCutoffs:
    """LWF-index-space cutoffs applied at enumeration time.

    max_shell:
        Maximum ``|R|`` (Euclidean, lattice units) of any single factor.
    max_pair_distance:
        Maximum pairwise ``|R_i - R_j|`` within the Q part of a cluster.
    ``None`` means unbounded (the supplied ``Rlist`` then bounds the pool).
    """

    max_shell: Optional[float] = None
    max_pair_distance: Optional[float] = None


# ---------------------------------------------------------------------------
# invariant construction
# ---------------------------------------------------------------------------

def _reynolds(seed: ClusterKey, action: ClusterAction) -> Dict[MonoKey, Fraction]:
    """Orbit average of the seed monomial as ``{mono_key: Fraction}``.

    ``mono_key = (sorted q-factors, sorted strain indices)``; coefficients are
    exact rationals: branch signs (+-1) times Voigt matrix elements, summed
    over the group and normalized by ``1/|G|``.

    Pullback convention (consistent with the coordinate matrices of
    :func:`coordinate_matrix`, whose ``(row=image, column=source)`` layout
    means ``(D_g z)_c = s_g(g^-1(c)) z_{g^-1(c)}``): a Q factor is imaged under
    the INVERSE operation ``g^-1``, while a strain factor expands through ROW
    ``v`` of the Voigt matrix ``V_g`` (``eps -> V_g eps`` acts on the vector
    components).  Pairing both parts under the same ``g`` is what makes the
    orbit average invariant.
    """
    acc: Dict[MonoKey, Fraction] = {}
    for g in range(action.n_ops):
        ginv = action.op_inverse(g)
        sign = Fraction(1)
        qfac = []
        for f in seed.factors:
            (b2, r2), s = action.image_factor(f, ginv)
            sign *= s
            qfac.append((b2, r2))
        qkey = tuple(sorted(qfac))
        # expand the strain monomial through the Voigt matrix rows
        sterms = {(): Fraction(1)}
        for v in seed.strain_factors:
            row = action.strain_row_exact(v, g)
            nxt: Dict[Tuple[int, ...], Fraction] = {}
            for skey, c0 in sterms.items():
                for u, cu in row:
                    k2 = tuple(sorted(skey + (u,)))
                    nxt[k2] = nxt.get(k2, Fraction(0)) + c0 * cu
            sterms = nxt
        for skey, cs in sterms.items():
            if cs == 0:
                continue
            key = (qkey, skey)
            acc[key] = acc.get(key, Fraction(0)) + sign * cs
    norm = Fraction(action.n_ops)
    return {k: c / norm for k, c in acc.items() if c != 0}


@dataclass
class InvariantTerm:
    """One basis column: the Reynolds image of the orbit of ``seed``.

    ``coeffs`` maps canonical monomial keys to exact rational coefficients
    (``1/|G|``-normalized orbit averages).
    """

    seed: ClusterKey
    coeffs: Dict[MonoKey, Fraction]
    order: int
    sector: str


class InvariantBasis:
    """Constructed invariant basis with vectorized polynomial evaluation."""

    def __init__(
        self,
        terms: Sequence[InvariantTerm],
        coord_labels: Sequence[Factor],
        nlwf: int,
        action: ClusterAction,
        orders: Sequence[int],
        include_strain: bool,
        max_strain_power: Optional[int],
        cutoff_active: bool,
        rlist: Tuple[Tuple[int, int, int], ...],
        even_pure_strain: bool = False,
    ):
        self.terms: List[InvariantTerm] = list(terms)
        self.coord_labels: Tuple[Factor, ...] = tuple(coord_labels)
        self.nlwf = int(nlwf)
        self.action = action
        self.orders = tuple(int(o) for o in orders)
        self.include_strain = bool(include_strain)
        self.max_strain_power = max_strain_power
        self.cutoff_active = bool(cutoff_active)
        self.rlist = tuple(rlist)
        self.even_pure_strain = bool(even_pure_strain)
        self._coord_index = {lab: i for i, lab in enumerate(self.coord_labels)}
        self._cache_built = False
        self._fingerprint: Optional[str] = None

    # -- fingerprint --------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """sha256 over the SORTED canonical term keys and exact coefficients."""
        if self._fingerprint is None:
            rows = []
            for t in self.terms:
                items = sorted(
                    (qk, sk, str(c)) for (qk, sk), c in t.coeffs.items()
                )
                rows.append(repr((t.seed.sort_key, t.order, t.sector, items)))
            h = hashlib.sha256()
            for row in sorted(rows):
                h.update(row.encode())
                h.update(b"\n")
            self._fingerprint = h.hexdigest()
        return self._fingerprint

    # -- evaluation ---------------------------------------------------------
    def _build_cache(self):
        if self._cache_built:
            return
        qkeys, skeys = [], []
        for t in self.terms:
            for qk, sk in t.coeffs:
                qkeys.append(qk)
                skeys.append(sk)
        self._q_keys = sorted(set(qkeys))
        self._s_keys = sorted(set(skeys))
        self._q_index = {k: i for i, k in enumerate(self._q_keys)}
        self._s_index = {k: i for i, k in enumerate(self._s_keys)}
        self._q_idx_arrays = [
            np.array(
                [self._coord_index[f] for f in qk], dtype=int
            )
            for qk in self._q_keys
        ]
        self._s_idx_arrays = [np.array(sk, dtype=int) for sk in self._s_keys]
        self._cache_built = True

    def evaluate(self, Q, strain=None) -> np.ndarray:
        """Evaluate every term on frames of amplitudes (and strain).

        Parameters
        ----------
        Q:
            ``(nframes, ncoord)`` or ``(ncoord,)`` amplitudes with
            ``c = iR * nlwf + b`` (``iR`` indexes :attr:`coord_labels`).
        strain:
            Optional ``(nframes, 6)`` / ``(6,)`` Voigt strain (sampling order).

        Returns
        -------
        ``(nframes, n_terms)`` (or ``(n_terms,)`` for a single frame).
        Vectorized over frames; the only Python loops run over terms and
        monomials.
        """
        Q = np.asarray(Q, dtype=float)
        single = Q.ndim == 1
        if single:
            Q = Q[None, :]
        if Q.ndim != 2 or Q.shape[1] != len(self.coord_labels):
            raise ValueError(
                f"Q must have {len(self.coord_labels)} amplitudes, got {Q.shape}"
            )
        need_strain = any(t.sector in ("strain", "coupled") for t in self.terms)
        E = None
        if need_strain:
            if strain is None:
                raise ValueError("basis contains strain terms; pass strain")
            E = np.asarray(strain, dtype=float)
            if E.ndim == 1:
                E = E[None, :]
            if E.shape != (Q.shape[0], 6):
                raise ValueError(f"strain must be (nframes, 6), got {E.shape}")
        self._build_cache()
        # monomial products, vectorized over frames
        qprods = np.ones((Q.shape[0], len(self._q_keys)))
        for i, idx in enumerate(self._q_idx_arrays):
            qprods[:, i] = np.prod(Q[:, idx], axis=1) if idx.size else 1.0
        if E is not None:
            sprods = np.ones((E.shape[0], len(self._s_keys)))
            for i, idx in enumerate(self._s_idx_arrays):
                if idx.size:
                    sprods[:, i] = np.prod(E[:, idx], axis=1)
        vals = np.zeros((Q.shape[0], len(self.terms)))
        for ti, term in enumerate(self.terms):
            acc = np.zeros(Q.shape[0])
            for (qk, sk), coeff in term.coeffs.items():
                v = qprods[:, self._q_index[qk]]
                if sk:
                    v = v * sprods[:, self._s_index[sk]]
                acc += float(coeff) * v
            vals[:, ti] = acc
        return vals[0] if single else vals


def build_invariant_basis(
    action: ClusterAction,
    nlwf: int,
    Rlist,
    orders: Sequence[int] = (3, 4),
    cutoffs: Optional[ClusterCutoffs] = None,
    include_strain: bool = True,
    max_strain_power: Optional[int] = None,
    even_pure_strain: bool = False,
) -> InvariantBasis:
    """Enumerate cluster orbits and Reynolds-construct the invariant basis.

    Parameters
    ----------
    action:
        Finite cluster action (:class:`ClusterAction`).
    nlwf:
        Number of LWF branches per primitive cell.
    Rlist:
        Lattice-translation index space (integer 3-tuples); sorted
        lexicographically and required to be closed under all ``action``
        rotations.
    orders:
        Total polynomial degrees to construct.
    cutoffs:
        Optional per-factor shell and pairwise-distance cutoffs.
    include_strain:
        Whether to enumerate strain factors (coupled and pure-strain sectors).
    max_strain_power:
        Cap on the number of strain factors per cluster (default: max order).

    Returns
    -------
    :class:`InvariantBasis` with deterministic term order (identical inputs
    give the identical ordered list).
    """
    cutoffs = cutoffs or ClusterCutoffs()
    orders = tuple(sorted({int(o) for o in orders}))
    if not orders or orders[0] < 1:
        raise ValueError("orders must be a nonempty sequence of positive degrees")
    if nlwf < 1:
        raise ValueError("nlwf must be positive")
    Rlist = tuple(sorted((int(r[0]), int(r[1]), int(r[2])) for r in Rlist))
    if not Rlist:
        raise ValueError("Rlist must be nonempty")
    # closure of Rlist under the action's rotations (needed for both the
    # Reynolds images and the Molien coordinate rep)
    for g in range(action.n_ops):
        for r in Rlist:
            r2 = tuple(int(x) for x in action.point_rep(g) @ np.asarray(r))
            if r2 not in set(Rlist):
                raise ValueError(
                    f"Rlist not closed under rotation {g}: {r} -> {r2}"
                )
    max_strain_power = max(orders) if max_strain_power is None else int(max_strain_power)
    # factor pool (deterministic order: R-major, then branch)
    pool: List[Factor] = []
    for r in Rlist:
        for b in range(nlwf):
            if cutoffs.max_shell is not None and np.linalg.norm(r) > cutoffs.max_shell + 1e-9:
                continue
            pool.append((b, r))

    def pair_ok(qcombo) -> bool:
        if cutoffs.max_pair_distance is None:
            return True
        rs = [r for _, r in qcombo]
        lim = cutoffs.max_pair_distance + 1e-9
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if np.linalg.norm(np.subtract(rs[i], rs[j])) > lim:
                    return False
        return True

    # ---- enumerate monomials and group them into point-group orbits
    # (Reynolds images are constant on an orbit, so one column per orbit spans
    # the fixed space; seeds are RAW in-pool clusters so all images stay in the
    # coordinate pool.  Translation anchoring (ClusterKey.canonical) is the
    # deterministic bookkeeping for translation-equivalent clusters; the
    # physically translation-summed energy coefficients are assembled from
    # these columns downstream.)
    seen_orbits = set()
    orbit_reps: List[ClusterKey] = []
    for n in orders:
        for n_q in range(n, -1, -1):
            n_s = n - n_q
            if n_s > max_strain_power:
                continue
            if not include_strain and n_s:
                continue
            if even_pure_strain and n_q == 0 and n_s % 2 == 1:
                continue  # opt-in Fortran-compat rule: even pure-strain power only
            for qcombo in itertools.combinations_with_replacement(pool, n_q):
                if not pair_ok(qcombo):
                    continue
                for scombo in itertools.combinations_with_replacement(range(6), n_s):
                    raw = ClusterKey(qcombo, scombo)
                    if raw in seen_orbits:
                        continue
                    orb = {action.cluster_image(raw, g) for g in range(action.n_ops)}
                    seen_orbits |= orb
                    orbit_reps.append(raw)

    # ---- Reynolds per orbit, then keep a maximal linearly independent set
    # of the resulting columns.  Exact rational arithmetic; the accepted rows
    # are maintained in REDUCED row echelon form (each pivot monomial appears
    # in exactly one row), so the greedy independence test is exact.  For pure
    # monomial (signed-permutation) sectors the orbit supports are disjoint
    # and nothing is dropped; in the strain/coupled sectors the Voigt matrix
    # elements mix monomial supports, distinct orbits can produce dependent
    # images, and rank selection is required for the Molien count to match.
    terms: List[InvariantTerm] = []
    rref: Dict[MonoKey, Dict[MonoKey, Fraction]] = {}
    for key in orbit_reps:
        coeffs = _reynolds(key, action)
        if not coeffs:
            continue  # orbit average vanishes identically (sign cancellations)
        red = dict(coeffs)
        # eliminate using existing pivots, largest pivot first
        for p in sorted(rref, reverse=True):
            f = red.get(p)
            if f:
                row_p = rref[p]
                for k, c in row_p.items():
                    red[k] = red.get(k, Fraction(0)) - f * c
        red = {k: c for k, c in red.items() if c != 0}
        if not red:
            continue  # dependent column
        pivot = max(red)
        lead = red[pivot]
        red = {k: c / lead for k, c in red.items()}
        # back-substitute: remove the new pivot from all existing rows
        for p in rref:
            f = rref[p].get(pivot)
            if f:
                for k, c in red.items():
                    rref[p][k] = rref[p].get(k, Fraction(0)) - f * c
        rref[pivot] = red
        terms.append(
            InvariantTerm(seed=key, coeffs=dict(coeffs), order=key.order, sector=key.sector)
        )


    return InvariantBasis(
        terms=terms,
        coord_labels=pool,
        nlwf=nlwf,
        action=action,
        orders=orders,
        include_strain=include_strain,
        max_strain_power=max_strain_power,
        even_pure_strain=even_pure_strain,
        cutoff_active=cutoffs.max_shell is not None or cutoffs.max_pair_distance is not None,
        rlist=Rlist,
    )


# ---------------------------------------------------------------------------
# Molien series
# ---------------------------------------------------------------------------
def _poly_mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0:
            continue
        for j, cb in enumerate(b):
            out[i + j] += ca * cb
    return out


def _series_inverse(poly, nmax: int) -> List[Fraction]:
    """Coefficients of ``1 / poly(t)`` truncated at ``nmax`` (``poly[0] == 1``)."""
    c = [Fraction(1)] + [Fraction(0)] * nmax
    for n in range(1, nmax + 1):
        s = Fraction(0)
        for k in range(1, min(n, len(poly) - 1) + 1):
            s += poly[k] * c[n - k]
        c[n] = -s
    return c


def _signed_perm_check(D) -> List[Tuple[int, Fraction]]:
    """Column-wise (row, sign) of a signed permutation matrix."""
    D = np.asarray(D)
    n = D.shape[0]
    cols = []
    for j in range(n):
        rows = np.nonzero(D[:, j])[0]
        if len(rows) != 1:
            raise ValueError("matrix is not a signed permutation matrix")
        r = int(rows[0])
        val = _frac(D[r, j])
        if val not in (Fraction(1), Fraction(-1)):
            raise ValueError("matrix is not a signed permutation matrix")
        cols.append((r, val))
    return cols


def _det_inverse_series_signed_perm(D, nmax: int) -> List[Fraction]:
    """Exact series of ``1/det(I - t D)`` for a signed permutation matrix.

    ``det(I - tD) = prod over cycles of prod_{k in cycle} (1 - s_k t)`` with
    ``s_k`` the entry signs along the cycle; inverted as a truncated power
    series in exact rational arithmetic.
    """
    cols = _signed_perm_check(D)
    n = len(cols)
    poly = [Fraction(1)]
    visited = [False] * n
    for j0 in range(n):
        if visited[j0]:
            continue
        j, signs = j0, []
        while not visited[j]:
            visited[j] = True
            signs.append(cols[j][1])
            j = cols[j][0]
        # cycle of length L with entry signs s_k contributes
        # det(I - t P_cycle) = 1 - (prod_k s_k) t^L  (NOT the per-column
        # product, which coincides only for L == 1)
        sprod = Fraction(1)
        for s in signs:
            sprod *= s
        poly = _poly_mul(
            poly, [Fraction(1)] + [Fraction(0)] * (len(signs) - 1) + [-sprod]
        )
    return _series_inverse(poly, nmax)


def molien_series_coefficients(mats, nmax: int) -> List[Fraction]:
    """Exact Molien coefficients ``[t^0 .. t^nmax]`` of ``1/|G| sum 1/det(I-tD)``."""
    mats = list(mats)
    acc = [Fraction(0)] * (nmax + 1)
    for D in mats:
        ser = _det_inverse_series_signed_perm(D, nmax)
        for k in range(nmax + 1):
            acc[k] += ser[k]
    return [c / len(mats) for c in acc]


def molien_series_expr_sympy(mats):
    """Symbolic Molien series ``1/|G| sum_g 1/det(I - t D_g)`` (sympy)."""
    import sympy as sp

    t = sp.Symbol("t")
    mats = [np.asarray(m, dtype=int) for m in mats]
    n = mats[0].shape[0]
    total = sp.S.Zero
    for D in mats:
        eye = sp.eye(n)
        Dm = sp.Matrix(D.tolist())
        total += sp.Rational(1, len(mats)) / (eye - t * Dm).det()
    return sp.cancel(sp.together(total))


def molien_series_sympy(mats, nmax: int) -> List[int]:
    """Molien coefficients via the executed sympy integral formula (reference)."""
    import sympy as sp

    t = sp.Symbol("t")
    expr = molien_series_expr_sympy(mats)
    ser = sp.series(expr, t, 0, nmax + 1).removeO()
    out = []
    for k in range(nmax + 1):
        c = sp.expand(ser).coeff(t, k)
        c = sp.nsimplify(c)
        assert c.is_integer, f"Molien coefficient at t^{k} is not an integer: {c}"
        out.append(int(c))
    return out


# ---------------------------------------------------------------------------
# Molien cross-check
# ---------------------------------------------------------------------------
@dataclass
class MolienRow:
    """One (order, sector) comparison row."""

    order: int
    sector: str
    molien: int
    constructed: int
    consistent: bool
    note: str = ""


@dataclass
class MolienReport:
    """Per-order, per-sector Molien-vs-constructed comparison."""

    rows: List[MolienRow] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return all(r.consistent for r in self.rows)

    def row(self, order: int, sector: str) -> MolienRow:
        for r in self.rows:
            if r.order == order and r.sector == sector:
                return r
        raise KeyError(f"no row for (order={order}, sector={sector})")

    def table(self) -> str:
        head = f"{'order':>5}  {'sector':<8}  {'molien':>6}  {'constructed':>11}  {'ok':>3}  note"
        lines = [head]
        for r in self.rows:
            lines.append(
                f"{r.order:>5}  {r.sector:<8}  {r.molien:>6}  {r.constructed:>11}"
                f"  {('yes' if r.consistent else 'NO'):>3}  {r.note}"
            )
        return "\n".join(lines)


def molien_check(basis: InvariantBasis, action: ClusterAction = None,
                 nmax: Optional[int] = None) -> MolienReport:
    """Compare Reynolds-constructed counts against Molien coefficients.

    Sectors: pure-Q (Q monomials only), pure-strain, coupled (Q and strain
    factors).  The bivariate Molien coefficient ``[u^a t^b]`` of
    ``1/|G| sum_g 1/det(I - u D^Q_g - t D^eps_g)`` factorizes over the block
    structure, so sector counts are means of products of per-operation
    univariate series.  With an uncut coordinate pool the constructed count
    must EQUAL the coefficient (spanning lemma, see module docstring); with
    active cutoffs ``constructed <= molien`` is required.  Odd pure-strain
    orders are excluded by the even-power generator rule (noted per row).
    """
    action = action or basis.action
    nmax = nmax or max(basis.orders)
    q_mats = [
        coordinate_matrix(action, g, basis.coord_labels, basis.nlwf)
        for g in range(action.n_ops)
    ]
    q_ser = [_det_inverse_series_signed_perm(D, nmax) for D in q_mats]
    e_ser = None
    if basis.include_strain:
        e_mats = [
            [ [ Fraction(0) for _ in range(6) ] for _ in range(6) ]
            for g in range(action.n_ops)
        ]
        for g in range(action.n_ops):
            V = action.strain_voigt_matrix(g)
            for v in range(6):
                for u in range(6):
                    e_mats[g][v][u] = _frac(V[v, u])
        e_ser = [_det_inverse_series_signed_perm(D, nmax) for D in e_mats]

    def mean(vals) -> Fraction:
        vals = list(vals)
        return sum(vals, Fraction(0)) / len(vals)

    constructed: Dict[Tuple[int, str], int] = {}
    for t in basis.terms:
        key = (t.order, t.sector)
        constructed[key] = constructed.get(key, 0) + 1

    rows = []
    for n in basis.orders:
        entries = [("q", mean(qs[n] for qs in q_ser), constructed.get((n, "q"), 0), "")]
        if e_ser is not None:
            entries.append(
                (
                    "strain",
                    mean(es[n] for es in e_ser),
                    constructed.get((n, "strain"), 0),
                    "",
                )
            )
            # aggregate the (a, n-a) coupled splits into one sector row: the
            # constructed count is per (order, sector), and the total Molien
            # coefficient is the sum over the Q/strain power splits
            coupled_mol = sum(
                mean(qs[a] * es[n - a] for qs, es in zip(q_ser, e_ser))
                for a in range(1, n)
            )
            entries.append(
                (
                    "coupled",
                    coupled_mol,
                    constructed.get((n, "coupled"), 0),
                    "",
                )
            )
        for sector, mol, cnt, note in entries:
            mol_int = int(mol)
            if mol.denominator != 1:
                raise AssertionError(f"non-integer Molien coefficient: {mol}")
            odd_rule = (
                basis.even_pure_strain and sector == "strain" and n % 2 == 1
            )
            capped = (
                sector == "strain"
                and basis.max_strain_power is not None
                and n > basis.max_strain_power
            )
            if odd_rule:
                note = "even-pure-strain generator rule: odd orders excluded"
                consistent = cnt == 0
            elif capped:
                note = f"capped by max_strain_power={basis.max_strain_power}"
                consistent = cnt == 0
            else:
                consistent = (cnt == mol_int) or (
                    basis.cutoff_active and cnt <= mol_int
                )
            rows.append(
                MolienRow(
                    order=n,
                    sector=sector,
                    molien=mol_int,
                    constructed=cnt,
                    consistent=consistent,
                    note=note,
                )
            )
    return MolienReport(rows=rows)
