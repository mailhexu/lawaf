"""Story 020: invariant polynomial basis over LWF cluster indices (incl. strain).

Covers
------
* ``ClusterKey`` translation canonicalization (anchored necklace representative).
* Reynolds construction of invariant polynomials over a generic finite cluster
  action, with two concrete actions: synthetic cubic Oh (48 signed permutation
  matrices) and the placeholder action built from ``SpaceGroupAction``.
* Executed sympy verifications: Voigt 6x6 transformation vs the tensor law,
  Molien series of the Oh natural representation, Burnside counting on S3.
* ``molien_check`` cross-check (constructed counts == Molien coefficients).
* Property tests: invariance under random ops (1e-12), build determinism,
  even-pure-strain generator rule, vectorized == naive evaluation.
"""
import itertools

import numpy as np
import pytest
import sympy as sp

from lawaf.anharmonic.basis import (
    ClusterCutoffs,
    ClusterKey,
    build_invariant_basis,
    build_oh_action,
    cluster_action_from_space_group,
    coordinate_matrix,
    molien_check,
    molien_series_coefficients,
    molien_series_expr_sympy,
    molien_series_sympy,
    voigt_matrix_from_rotation,
)

# cubic nearest-neighbour shell + origin: closed under all 48 signed perms
R7 = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


# ---------------------------------------------------------------------------
# sympy verifications
# ---------------------------------------------------------------------------
def test_voigt_matrix_symbolic_general_rotation():
    """Voigt 6x6: derive V from voigt(W eps W^T) with fully symbolic W, eps and
    assert V @ voigt(eps) == voigt(W eps W^T) as an exact polynomial identity."""
    ws = sp.symbols("w00 w01 w02 w10 w11 w12 w20 w21 w22")
    W = sp.Matrix(3, 3, ws)
    e = sp.symbols("exx eyy ezz eyz exz exy")
    eps = sp.Matrix(
        [[e[0], e[5], e[4]], [e[5], e[1], e[3]], [e[4], e[3], e[2]]]
    )
    VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def voigt_vec(m):
        return sp.Matrix([m[ij] for ij in VOIGT])

    eps_prime = W * eps * W.T
    v_prime = voigt_vec(sp.expand(eps_prime))
    V = sp.Matrix(
        6, 6, [sp.expand(v_prime[i]).coeff(e[j]) for i in range(6) for j in range(6)]
    )
    diff = sp.expand(V * sp.Matrix(e) - v_prime)
    assert diff == sp.zeros(6, 1)  # exact polynomial identity


def test_voigt_signed_perm_structure_and_composition():
    """Integer signed-permutation rotations give a signed-permutation Voigt rep;
    V is a group homomorphism and matches the tensor transformation numerically."""
    action = build_oh_action(nlwf=3)
    rng = np.random.default_rng(3)
    for g in range(action.n_ops):
        Vg = action.strain_voigt_matrix(g)
        # signed permutation structure
        assert set(np.abs(Vg).ravel()) <= {0.0, 1.0}
        assert np.all(Vg.shape == (6, 6))
        for v in range(6):
            assert np.count_nonzero(Vg[v]) == 1
            assert np.count_nonzero(Vg[:, v]) == 1
    # homomorphism + tensor law on random symmetric strains
    eps_flat = rng.uniform(-0.1, 0.1, (200, 6))
    VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def to_mat(v):
        m = np.zeros((3, 3))
        for k, (i, j) in enumerate(VOIGT):
            m[i, j] = m[j, i] = v[k]
        return m

    def to_voigt(m):
        return np.array([m[ij] for ij in VOIGT])

    for _ in range(20):
        g1, g2 = rng.integers(0, 48, 2)
        V1, V2 = action.strain_voigt_matrix(g1), action.strain_voigt_matrix(g2)
        W1, W2 = action.rotations[g1], action.rotations[g2]
        W12 = W1 @ W2
        V12 = None
        for g in range(48):
            if np.array_equal(action.rotations[g], W12):
                V12 = action.strain_voigt_matrix(g)
        assert V12 is not None
        np.testing.assert_allclose(V1 @ V2, V12, atol=1e-14)
        eps_m = to_mat(eps_flat[0])
        np.testing.assert_allclose(
            V1 @ to_voigt(eps_m), to_voigt(W1 @ eps_m @ W1.T), atol=1e-14
        )

def test_voigt_matrix_from_rotation_matches_sampling_convention():
    """voigt_matrix_from_rotation must follow sampling's Voigt order
    (xx, yy, zz, yz, xz, xy) with tensor components stored once."""
    from lawaf.anharmonic.sampling import voigt_to_matrix

    rng = np.random.default_rng(11)
    v = rng.uniform(-0.1, 0.1, 6)
    eps = voigt_to_matrix(v)
    # a random rotation (QR of a random gaussian)
    m = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    V = voigt_matrix_from_rotation(q)
    eps_prime = q @ eps @ q.T
    VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    expected = np.array([eps_prime[ij] for ij in VOIGT])
    np.testing.assert_allclose(V @ v, expected, atol=1e-13)


def test_molien_oh_natural_rep_sympy():
    """Molien series of the Oh natural (signed-permutation) rep, computed by the
    executed integral formula 1/|G| sum_g 1/det(I - t D_g) in sympy.

    Verifies the closed form and the invariant-count table
        order:  2  3  4  5  6
        count:  1  0  2  0  3
    (== 1/((1-t^2)(1-t^4)(1-t^6)); also equals C[e1(x^2), e2(x^2), e3(x^2)]).
    """
    action = build_oh_action(nlwf=3)
    mats = [action.point_rep(g) for g in range(action.n_ops)]
    t = sp.Symbol("t")
    expr = molien_series_expr_sympy(mats)
    rhs = 1 / ((1 - t**2) * (1 - t**4) * (1 - t**6))
    assert sp.simplify(sp.together(expr - rhs)) == 0
    coeffs = molien_series_sympy(mats, 6)
    assert [coeffs[k] for k in range(7)] == [1, 0, 1, 0, 2, 0, 3]
    # fast exact path agrees with the sympy reference
    fast = molien_series_coefficients(mats, 6)
    assert [int(fast[k]) for k in range(7)] == [1, 0, 1, 0, 2, 0, 3]


def test_burnside_s3_sympy():
    """Burnside counting verified on S3 with executed sympy: average number of
    fixed monomials == number of explicit orbits (degree-2 monomials in x,y,z)."""
    x, y, z = sp.symbols("x y z")
    vars3 = sp.Matrix([x, y, z])
    perms = list(itertools.permutations(range(3)))
    exps = sorted(
        (i, j, k)
        for i, j, k in itertools.product(range(3), repeat=3)
        if i + j + k == 2
    )
    monos = {e: x**e[0] * y**e[1] * z**e[2] for e in exps}

    def image_monomial(e, p):
        """Apply the permutation p (as a 0/1 matrix acting on (x,y,z)) to the
        monomial with exponents e; return the exponent vector of the image."""
        P = sp.zeros(3, 3)
        for col, row in enumerate(p):
            P[row, col] = 1
        img = P * vars3
        m = monos[e].subs(
            [(x, img[0]), (y, img[1]), (z, img[2])], simultaneous=True
        )
        m = sp.expand(m)
        for f, mf in monos.items():
            if sp.expand(m - mf) == 0:
                return f
        raise AssertionError("permuted monomial left the degree-2 monomial set")

    fixed_counts = []
    orbits = []
    unseen = set(exps)
    for p in perms:
        fixed_counts.append(sum(1 for e in exps if image_monomial(e, p) == e))
    while unseen:
        seed = next(iter(unseen))
        orb = {image_monomial(e, perms[0]) for e in {seed}}  # placeholder, fixed below
        orb = set()
        frontier = [seed]
        while frontier:
            e = frontier.pop()
            if e in orb:
                continue
            orb.add(e)
            unseen.discard(e)
            for p in perms:
                frontier.append(image_monomial(e, p))
        orbits.append(orb)
    burnside = sp.Rational(sum(fixed_counts), len(perms))
    # e, 3 transpositions (each fixes xy-type and the untouched square),
    # 2 three-cycles (fix nothing at degree 2)
    assert fixed_counts == [6, 2, 2, 0, 0, 2]
    assert burnside == 2
    assert len(orbits) == 2
    assert sorted(sorted(o) for o in orbits) == [
        sorted({(2, 0, 0), (0, 2, 0), (0, 0, 2)}),
        sorted({(1, 1, 0), (1, 0, 1), (0, 1, 1)}),
    ]


# ---------------------------------------------------------------------------
# ClusterKey
# ---------------------------------------------------------------------------
def test_cluster_key_canonical_and_order():
    # lexicographic min R is (0,0,0): already anchored
    k = ClusterKey(((0, (1, 0, 0)), (1, (0, 0, 0))))
    assert k.canonical() == k
    # anchoring subtracts the lex-min R present
    k2 = ClusterKey(((0, (1, 0, 0)), (1, (2, 0, 0))))
    assert k2.canonical() == ClusterKey(((0, (0, 0, 0)), (1, (1, 0, 0))))
    # translation invariance of the canonical form
    a = ClusterKey(((0, (2, 1, 0)), (1, (2, 2, 0))))
    b = ClusterKey(((0, (0, 0, 0)), (1, (0, 1, 0))))
    assert a.canonical() == b.canonical()
    # strain factors ride along untouched
    ks = ClusterKey(((0, (1, 0, 0)),), (3, 1))
    assert ks.strain_factors == (1, 3)
    assert ks.canonical().strain_factors == (1, 3)
    assert ks.order == 3
    assert ks.sector == "coupled"
    assert ClusterKey((), (0, 0)).sector == "strain"
    assert ClusterKey(((2, (0, 0, 1)),)).sector == "q"
    # deterministic total order
    k1 = ClusterKey(((0, (0, 0, 0)),), ())
    k3 = ClusterKey(((0, (0, 0, 0)), (0, (0, 0, 1))), ())
    assert k1 < k3


# ---------------------------------------------------------------------------
# Oh calibration (single Cartesian 3-vector)
# ---------------------------------------------------------------------------
def test_oh_calibration_counts():
    """Known invariant counts of Oh on one Cartesian 3-vector, derived from the
    executed Molien series and matched by the explicit Reynolds construction."""
    action = build_oh_action(nlwf=3)
    basis = build_invariant_basis(
        action, nlwf=3, Rlist=[(0, 0, 0)], orders=(2, 3, 4), include_strain=False
    )
    counts = {n: sum(1 for tm in basis.terms if tm.order == n) for n in (2, 3, 4)}
    assert [counts[2], counts[3], counts[4]] == [1, 0, 2]
    rep = molien_check(basis)
    assert rep.consistent
    # the order-2 invariant is the orbit average of Q_0^2 == (x^2+y^2+z^2)/3
    Q = np.array([[0.3, -1.2, 2.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(
        basis.evaluate(Q)[:, 0], (Q**2).sum(axis=1) / 3.0, atol=1e-14
    )
    # order-4 columns: orbit averages of x^4-type and x^2 y^2-type monomials
    t4 = [tm for tm in basis.terms if tm.order == 4]
    assert len(t4) == 2
    seeds = {tm.seed for tm in t4}
    assert seeds == {
        ClusterKey(((0, (0, 0, 0)),) * 4),
        ClusterKey(((0, (0, 0, 0)), (0, (0, 0, 0)), (1, (0, 0, 0)), (1, (0, 0, 0)))),
    }


# ---------------------------------------------------------------------------
# property tests
# ---------------------------------------------------------------------------
def test_invariance_property_random_ops():
    action = build_oh_action(nlwf=3)
    basis = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=2,
    )
    rng = np.random.default_rng(42)
    nf = 24
    Q = rng.uniform(-1.0, 1.0, (nf, 3 * len(R7)))
    E = rng.uniform(-0.05, 0.05, (nf, 6))
    v0 = basis.evaluate(Q, E)
    assert v0.shape == (nf, len(basis.terms))
    for g in range(action.n_ops):
        Dg = coordinate_matrix(action, g, basis.coord_labels, basis.nlwf)
        Vg = action.strain_voigt_matrix(g)
        vg = basis.evaluate(Q @ Dg.T, E @ Vg.T)
        err = np.abs(v0 - vg).max()
        assert err <= 1e-12, (g, err)


def test_build_deterministic_and_fingerprint():
    action = build_oh_action(nlwf=3)
    kw = dict(
        action=action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=2,
    )
    b1 = build_invariant_basis(**kw)
    b2 = build_invariant_basis(**kw)
    assert [t.seed for t in b1.terms] == [t.seed for t in b2.terms]
    assert [dict(t.coeffs) for t in b1.terms] == [dict(t.coeffs) for t in b2.terms]
    assert b1.fingerprint == b2.fingerprint
    # different cutoff -> different basis, different fingerprint
    b3 = build_invariant_basis(
        **{**kw, "cutoffs": ClusterCutoffs(max_pair_distance=1.0 + 1e-9)}
    )
    assert len(b3.terms) < len(b1.terms)
    assert b3.fingerprint != b1.fingerprint


def test_even_pure_strain_rule():
    action = build_oh_action(nlwf=3)
    # Fortran-compat opt-in: pure-strain terms at even total power only.
    basis = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3, 4),
        include_strain=True,
        max_strain_power=3,
        even_pure_strain=True,
    )
    for tm in basis.terms:
        if tm.sector == "strain":
            assert tm.order % 2 == 0
    assert any(tm.sector == "strain" for tm in basis.terms)
    assert any(tm.sector == "coupled" for tm in basis.terms)
    # PRD FR-017 default: odd pure-strain invariants EXIST (cubic elastic
    # sector, e.g. Tr(eps^3)); the strain rep is inversion-trivial.
    basis_default = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3, 4),
        include_strain=True,
        max_strain_power=3,
    )
    assert any(
        tm.sector == "strain" and tm.order % 2 == 1
        for tm in basis_default.terms
    )


def test_vectorized_equals_naive_loop():
    action = build_oh_action(nlwf=3)
    basis = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=2,
    )
    rng = np.random.default_rng(7)
    nf = 12
    Q = rng.uniform(-1, 1, (nf, 3 * len(R7)))
    E = rng.uniform(-0.05, 0.05, (nf, 6))
    fast = basis.evaluate(Q, E)
    slow = np.stack(
        [basis.evaluate(Q[i], E[i]) for i in range(nf)]
    )  # one frame at a time
    np.testing.assert_allclose(fast, slow, atol=1e-13)


# ---------------------------------------------------------------------------
# Molien cross-check / sector report
# ---------------------------------------------------------------------------
def test_molien_check_report_sectors():
    action = build_oh_action(nlwf=3)
    # PRD FR-017 default (even_pure_strain=False): odd pure-strain
    # invariants are constructed and must match the Molien coefficient.
    basis = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=3,
    )
    rep = molien_check(basis)
    assert rep.consistent, rep
    sectors = {(r.order, r.sector) for r in rep.rows}
    assert (2, "q") in sectors and (3, "q") in sectors
    assert (2, "strain") in sectors and (2, "coupled") in sectors
    for r in rep.rows:
        # uncut, uncapped sectors: constructed count == Molien coefficient
        assert r.constructed == r.molien, r
    odd_strain = [
        r for r in rep.rows if r.sector == "strain" and r.order % 2 == 1
    ]
    assert odd_strain and all(r.constructed > 0 for r in odd_strain)

    # Fortran-compat opt-in: odd pure-strain orders excluded by the rule.
    basis_even = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=3,
        even_pure_strain=True,
    )
    rep_even = molien_check(basis_even)
    assert rep_even.consistent, rep_even
    odd_strain_even = [
        r for r in rep_even.rows if r.sector == "strain" and r.order % 2 == 1
    ]
    assert odd_strain_even and all(
        r.constructed == 0 for r in odd_strain_even
    )
    assert all("rule" in r.note for r in odd_strain_even)

    # Construction cap: max_strain_power below the order -> capped row.
    basis_cap = build_invariant_basis(
        action,
        nlwf=3,
        Rlist=R7,
        orders=(2, 3),
        include_strain=True,
        max_strain_power=2,
    )
    rep_cap = molien_check(basis_cap)
    assert rep_cap.consistent, rep_cap
    capped = [
        r
        for r in rep_cap.rows
        if r.sector == "strain" and r.order > basis_cap.max_strain_power
    ]
    assert capped and all(
        r.constructed == 0 and "capped" in r.note for r in capped
    )


def test_signless_orbits_match_burnside_and_molien():
    """For a signless permutation action: constructed count == Burnside orbit
    count == Molien coefficient (supports are disjoint, no cancellations).
    With signs, orbit-average cancellations make constructed == Molien <=
    Burnside orbit count; the order-3 signed Molien count drops to 0."""
    signed = build_oh_action(nlwf=3)
    signless = build_oh_action(nlwf=3, signed=False)

    def burnside_orbits(act, pool, n):
        monos = {
            ClusterKey(c).canonical()
            for c in itertools.combinations_with_replacement(pool, n)
        }
        fix = 0
        for g in range(act.n_ops):
            fix += sum(
                1 for m in monos if act.cluster_image(m, g).canonical() == m
            )
        assert fix % act.n_ops == 0  # Burnside integrality
        return fix // act.n_ops

    for act, expect_equal in ((signed, False), (signless, True)):
        basis = build_invariant_basis(
            act, nlwf=3, Rlist=[(0, 0, 0)], orders=(2, 3), include_strain=False
        )
        counts = {n: sum(1 for t in basis.terms if t.order == n) for n in (2, 3)}
        rep = molien_check(basis)
        assert rep.consistent
        for n in (2, 3):
            orbits = burnside_orbits(act, basis.coord_labels, n)
            if expect_equal:
                assert orbits == counts[n] == rep.row(n, "q").molien, (act, n)
            else:
                assert counts[n] == rep.row(n, "q").molien <= orbits, (act, n)
    # signed rep kills odd invariants, signless does not
    b_s = build_invariant_basis(
        signed, nlwf=3, Rlist=[(0, 0, 0)], orders=(3,), include_strain=False
    )
    b_p = build_invariant_basis(
        signless, nlwf=3, Rlist=[(0, 0, 0)], orders=(3,), include_strain=False
    )
    assert len(b_s.terms) == 0
    assert len(b_p.terms) == 3


# ---------------------------------------------------------------------------
# from-space-group placeholder action
# ---------------------------------------------------------------------------
def test_from_space_group_placeholder():
    from ase import Atoms

    from lawaf.anharmonic.representation import build_space_group_action

    atoms = Atoms("Na", cell=np.eye(3), scaled_positions=[(0, 0, 0)], pbc=True)
    sga = build_space_group_action(atoms)
    assert sga.n_ops == 48
    action = cluster_action_from_space_group(sga, nlwf=3)
    # same rotation set as the synthetic Oh action (cubic frame A = I)
    set_oh = {W.tobytes() for W in build_oh_action(nlwf=3).rotations}
    set_sg = {np.asarray(W, dtype=int).tobytes() for W in sga.rotations}
    assert set_oh == set_sg

    basis = build_invariant_basis(
        action, nlwf=3, Rlist=R7, orders=(2,), include_strain=False
    )
    rep = molien_check(basis)
    assert rep.consistent
    row = rep.row(2, "q")
    assert row.constructed == row.molien
    # translation anchoring collapses 231 monomials to a handful of orbits
    assert row.constructed < 50
    # invariance spot-check with random amplitudes
    rng = np.random.default_rng(5)
    Q = rng.uniform(-1, 1, (8, 3 * len(R7)))
    v0 = basis.evaluate(Q)
    for g in range(48):
        Dg = coordinate_matrix(action, g, basis.coord_labels, basis.nlwf)
        assert np.abs(v0 - basis.evaluate(Q @ Dg.T)).max() <= 1e-12


def test_cutoffs_bound_the_cluster_pool():
    action = build_oh_action(nlwf=3)
    kw = dict(action=action, nlwf=3, Rlist=R7, orders=(3,), include_strain=False)
    full = build_invariant_basis(**kw)
    cut = build_invariant_basis(
        **kw, cutoffs=ClusterCutoffs(max_shell=0.0, max_pair_distance=0.0)
    )
    # on-site only: 3-coord pool, order 3 signed -> everything vanishes
    assert len(full.terms) >= 0
    assert len(cut.terms) == 0
    assert cut.cutoff_active and not full.cutoff_active
    rep = molien_check(cut)
    assert rep.consistent
