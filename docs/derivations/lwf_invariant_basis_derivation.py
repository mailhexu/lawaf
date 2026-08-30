#!/usr/bin/env python3
"""Executable sympy derivation for the story-020 invariant polynomial basis
(lawaf epic-7, LWF anharmonic effective model).

Every equation below is checked by executed sympy code with a PASS/FAIL
identity.  INPUTS are the discrete definitions (Voigt mapping of
``lawaf.anharmonic.sampling``, the Molien integral formula, the Burnside
lemma); every consequence is DERIVED and checked here, not transcribed.

Sections
--------
1. Strain Voigt representation: the 6x6 matrix V of ``eps -> W eps W^T``,
   derived from the sampling Voigt order (xx, yy, zz, yz, xz, xy) with tensor
   components stored ONCE.  Checked symbolically against the tensor law for a
   general rotation W and general symmetric eps, and numerically for
   composition (V is a group homomorphism).
2. Molien series of the Oh natural representation (48 signed permutation
   matrices): closed form, invariant-count table (orders 0..6), and the
   character-theory cross-check of the order-2 coefficient,
   dim Sym^2 = <(chi^2 - chi(g^2))/2>.
3. Burnside counting, S3 toy case: fixed-monomial counts via symbolic
   substitution; Burnside average == explicit orbit count.
4. Reynolds == Molien spanning lemma (numeric, on the Oh natural rep):
   the number of linearly independent nonzero orbit averages per degree equals
   the Molien coefficient; this is the theoretical guarantee used by
   ``lawaf.anharmonic.basis.molien_check``.
5. Molien of the strain (Voigt) representation of Oh: coefficients orders 0..4.

Run:  python lwf_invariant_basis_derivation.py   (mydev env, sympy >= 1.12)
"""
import sys
from fractions import Fraction
from itertools import permutations, product

import numpy as np
import sympy as sp

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def oh_matrices():
    """The 48 Oh signed permutation matrices, numpy int, deterministic order."""
    mats = []
    for p in permutations(range(3)):
        for s in product((1, -1), repeat=3):
            W = np.zeros((3, 3), dtype=int)
            for j in range(3):
                W[p[j], j] = s[j]
            mats.append(W)
    return mats


# ---------------------------------------------------------------------------
# 1. Strain Voigt representation
# ---------------------------------------------------------------------------
def section1():
    print("-" * 79)
    print("1. Strain Voigt representation (eps -> W eps W^T)")
    ws = sp.symbols("w00 w01 w02 w10 w11 w12 w20 w21 w22")
    W = sp.Matrix(3, 3, ws)
    e = sp.symbols("exx eyy ezz eyz exz exy")
    eps = sp.Matrix([[e[0], e[5], e[4]], [e[5], e[1], e[3]], [e[4], e[3], e[2]]])
    VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def voigt_vec(m):
        return sp.Matrix([m[ij] for ij in VOIGT])

    # derive V column by column: column u = voigt(W E_u W^T)
    V = sp.zeros(6, 6)
    for u, (i, j) in enumerate(VOIGT):
        E = sp.zeros(3, 3)
        E[i, j] = E[j, i] = 1
        col = voigt_vec(sp.expand(W * E * W.T))
        V[:, u] = col
    # 1a. V @ voigt(eps) == voigt(W eps W^T) as an exact polynomial identity
    diff = sp.expand(V * sp.Matrix(e) - voigt_vec(sp.expand(W * eps * W.T)))
    check("1a V @ voigt(eps) == voigt(W eps W^T), general symbolic W, eps",
          diff == sp.zeros(6, 1))

    rng = np.random.default_rng(20)
    def rand_rot():
        m = rng.standard_normal((3, 3))
        q, _ = np.linalg.qr(m)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        return q

    def Vnum(R):
        Vn = np.zeros((6, 6))
        for u, (i, j) in enumerate(VOIGT):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0
            M = R @ E @ R.T
            Vn[:, u] = [M[a, b] for (a, b) in VOIGT]
        return Vn

    ok_c, ok_t, ok_e = True, True, True
    for _ in range(50):
        R1, R2 = rand_rot(), rand_rot()
        V1, V2 = Vnum(R1), Vnum(R2)
        V12 = Vnum(R1 @ R2)
        ok_c &= np.abs(V1 @ V2 - V12).max() < 1e-12
        ev = rng.uniform(-0.1, 0.1, 6)
        em = np.zeros((3, 3))
        for k, (i, j) in enumerate(VOIGT):
            em[i, j] = em[j, i] = ev[k]
        ok_t &= np.abs(V1 @ ev - np.array(
            [(R1 @ em @ R1.T)[ij] for ij in VOIGT])).max() < 1e-12
        ok_e &= abs(np.linalg.det(R1) - 1) < 1e-12
    check("1b V(g) V(h) == V(gh) (homomorphism, 50 random rotation pairs)", ok_c)
    check("1c V(g) @ voigt(eps) == voigt(g eps g^T) numeric (tensor law)", ok_t)
    return V


# ---------------------------------------------------------------------------
# 2. Molien series of the Oh natural representation
# ---------------------------------------------------------------------------
def section2():
    print("-" * 79)
    print("2. Molien series of Oh on the natural 3d rep")
    t = sp.Symbol("t")
    mats = oh_matrices()
    assert len(mats) == 48
    # group closure
    lut = {W.tobytes() for W in mats}
    ok_g = all((W1 @ W2).tobytes() in lut for W1 in mats for W2 in mats)
    check("2.0 the 48 signed permutation matrices form a group (closure)", ok_g)

    total = sp.S.Zero
    for W in mats:
        D = sp.Matrix(W.tolist())
        total += sp.Rational(1, 48) / (sp.eye(3) - t * D).det()
    M = sp.cancel(sp.together(total))
    rhs = 1 / ((1 - t**2) * (1 - t**4) * (1 - t**6))
    check("2a M(t) = 1/((1-t^2)(1-t^4)(1-t^6))  [Hilbert series of "
          "C[e1(x^2), e2(x^2), e3(x^2)], degrees 2, 4, 6]",
          sp.simplify(sp.together(M - rhs)) == 0)

    ser = sp.expand(sp.series(M, t, 0, 7).removeO())
    coeffs = [int(ser.coeff(t, n)) for n in range(7)]
    check("2b invariant-count table [t^0..t^6] == [1, 0, 1, 0, 2, 0, 3]",
          coeffs == [1, 0, 1, 0, 2, 0, 3])
    print("    order :  2  3  4  5  6")
    print("    count :", coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6])
    print("    (order 2: x^2+y^2+z^2 only; order 4: sum x^4 and sum x^2 y^2"
          " types; order 6: +x^6-type)")

    # 2c. character cross-check of the order-2 coefficient:
    #     dim (Sym^2 V)^G = (1/2) <chi(g)^2 + chi(g^2)>
    chi = [int(np.trace(W)) for W in mats]
    chi2 = [int(np.trace(W @ W)) for W in mats]
    dim2 = sp.Rational(sum((c**2 + c2) for c, c2 in zip(chi, chi2)), 2 * 48)
    check("2c character formula dim (Sym^2)^G = <(chi^2 + chi(g^2))/2> == 1",
          dim2 == coeffs[2])
    # order-3: dim (Sym^3)^G = <(chi^3 + 3 chi chi2 + 2 chi3)/6>
    chi3 = [int(np.trace(W @ W @ W)) for W in mats]
    dim3 = sp.Rational(
        sum(c**3 + 3 * c * c2 + 2 * c3 for c, c2, c3 in zip(chi, chi2, chi3)), 6 * 48)
    check("2d character formula dim (Sym^3)^G == 0 (inversion kills odd)", dim3 == 0)


# ---------------------------------------------------------------------------
# 3. Burnside counting, S3 toy case
# ---------------------------------------------------------------------------
def section3():
    print("-" * 79)
    print("3. Burnside counting on S3 (degree-2 monomials in x, y, z)")
    x, y, z = sp.symbols("x y z")
    vars3 = sp.Matrix([x, y, z])
    perms = list(permutations(range(3)))
    exps = sorted((i, j, k) for i in range(3) for j in range(3) for k in range(3)
                  if i + j + k == 2)
    monos = {ev: x**ev[0] * y**ev[1] * z**ev[2] for ev in exps}

    def image_exponent(ev, p):
        P = sp.zeros(3, 3)
        for col, row in enumerate(p):
            P[row, col] = 1
        img = P * vars3
        m = sp.expand(monos[ev].subs([(x, img[0]), (y, img[1]), (z, img[2])],
                                     simultaneous=True))
        for f, mf in monos.items():
            if sp.expand(m - mf) == 0:
                return f
        raise AssertionError("permuted monomial left the monomial set")

    fixed = [sum(1 for ev in exps if image_exponent(ev, p) == ev) for p in perms]
    burnside = sp.Rational(sum(fixed), 6)
    # explicit orbits
    unseen, n_orbits = set(exps), 0
    while unseen:
        n_orbits += 1
        frontier = [unseen.pop()]
        while frontier:
            ev = frontier.pop()
            for p in perms:
                im = image_exponent(ev, p)
                if im in unseen:
                    unseen.discard(im)
                    frontier.append(im)
    check("3a fixed-monomial counts == [6, 2, 2, 0, 0, 2] "
          "(e, 3 transpositions, 2 three-cycles)", fixed == [6, 2, 2, 0, 0, 2])
    check("3b Burnside average <fixed> == 2", burnside == 2)
    check("3c explicit orbit count == Burnside average == 2", n_orbits == 2)


# ---------------------------------------------------------------------------
# 4. Reynolds == Molien spanning lemma (numeric, Oh natural rep)
# ---------------------------------------------------------------------------
def section4():
    print("-" * 79)
    print("4. Reynolds orbit averages span the fixed space; counts == Molien")
    mats = oh_matrices()
    t = sp.Symbol("t")
    coeffs = []
    for n in range(7):
        tot = sp.S.Zero
        for W in mats:
            D = sp.Matrix(W.tolist())
            tot += sp.Rational(1, 48) / (sp.eye(3) - t * D).det()
        ser = sp.expand(sp.series(sp.cancel(sp.together(tot)), t, 0, n + 1).removeO())
        coeffs.append(int(ser.coeff(t, n)))

    # coordinates: the 3 branches; monomials via exponent vectors
    from itertools import combinations_with_replacement

    def reynolds_columns(deg):
        cols = []
        for ev in combinations_with_replacement(range(3), deg):
            v = {}
            for W in mats:
                img = []
                sign = 1
                for b in ev:
                    col = int(np.argmax(np.abs(W[:, b])))
                    s = int(W[col, b])
                    sign *= s
                    img.append(col)
                key = tuple(sorted(img))
                v[key] = v.get(key, Fraction(0)) + Fraction(sign)
            cols.append({k: c / 48 for k, c in v.items() if c != 0})
        return [c for c in cols if c]

    ok = True
    counts = {}
    for deg in range(1, 7):
        cols = reynolds_columns(deg)
        keys = sorted({k for c in cols for k in c})
        Mx = np.zeros((len(keys), len(cols)))
        for j, c in enumerate(cols):
            for i, k in enumerate(keys):
                Mx[i, j] = float(c.get(k, 0))
        rank = int(np.linalg.matrix_rank(Mx, tol=1e-10))
        counts[deg] = (len(cols), rank)
        ok &= rank == coeffs[deg]
    check("4a rank of nonzero Reynolds orbit averages == Molien coefficient "
          "for degrees 1..6", ok)
    print("    degree : nonzero-orbit columns / rank / Molien")
    for deg in range(1, 7):
        nc, rank = counts[deg]
        print(f"      {deg}    : {nc:^19} / {rank} / {coeffs[deg]}")


# ---------------------------------------------------------------------------
# 5. Molien of the strain (Voigt) representation of Oh
# ---------------------------------------------------------------------------
def section5():
    print("-" * 79)
    print("5. Molien series of the Oh strain rep eps -> W eps W^T (6-dim)")
    t = sp.Symbol("t")
    mats = oh_matrices()
    VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def vmat(W):
        V = np.zeros((6, 6))
        for u, (i, j) in enumerate(VOIGT):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0
            M = W @ E @ W.T
            V[:, u] = [M[a, b] for (a, b) in VOIGT]
        return V

    total = sp.S.Zero
    for W in mats:
        D = sp.Matrix(np.round(vmat(W)).astype(int).tolist())
        total += sp.Rational(1, 48) / (sp.eye(6) - t * D).det()
    M = sp.cancel(sp.together(total))
    ser = sp.expand(sp.series(M, t, 0, 5).removeO())
    coeffs = [int(ser.coeff(t, n)) for n in range(5)]
    print("    strain-rep invariant counts [t^0..t^4]:", coeffs)
    check("5a linear invariant unique: count(t^1) == 1 (Tr eps)", coeffs[1] == 1)
    check("5b inversion acts trivially on eps -> odd orders NOT killed "
          f"(count(t^3) = {coeffs[3]} > 0; contrast with the Q sector)",
          coeffs[3] > 0)

    # independent cross-check via characters (traces of powers, no inverses):
    def sym_chi(gm, n):
        p = [0] + [int(np.trace(np.linalg.matrix_power(gm, k))) for k in range(1, n + 1)]
        # complete homogeneous symmetric polynomials via Newton's recursion,
        # exact integer arithmetic: m h_m = sum_{k=1..m} p_k h_{m-k}
        h = [1] + [0] * n
        for m in range(1, n + 1):
            num = sum(p[k] * h[m - k] for k in range(1, m + 1))
            assert num % m == 0, "Newton recursion must divide exactly"
            h[m] = num // m
        return h[n]

    vms = [np.round(vmat(W)).astype(int) for W in mats]
    ok = True
    for n in range(1, 5):
        char_count = sp.Rational(sum(sym_chi(Vm, n) for Vm in vms), 48)
        ok &= char_count == coeffs[n]
    check("5c character-formula counts == Molien coefficients (orders 1..4)", ok)
    print("    NOTE: the story-020 'even pure strain only' generator rule is a")
    print("    MODELING restriction; the epsilon rep itself admits odd invariants")
    print("    such as Tr eps^3 (basis.molien_check flags those rows).")


def main():
    section1()
    section2()
    section3()
    section4()
    section5()
    print("=" * 79)
    failed = [n for n, ok in CHECKS if not ok]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
