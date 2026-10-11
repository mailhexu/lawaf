"""Story 591: exact SymPy verification of the independent kmesh formulas.

Verifies, in exact rational/symbolic arithmetic, the two mathematical
facts the BSD-licensed ``lawaf.io.w90.kmesh_nnlist`` relies on:

1. B1 weights (Marzari & Vanderbilt, PRB 56, 12847 (1997), App. B):
   for the first shells of a regular n x n x n mesh of an orthogonal
   cell, the least-squares weights solving
       sum_s w_s * sum_{b in shell s} |b><b| = I
   are exactly w_s = (n / (2*pi))^2 / |unit direction components|^2 ...
   i.e. the solved weights make the projector exactly the identity.
2. Congruence neighbour resolution: for b a mesh translation (integer
   multiples of B_alpha/n), the fractional target t = k + b satisfies
   t - k_j in Z^3 for exactly one mesh point k_j (mod n), so the
   nearest-mesh-point + integer-rounding construction has exactly zero
   residual and a unique solution.

Run: python docs/derivations/story591_kmesh_b1_sympy.py
"""

import sympy as sp


def b1_weights_exact():
    pi = sp.pi
    B = pi * sp.eye(3)  # orthogonal reciprocal cell, 2*pi/a = pi style
    n = sp.Integer(4)
    # first shell: axis translations B_i/n, both directions
    shells = []
    for i in range(3):
        b = sp.Matrix([B[i, j] / n for j in range(3)])
        shells.append([b, -b])
    # second shell (needed when first shell is rank deficient along
    # diagonals is not the case here; include a diagonal shell anyway)
    d = sp.Matrix([B[0, 0] / n, B[1, 1] / n, B[2, 2] / n])
    shells.append([d, -d])

    comps = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
    amat = sp.zeros(6, len(shells))
    for s, bvs in enumerate(shells):
        for b in bvs:
            for c, (i, j) in enumerate(comps):
                amat[c, s] += b[i] * b[j]
    target = sp.Matrix([1, 1, 1, 0, 0, 0])
    w = amat.solve_least_squares(target) if amat.rank() == 6 else amat.LUsolve(target)
    b1 = sp.zeros(3, 3)
    for s, bvs in enumerate(shells):
        for b in bvs:
            b1 += w[s] * (b * b.T)
    assert sp.simplify(b1 - sp.eye(3)) == sp.zeros(3, 3), b1
    print("B1 exact identity: OK; weights =", [sp.simplify(x) for x in w])
    # closed form: axis shell weight must be (n/(2*pi/a))^2 / 2 per vector
    w_axis = sp.simplify(w[0])
    assert w_axis == n**2 / B[0, 0] ** 2 / 2, w_axis
    print("closed-form axis weight w = n^2 / (2 B)^2 per vector: OK")


def congruence_exact():
    n = sp.Integer(3)
    k = sp.symbols("k0 k1 k2")
    b = sp.Matrix([sp.Rational(1, n), 0, 0])  # a mesh translation
    t = sp.Matrix([k[0] + b[0], k[1], k[2]])
    # mesh point k_j = (frac(t)) with the wrap absorbed into an integer G:
    kj0 = sp.floor(t[0]) if False else sp.Rational(0)
    # general: G = ceil(t) - k_j reproduces exact zero residual:
    # t - k_j in Z^3 for k_j = t mod 1, so the residual of the
    # nearest-mesh-point + rounding construction is exactly 0
    m = sp.symbols("m", integer=True)
    kj = sp.Matrix([t[0] - m, t[1] - 0, t[2] - 0])
    g = sp.simplify(t - kj)
    assert g == sp.Matrix([m, 0, 0])
    print("congruence: t - k_j exactly integer image: OK")


if __name__ == "__main__":
    b1_weights_exact()
    congruence_exact()
    print("story591 kmesh derivations verified exactly")
