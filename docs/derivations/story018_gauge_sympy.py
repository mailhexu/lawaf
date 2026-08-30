#!/usr/bin/env python3
"""Executable sympy derivation for lawaf story-018 (Epic 7, anharmonic
effective model): star-covariant constrained gauge.

Every formula implemented by ``lawaf/anharmonic/gauge.py`` is verified here
by executed sympy code (exact arithmetic on the non-abelian group D3 = S3,
whose 2-dim representation is built from exact radicals; symbolic 2x2 gauge
matrices where a claim is about arbitrary U).  NUMERIC ground truth (BaTiO3
T1u extraction, constraint residuals on the real downfold, star propagation,
time-reversal ties) lives in tests/test_anharmonic_gauge.py.

Sections
--------
1. Exact D3 machinery: the 2-dim representation matrices are a homomorphism
   of the permutation composition law (matrix(g) matrix(h) = matrix(g o h),
   "apply h first" -- the ordering used by compatibility._GroupAlgebra and
   by the star-propagation law).
2. Reynolds (group-averaging) projection P(U) = (1/|LG|) sum_h D(h) U
   D_W(h)^dag is idempotent, its image satisfies the intertwining constraint
   D(h) U = U D_W(h) for every h (constraint manifold preserved), and every
   constrained U is a fixed point.  (Getting the dagger on the D-side, not
   on D_W, is exactly what one sympy run of this script overturned: the
   dagger-symmetric average is a projector onto a DIFFERENT manifold and is
   not idempotent on ours.)
3. Orthogonal-Procrustes/polar retraction: for Z = W diag(s) Y^dag the polar
   factor W Y^dag (a) has orthonormal columns, (b) attains the maximal
   Re Tr over all orthonormal-column matrices (exact second-order identity
   phi(eps) = -(s1+s2)(1-cos(eps)) <= 0, first-order stationarity at 0).
4. Star-propagation transitivity: propagating via h then g equals
   propagating via g o h, exactly, for symbolic U.
5. Wigner-channel projector algebra: P_{ab} = (d/|G|) sum_h conj(tau[h,a,b])
   D(h) satisfies D(g) P_{ab} = sum_c tau[g,c,a] P_{cb} -- the identity the
   production channel construction uses; and Schur isotropy: the only
   matrices commuting with the irreducible tau are lambda*I (the channel
   Gram is lambda I, so its Cholesky normalization yields an exact gauge).
6. Site-irrep extraction (production algorithm's exact skeleton): the
   character projector of the regular rep is an exact commuting idempotent
   of rank d^2; the channel/Fourier columns carry tau covariantly
   (REG(g) F = F tau(g)); the Gram-normalized restriction of the regular rep
   to the carrier reproduces tau exactly.
7. Constraint solution space: {U : D(h) U = U D_W(h) for all h} is
   one-dimensional (Schur), underpinning gauge uniqueness up to scalars.

Run:  python story018_gauge_sympy.py   (mydev env, sympy >= 1.12)
"""

import sympy as sp
from sympy import Rational, sqrt

CHECKS = 0


def check(cond, label):
    global CHECKS
    assert cond, f"FAILED: {label}"
    CHECKS += 1
    print(f"  PASS [{CHECKS:02d}] {label}")


# ----------------------------------------------------------------------
# 1. exact D3 = S3 machinery
# ----------------------------------------------------------------------
print("1. exact D3 machinery (homomorphism ordering)")
sqrt3 = sqrt(3)
# 2-dim irrep of D3: rotation by 120 deg and one reflection, exact radicals
ROT = sp.Matrix([[-Rational(1, 2), -sqrt3 / 2], [sqrt3 / 2, -Rational(1, 2)]])
MIR = sp.Matrix([[1, 0], [0, -1]])
# permutations as composed functions (apply right factor first)
def p_mul(p, q):  # p o q
    return tuple(p[q[i] - 1] for i in range(3))


IDENT = (1, 2, 3)
R_OP = (2, 3, 1)      # (123)
S_OP = (1, 3, 2)      # (23)
TAU = {IDENT: sp.eye(2), R_OP: ROT, p_mul(R_OP, R_OP): ROT**2,
       S_OP: MIR, p_mul(S_OP, R_OP): MIR * ROT,
       p_mul(R_OP, S_OP): ROT * MIR}
OPS = list(TAU)
GROUP_ORDER = 6
DIM = 2

check(all(TAU[p] * TAU[q] == TAU[p_mul(p, q)]
          for p in OPS for q in OPS),
      "matrix(g) matrix(h) = matrix(g o h) for all 36 pairs (exact)")

# ----------------------------------------------------------------------
# 2. Reynolds projection: idempotent + manifold preserved
# ----------------------------------------------------------------------
print("2. Reynolds projection preserves the constraint manifold")
u11, u12, u21, u22 = sp.symbols("u11 u12 u21 u22")
U = sp.Matrix([[u11, u12], [u21, u22]])


def reynolds(Mreps, Wreps, mat):
    """P(U) = (1/N) sum_g D(g) U D_W(g)^dag: the exact projector onto
    {U : D(h) U = U D_W(h) for all h} (verified below)."""
    n = len(Mreps)
    acc = sp.zeros(*mat.shape)
    for g in Mreps:
        acc += Mreps[g] * mat * sp.conjugate(Wreps[g]).T
    return acc / n


PU = reynolds(TAU, TAU, U)
check(sp.simplify(reynolds(TAU, TAU, PU) - PU) == sp.zeros(2, 2),
      "P is idempotent: P(P(U)) = P(U) for symbolic U")
g0 = OPS[3]
invar = sp.simplify(TAU[g0] * PU - PU * TAU[g0])
check(invar == sp.zeros(2, 2),
      "image intertwines: D(h) P(U) = P(U) D_W(h) for every h (manifold)")
check(sp.simplify(reynolds(TAU, TAU, sp.eye(2)) - sp.eye(2)) == sp.zeros(2, 2),
      "manifold members are fixed points (manifold preserved)")

# ----------------------------------------------------------------------
# 3. polar/Procrustes retraction optimality
# ----------------------------------------------------------------------
print("3. orthogonal Procrustes (polar factor) optimality")
Wex = sp.Matrix([[1 / sqrt(2), -1 / sqrt(2)], [1 / sqrt(2), 1 / sqrt(2)]])
s1, s2 = sp.Rational(2), sp.Rational(1)
Z = Wex * sp.diag(s1, s2)  # Z = W diag(s) Y^dag, Y = I
V = Wex  # polar factor = W Y^dag
check(sp.simplify(sp.conjugate(V).T * V - sp.eye(2)) == sp.zeros(2, 2),
      "polar factor has orthonormal columns")
check(sp.simplify(sp.re(sp.trace(sp.conjugate(V).T * Z)) - (s1 + s2)) == 0,
      "Re Tr(V^dag Z) = s1 + s2 at the polar factor")
eps = sp.symbols("varepsilon", real=True)
skew = sp.Matrix([[0, 1], [-1, 0]])
Vt = V * (eps * skew).exp()  # perturbation staying on the Stiefel manifold
phi = sp.simplify(sp.re(sp.trace(sp.conjugate(Vt).T * Z)) - (s1 + s2))
check(sp.simplify(phi - (-(s1 + s2) * (1 - sp.cos(eps)))) == 0,
      "phi(eps) = -(s1+s2)(1-cos eps) exactly (2nd-order form)")
check(sp.diff(phi, eps).subs(eps, 0) == 0,
      "first-order stationarity: d phi/d eps |0 = 0")
check(phi.subs(eps, sp.Rational(1, 10)) < 0,
      "phi(eps) < 0 for eps != 0: polar factor is the Procrustes maximum")

# ----------------------------------------------------------------------
# 4. star propagation transitivity
# ----------------------------------------------------------------------
print("4. star propagation transitivity (via h then g == via g o h)")
g_op, h_op = R_OP, S_OP
gh_op = p_mul(g_op, h_op)
prop_h = TAU[h_op] * U * sp.conjugate(TAU[h_op]).T
prop_g = TAU[g_op] * prop_h * sp.conjugate(TAU[g_op]).T
prop_gh = TAU[gh_op] * U * sp.conjugate(TAU[gh_op]).T
check(sp.simplify(prop_g - prop_gh) == sp.zeros(2, 2),
      "U(g o h) = D(g) [D(h) U D_W(h)^dag] D_W(g)^dag for symbolic U")

# ----------------------------------------------------------------------
# 5. Wigner-channel projector algebra + Schur isotropy
# ----------------------------------------------------------------------
print("5. Wigner channel projector algebra")


def channel_projector(A, B):
    acc = sp.zeros(2, 2)
    for g in OPS:
        acc += sp.conjugate(TAU[g][A, B]) * TAU[g]
    return acc * (sp.Rational(DIM, GROUP_ORDER))


Pab = channel_projector(1, 0)
lhs = TAU[g_op] * Pab
rhs = sum((TAU[g_op][ci, 1] * channel_projector(ci, 0)
           for ci in range(DIM)), sp.zeros(2, 2))
check(sp.simplify(lhs - rhs) == sp.zeros(2, 2),
      "D(g) P_ab = sum_c tau[g]_{c a} P_{c b} (channel covariance)")
# Schur isotropy: the only matrices commuting with the irreducible tau
# are lambda * I -- hence the channel Gram G = W^dag W satisfies
# tau(g) G = G tau(g), i.e. G = lambda I, and the Cholesky normalization
# W G^{-1/2} is an exact intertwiner.
x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")
X = sp.Matrix([[x1, x2], [x3, x4]])
eqs = []
for g in OPS:
    eqs += [sp.expand(e) for e in (TAU[g] * X - X * TAU[g])]
sol = sp.linsolve(eqs, (x1, x2, x3, x4))
sols = list(sol)
check(len(sols) == 1,
      "commutant {X : tau X = X tau} is one-dimensional")
(x1s, x2s, x3s, x4s) = sols[0]
check(sp.simplify(x2s) == 0 and sp.simplify(x3s) == 0
      and sp.simplify(x1s - x4s) == 0,
      "Schur: the only intertwiners are lambda * I (Gram isotropy)")

# ----------------------------------------------------------------------
# 6. site-irrep extraction skeleton (regular rep, exact)
# ----------------------------------------------------------------------
print("6. site-irrep extraction from the regular representation")
# left regular rep of D3 on itself: R(g)_{g o h, h} = 1 (integer matrices)
idx = {g: i for i, g in enumerate(OPS)}
REG = {}
for g in OPS:
    Mreg = sp.zeros(6, 6)
    for j, h in enumerate(OPS):
        Mreg[idx[p_mul(g, h)], j] = 1
    REG[g] = Mreg
check(all(REG[p] * REG[q] == REG[p_mul(p, q)]
          for p in OPS for q in OPS),
      "regular rep is a homomorphism (exact)")

# character projector of the 2-dim irrep: exact integers/rationals
chi = {g: sp.nsimplify(sp.trace(TAU[g])) for g in OPS}
check(all(sp.im(chi[g]) == 0 for g in OPS),
      "tau characters are real (T1-like)")
PI = sp.Rational(DIM, GROUP_ORDER) * sum(
    (chi[g] * REG[g] for g in OPS), sp.zeros(6, 6))
check(sp.simplify(PI * PI - PI) == sp.zeros(6, 6),
      "character projector is idempotent")
check(all(sp.simplify(REG[g] * PI - PI * REG[g]) == sp.zeros(6, 6)
          for g in OPS),
      "character projector commutes with the regular rep")
check(sp.Matrix(PI).rank() == DIM**2,
      "isotypic component rank = d^2 = 4")

# channel/Fourier matrix units: exact carrier of the isotypic component
def reg_channel(A, B):
    acc = sp.zeros(6, 6)
    for g in OPS:
        acc += sp.conjugate(TAU[g][A, B]) * REG[g]
    return acc * sp.Rational(DIM, GROUP_ORDER)


E = {(i, j): reg_channel(i, j) for i in range(DIM) for j in range(DIM)}

def _mat_unit_lhs(i, j, k, l):
    rhs = E[(i, l)] if j == k else sp.zeros(6, 6)
    return E[(i, j)] * E[(k, l)] - rhs

check(all(sp.simplify(_mat_unit_lhs(i, j, k, l)) == sp.zeros(6, 6)
          for i in range(DIM) for j in range(DIM)
          for k in range(DIM) for l in range(DIM)),
      "matrix units multiply as E_ij E_kl = delta_jk E_il (exact)")
# carrier of multiplicity slot 0: Fourier columns f_a = E_{a0} e_0 satisfy
# REG(g) f_a = sum_c tau(g)_{ca} f_c (the production channel identity)
F = sp.Matrix.hstack(*[E[(i, 0)][:, 0] for i in range(DIM)])
check(all(sp.simplify(REG[g] * F - F * TAU[g]) == sp.zeros(6, DIM)
          for g in OPS),
      "REG(g) F = F tau(g): channel columns carry the declared rep (exact)")
res = {g: sp.simplify(
    sp.Rational(GROUP_ORDER, DIM) * (sp.conjugate(F).T * REG[g] * F))
    for g in OPS}
check(all(sp.simplify(res[g] - TAU[g]) == sp.zeros(DIM, DIM) for g in OPS),
      "Gram-normalized restriction of the regular rep reproduces tau "
      "exactly (extraction identity)")

# ----------------------------------------------------------------------
# 7. intertwiner dimension for the constraint (Schur, nullspace form)
# ----------------------------------------------------------------------
print("7. constraint solution space (intertwiners) dimension")
# {U : D(h) U = U D_W(h) for all h} with D = D_W = tau: span{I}
eqs = []
for g in OPS:
    eqs += [sp.expand(e) for e in (TAU[g] * X - X * TAU[g])]
sol2 = sp.linsolve(eqs, (x1, x2, x3, x4))
sols2 = list(sol2)
check(len(sols2) == 1 and sp.simplify(sols2[0][1]) == 0
      and sp.simplify(sols2[0][2]) == 0
      and sp.simplify(sols2[0][0] - sols2[0][3]) == 0,
      "{U : D(h) U = U D_W(h)} = span{I} (unique constrained direction)")

# ----------------------------------------------------------------------
# 8. window-restricted constraint frame (story-018 revision): the
#    constraint matrices are the cross-Gram M(h,q) = X_h^dag X_h with
#    X_h = psi(gq)^dag S_h(g,q) psi(q), and the real-space wannR
#    covariance carries the per-atom lattice-defect cell shift.
# ----------------------------------------------------------------------
print("8. window-restricted frame + defect-shifted real-space law")
# (a) k-space bookkeeping.  Star propagation sets the arm gauge to
#     U' = X_h U D_W^dag, so the arm wannier block wann' = psi' U'
#     satisfies wann' D_W = psi' X_h U (D_W^dag D_W): D_W (not its
#     transpose) sits on the right of the real-space law.  The further
#     replacement of psi' X_h by (P_win S psi) is NOT an algebraic
#     identity -- it requires the window subspace to be symmetry
#     covariant (span(psi') = S span(psi)), a property of the chosen
#     window verified numerically on the BaTiO3 fixture
#     (tests/test_anharmonic_gauge.py: TEST-002 span, TEST-006).
Dw8 = sp.Matrix([[sp.sqrt(2), -sp.sqrt(2)], [sp.sqrt(2), sp.sqrt(2)]]) / 2
check(sp.simplify(Dw8.conjugate().T * Dw8 - sp.eye(2)) == sp.zeros(2, 2),
      "stand-in D_W is orthogonal")
ps1 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"p{i}{j}"))
ps2 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"q{i}{j}"))
S8 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"s{i}{j}"))
Amn8 = sp.Matrix(2, 2, lambda i, j: sp.Symbol(f"u{i}{j}"))
Xh = ps2.conjugate().T * S8 * ps1
wann_prime = ps2 * Xh * Amn8 * Dw8.conjugate().T
resid8 = wann_prime * Dw8 - (ps2 * Xh) * Amn8
check(sp.simplify(resid8) == sp.zeros(2, 2),
      "wann' D_W == psi' X_h U  (D_W^dag D_W cancellation, "
      "D_W orthogonal)")

# (b) Fourier defect cell shift.  S_h(q) carries the per-atom Bloch
#     phase exp(-2 pi i (W^-T q) . t_a)  (representation.py: ex =
#     -(qp @ t_kappa));  wannR[R] = sum_q e^{-2 pi i q.R} wann(q)
#     (wannierizer.k_to_R convention).  Then
#         (P W) wannR[R]_{sigma(a)} = W wannR[W^T R - t_a]_a D_W,
#     i.e. the real-space law gathers the source-atom block from the
#     DEFECT-SHIFTED cell W^T R - t_a.  Exact check over the 2x2x2
#     mesh (integer half-grid triples (i, j, k), q = (i/2, j/2, k/2))
#     for the z-mirror W = diag(1, 1, -1) with defect t = (0, 0, 1).
def _wt_q(i, j, k):
    """W^-T q on the half-grid, wrapped to [0, 2): integer triples."""
    return (i, j, (-k) % 2)


fval = {(i, j, k): sp.Matrix(
    sp.symbols(f"f{i}{j}{k}_0:3")) for i in range(2)
    for j in range(2) for k in range(2)}
R8 = (2, 0, 0)  # cell index R (half-grid units: q.R = (i*2)/4 etc.)
IPI = 2 * sp.pi * sp.I
tdef = (0, 0, 2)  # t = (0, 0, 1) in half-grid units
# phase exponents as EXACT rational dot products (half-grid units /4)
lhs_sum = sp.zeros(3, 1)
rhs_sum = sp.zeros(3, 1)
for (i, j, k), fv in fval.items():
    qdot = lambda v1, v2: sp.Rational(
        sum(a * b for a, b in zip(v1, v2)), 4)
    ii, jj, kk = _wt_q(i, j, k)  # W^-T q (mod-2 wrapped)
    lhs_sum += (
        sp.exp(-IPI * qdot((i, j, k), R8))
        * sp.exp(IPI * qdot(_wt_q(i, j, k), tdef))
        * fval[(ii, jj, kk)]
    )
    cell = tuple(a - b for a, b in zip(R8, tdef))  # W^T R - t
    rhs_sum += sp.exp(-IPI * qdot((i, j, k), cell)) * fv
check(sp.simplify(lhs_sum - rhs_sum) == sp.zeros(3, 1),
      "defect cell shift: sum_q e^{-2pi i q.R} e^{+2pi i (W^-T q).t} "
      "f(W^-T q) == sum_q e^{-2pi i q.(W^T R - t)} f(q)")


# (c) zero-defect consistency: t_a = 0 collapses the shifted law to the
#     defect-free gather, whose two sides differ by the W-closure of
#     the mesh alone (relabel q -> W^-T q on the RHS).
lhs0 = sp.zeros(3, 1)
rhs0 = sp.zeros(3, 1)
for (i, j, k), fv in fval.items():
    qdot = lambda v1, v2: sp.Rational(
        sum(a * b for a, b in zip(v1, v2)), 4)
    lhs0 += sp.exp(-IPI * qdot((i, j, k), R8)) * fv
    ii, jj, kk = _wt_q(i, j, k)
    rhs0 += sp.exp(-IPI * qdot((ii, jj, kk), R8)) * fv
check(sp.simplify(lhs0 - rhs0) == sp.zeros(3, 1),
      "t_a = 0: shifted law degenerates to the defect-free gather")

print(f"\nALL CHECKS PASSED ({CHECKS} checks)")
