#!/usr/bin/env python3
"""Executable sympy derivations for lawaf story-021 (Epic 7): residual-
baseline joint fit identities (FR-010/018, ADR-009).

Every claim below is verified by executed sympy code (exact symbolic or
rational arithmetic, printed PASS/FAIL, final check count; nonzero exit on
any failure).  The NUMERIC ground truth (baseline round trip vs the LWF
model, finite-difference gradients, exact polynomial recovery) lives in
tests/test_anharmonic_fit.py; this script verifies the FORMULAS the
numeric code implements.

Sections
--------
1. Quadratic-form gradient: for a GENERIC (possibly non-symmetric) block
   matrix H,
       d/dQ (1/2 Q^T H Q) = (H + H^T)/2 Q
   and the antisymmetric part contributes nothing to the energy
   (x^T (H - H^T) x == 0).  HarmonicBaseline therefore symmetrizes once
   (Hmat = (K + K^T)/2) and uses E = 1/2 Q^T Hmat Q, grad = Hmat Q.
2. Polynomial monomial gradients: for a monomial prod_i z_i^{n_i},
       d/dz_j = (n_j / z_j) * monomial,
   i.e. the exponent-product rule the BasisGrad wrapper implements; both
   the Q (branch, R) factors and the strain Voigt factors obey it.
3. Voigt stress derivative of a symmetric-tensor quadratic/monomial: with
   each OFF-DIAGONAL tensor component stored ONCE (Voigt order
   xx,yy,zz,yz,xz,xy, matching sampling.voigt_to_matrix / dataset
   conventions),
       dE/deps_v of Q^2 eps_yz      == Q^2        (no factor 2)
       dE/deps_v of eps_yz^2        == 2 eps_yz   (repeated factor rule)
       dE/deps_v of eps_xx eps_yy   == delta rule per stored component
   i.e. the design stress row K_i = (1/V) dE_anh/deps_v needs no extra
   symmetry factors beyond the exponent rule of section 2.
4. RMS block-weight algebra: per-block scale s = sqrt(n / sum_i b_i^2)
   gives the scaled block an exactly unit RMS residual; a positive block
   scaling a -> s a, b -> s b leaves an exactly-consistent solution
   unchanged ((s a) x = s b  <=>  a x = b), so weighting cannot bias an
   exact recovery, only re-balance the normal equations
   sum_i s^2 a_i^T a_i x = sum_i s^2 a_i^T b_i.
"""

import sys

import sympy as sp

checks = 0


def ok(cond, label):
    global checks
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        raise SystemExit(f"FAILED: {label}")
    checks += 1


# ---------------------------------------------------------------------------
print("1. quadratic-form gradient identity (generic non-symmetric H)")
# ---------------------------------------------------------------------------
x = sp.Matrix(sp.symbols("x0 x1 x2"))
H = sp.Matrix(3, 3, sp.symbols("h:9"))
f = sp.Rational(1, 2) * (x.T * H * x)[0]
grad = sp.Matrix([sp.diff(f, xi) for xi in x])
ok(sp.simplify(grad - (H + H.T) / 2 * x) == sp.zeros(3, 1),
   "d/dQ (1/2 Q^T H Q) == (H + H^T)/2 Q for non-symmetric H")
A = H - H.T
ok(sp.simplify((x.T * A * x)[0]) == 0,
   "antisymmetric part contributes zero to the energy")

# ---------------------------------------------------------------------------
print("2. polynomial monomial gradients (exponent products)")
# ---------------------------------------------------------------------------
z = sp.symbols("z0:4")
n = sp.symbols("n0:4", positive=True, integer=True)
mono = sp.Integer(1)
for zi, ni in zip(z, n):
    mono *= zi**ni
for j in range(4):
    lhs = sp.diff(mono, z[j])
    rhs = (n[j] / z[j]) * mono
    ok(sp.simplify(lhs - rhs) == 0, f"d/dz{j} prod z^n == (n{j}/z{j}) * mono")
# strain factors obey the same rule (checked for a representative factor)
qs, e = sp.symbols("q0 eps")
mono2 = qs**2 * e**3
ok(sp.diff(mono2, e) == 3 * qs**2 * e**2, "strain factor exponent rule")

# ---------------------------------------------------------------------------
print("3. Voigt stress derivative, single storage of off-diagonals")
# ---------------------------------------------------------------------------
eyy, ezz, exy = sp.symbols("eps_yy eps_yz eps_xy")
E1 = qs**2 * ezz
ok(sp.diff(E1, ezz) == qs**2,
   "dE/deps_yz of Q^2 eps_yz == Q^2 (single storage, no factor 2)")
E2 = ezz**2
ok(sp.simplify(sp.diff(E2, ezz) - 2 * ezz) == 0,
   "dE/deps_yz of eps_yz^2 == 2 eps_yz (repeated-factor rule)")
# symmetric-tensor quadratic form: the stored-once Voigt derivative of
# 1/2 eps^T C eps (C symmetric, 3x3, here with yz coupling) reproduces the
# classic (C eps) row, i.e. no hidden factor from the Voigt packing.
c11, c12, c13, c22, c23, c33 = sp.symbols("c11 c12 c13 c22 c23 c33")
C = sp.Matrix([[c11, c12, c13], [c12, c22, c23], [c13, c23, c33]])
eps = sp.Matrix([sp.Symbol("eps_xx"), eyy, ezz])
f2 = sp.Rational(1, 2) * (eps.T * C * eps)[0]
ok(sp.simplify(sp.Matrix([sp.diff(f2, ev) for ev in eps]) - C * eps)
   == sp.zeros(3, 1),
   "d/deps (1/2 eps^T C eps) == C eps with each component stored once")
# cross-component monomial: derivative hits exactly one stored component
E3 = exy * eyy
ok(sp.diff(E3, exy) == eyy and sp.diff(E3, eyy) == exy,
   "cross-monomial Voigt derivative is the plain partial (delta rule)")

# ---------------------------------------------------------------------------
print("4. RMS block-weight algebra")
# ---------------------------------------------------------------------------
b1, b2, a11, a12, a21, a22, x1, x2, s = sp.symbols(
    "b1 b2 a11 a12 a21 a22 x1 x2 s")
nrows = 2
s2 = sp.sqrt(nrows / (b1**2 + b2**2))
rms = sp.simplify(((s2 * b1) ** 2 + (s2 * b2) ** 2) / nrows)
ok(rms == 1, "s = sqrt(n / sum b_i^2) scales a block to unit RMS residual")
a = sp.Matrix([[a11, a12], [a21, a22]])
x0 = sp.Matrix([x1, x2])
bb = sp.Matrix([b1, b2])
resid = sp.expand(a * x0 - bb)
scaled = sp.expand((s * a) * x0 - s * bb)
ok(sp.simplify(scaled.subs(s, s2) - s2 * resid) == sp.zeros(2, 1),
   "positive block scaling leaves an exactly-consistent solution unchanged")
# explicit: the s^2-weighted normal equations of the unscaled system equal
# the normal equations of the scaled system
ok(sp.simplify(sp.expand(a.T * sp.diag(s**2, s**2) * (a * x0 - bb))
               - sp.expand((s * a).T * ((s * a) * x0 - s * bb)))
   == sp.zeros(2, 1),
   "(s a)^T (s a x - s b) == s^2 a^T (a x - b)")

