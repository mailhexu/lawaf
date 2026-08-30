#!/usr/bin/env python3
"""Executable sympy derivation of the Marzari-Vanderbilt (MV) MLWF equations
for lawaf story-009 / epic-5.

Every equation below is checked by executed sympy code with a PASS/FAIL
identity (simplify/expand difference == 0). The INPUTS are the MV97 discrete
definitions (see section 1); every consequence -- the Omega decomposition,
gauge structure, gradient, and analytic oracles -- is DERIVED and checked
here, not transcribed.

Ground truth cross-checked: wannier90 3.1.0 ``src/wannierise.F90``
(``wann_omega`` lines 1713-1984, ``wann_domega`` lines 1987-2210).
Run:  python mlwf_domega_derivation.py   (mydev env, sympy >= 1.12)

Conventions (lawaf / w90):
  M~^{k,b}_{mn} = <u~_mk | u~_n,k+b>,  M~ = U_k^dagger M U_{k+b}
  Gauge action used throughout:  U_k -> U_k e^{i eps A_k}  (A_k Hermitian,
  i eps A_k anti-Hermitian), so  M~^{k,b} -> e^{-i eps A_k} M~^{k,b}
  e^{+i eps A_{k+b}}.
  w_b: neighbor weights with sum_j w_j b_{j,alpha} b_{j,beta} = delta_{alpha,beta}
  MV97 DISCRETE DEFINITIONS (inputs, section 1):
    <r>_n     = -(1/Nk) sum_{k,j} w_j b_j theta~^{k,j}_n
    <r^2>_n   =  (1/Nk) sum_{k,j} w_j (1 - |M~_nn|^2 + theta~^2)
    theta~    = Im ln M~_nn   (sheet/csheet branch, w90 lines 1749/2038)
  DERIVED (sections 1-5): Omega_n = <r^2>_n - |<r>_n|^2 decomposes as
    Omega_I  = (1/Nk) sum w_j (N - Tr M~^dag M~)        [gauge invariant]
    Omega_D  = (1/Nk) sum w_j (theta~ + b.rbar_n)^2     [phase gauge]
    Omega_OD = (1/Nk) sum w_j sum_{m!=n} |M~_{mn}|^2     [unitary gauge]
  (== w90 om_i + om_d + om_od; MV97's Omega_OD == w90's om_d + om_od)
"""
import sys

import sympy as sp

I = sp.I
CHECKS = []


def check(name, expr):
    ok = sp.simplify(expr) == 0
    CHECKS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def check_true(name, condition, detail=""):
    CHECKS.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    return condition


def header(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


# ---------------------------------------------------------------------------
header("1. From Omega_n = <r^2>_n - |<r>_n|^2 (MV97 discrete definitions)"
       " to Omega_I + Omega_D + Omega_OD")
# ---------------------------------------------------------------------------
# INPUTS (definitions, not derived): the discrete moments above. Everything
# below is derived from them by executed algebra.
# 1a. per-band split: Omega_n = sum_j w(1-|M_nn|^2) + sum_j w(theta+b.rbar)^2
#     [completion of square; requires sum_j w_j b_ja b_jb = delta_ab]
tp, tm, bq = sp.symbols("theta_p theta_m b", real=True)
w1 = 1 / (2 * bq**2)
rbar1 = -(w1 * bq * tp + w1 * (-bq) * tm)          # Nk = 1 (per-k form)
# The 1-|M_nn|^2 entries cancel identically on both sides (made explicit):
mp_, mm_ = sp.symbols("m_p m_m", real=True)
check("1a'(1D) same with explicit 1-|M_nn|^2 entries",
      sp.expand((w1 * mp_ + w1 * mm_ + w1 * tp**2 + w1 * tm**2 - rbar1**2)
                - (w1 * mp_ + w1 * mm_
                   + w1 * (tp + bq * rbar1) ** 2 + w1 * (tm - bq * rbar1) ** 2)))
# 3D version: +-x, +-y, +-z axes, w=1/2 each (constraint satisfied)
r = sp.symbols("r0:6", real=True)
axes = [sp.Matrix(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1))]
w3 = [sp.Rational(1, 2)] * 6
rbar3 = -sum((w3[j] * axes[j] * r[j] for j in range(6)), sp.zeros(3, 1))
lhs3 = sum(w3[j] * r[j] ** 2 for j in range(6)) - (rbar3.T * rbar3)[0, 0]
rhs3 = sum(w3[j] * (r[j] + axes[j].dot(rbar3)) ** 2 for j in range(6))
check("1a(3D) completion of square on the 6-neighbor star", sp.expand(lhs3 - rhs3))

# 1b. band sum: sum_n (1-|M_nn|^2) = (N - Tr M^dag M) + sum_{m!=n}|M_mn|^2
#     => sum_n Omega_n = Omega_I + Omega_OD + Omega_D
xr, xi, yr, yi, ur, ui, vr, vi = sp.symbols(
    "x_r x_i y_r y_i u_r u_i v_r v_i", real=True)
M = sp.Matrix([[xr + I * xi, yr + I * yi], [ur + I * ui, vr + I * vi]])
Mc = M.conjugate()
diag = sum(M[n, n] * Mc[n, n] for n in range(2))
offd = sum(M[m, n] * Mc[m, n] for m in range(2) for n in range(2) if m != n)
check("1b N-sum_n|nn|^2 = (N-Tr M^dag M)+sum_{m!=n}|mn|^2",
      sp.expand((2 - diag) - ((2 - (diag + offd)) + offd)))

# 1c. per-entry quadratic identity (polar form; sheet gauge e^{-i arg z} z real)
rho, phi = sp.symbols("rho phi", positive=True, real=True)
z = rho * sp.exp(I * phi)
w_quad = 1 - sp.exp(-I * phi) * z
check("1c |1 - e^{-i arg z} z|^2 = (1-|z|)^2  (w90 r2ave entry algebra)",
      sp.expand(w_quad * sp.conjugate(w_quad) - (1 - rho) ** 2))

# ---------------------------------------------------------------------------
header("2. Gauge dependence of Omega_I / Omega_OD")
# ---------------------------------------------------------------------------
# 2a. Omega_I invariant under INDEPENDENT per-k unitaries:
#     M~ = U_k^dag M U_{k+b} with two independent U(2) matrices.


def csym(name):
    x = sp.Symbol(name + "_r", real=True)
    y = sp.Symbol(name + "_i", real=True)
    return x + I * y


a_, b_, c_, d_ = (csym(s) for s in ("a", "b", "c", "d"))
M2 = sp.Matrix([[a_, b_], [c_, d_]])


def U2(al_, t_, p_, s_):
    return sp.exp(I * al_) * sp.Matrix([
        [sp.cos(t_ / 2) * sp.exp(I * s_), sp.sin(t_ / 2) * sp.exp(I * p_)],
        [-sp.sin(t_ / 2) * sp.exp(-I * p_), sp.cos(t_ / 2) * sp.exp(-I * s_)]])


Uk = U2(*sp.symbols("ak tk pk sk", real=True))
Ukb = U2(*sp.symbols("ab tb pb sb", real=True))
Mtil2 = Uk.H * M2 * Ukb
lhs_2a = sp.simplify(sp.trace(Mtil2.H * Mtil2) - sp.trace(M2.H * M2))
ok_2a = (sp.simplify(sp.re(lhs_2a)) == 0) and (sp.simplify(sp.im(lhs_2a)) == 0)
CHECKS.append(("2a Omega_I invariant under independent U_k, U_{k+b}", ok_2a))
print(f"[{'PASS' if ok_2a else 'FAIL'}] "
      "2a Omega_I invariant under independent U_k, U_{k+b}")

# 2b. Omega_OD invariant under independent per-k diagonal phases,
#     NOT under general unitaries
e1, e2, e3, e4 = sp.symbols("eta1 eta2 eta3 eta4", real=True)
Dk = sp.diag(sp.exp(I * e1), sp.exp(I * e2))
Dkb = sp.diag(sp.exp(I * e3), sp.exp(I * e4))
Mp = Dk.H * M2 * Dkb
od0 = sp.expand_complex(Mp[0, 1] * sp.conjugate(Mp[0, 1])
                        + Mp[1, 0] * sp.conjugate(Mp[1, 0])
                        - M2[0, 1] * sp.conjugate(M2[0, 1])
                        - M2[1, 0] * sp.conjugate(M2[1, 0]))
check("2b Omega_OD invariant under independent diagonal phase gauges"
      " D_k, D_{k+b}", od0)
Mn = sp.Matrix([[sp.Rational(3, 5), sp.Rational(4, 5) * I],
                [sp.Rational(4, 5), sp.Rational(3, 5) * I]])
Rot = sp.Matrix([[sp.cos(sp.Rational(1, 3)), -sp.sin(sp.Rational(1, 3))],
                 [sp.sin(sp.Rational(1, 3)), sp.cos(sp.Rational(1, 3))]])
od_before = sp.expand(Mn[0, 1] * sp.conjugate(Mn[0, 1])
                      + Mn[1, 0] * sp.conjugate(Mn[1, 0]))
MR = Rot.T * Mn * Rot
od_after = sp.expand(MR[0, 1] * sp.conjugate(MR[0, 1])
                     + MR[1, 0] * sp.conjugate(MR[1, 0]))
check_true("2b Omega_OD changes under a general (real) rotation",
           sp.simplify(od_after - od_before) != 0,
           f": delta = {sp.simplify(od_after - od_before)}")

# 2c. diagonal gauge, per-k FIXED-rbar subproblem (the descent quantity within
#     one sweep; the global optimum iterates it with rbar refreshed):
#     d Omega_D/d phi_kn = 0 at fixed rbar.
phk = sp.Symbol("phi_k", real=True)
t1_0, t2_0 = sp.symbols("tau_1 tau_2", real=True)
rb = sp.Symbol("rb", real=True)
w1s, w2s, b1s, b2s = sp.symbols("w_1 w_2 b_1 b_2", positive=True, real=True)
b2s = sp.Symbol("b_2", real=True)  # second neighbor direction is signed
# Omega_D entries for this band at k: sum_j w_j (tau_j - phi + b_j rbar)^2
# (rbar FIXED within the sweep; b_j signed, e.g. b_2 = -b_1 on the star)
OmegaD_2c = w1s * (t1_0 - phk + b1s * rb) ** 2 + w2s * (t2_0 - phk + b2s * rb) ** 2
sol = sp.solve(sp.Eq(sp.diff(OmegaD_2c, phk), 0), phk)[0]
check("2c (per-k, fixed rbar) optimal phase = sum_j w_j(tau_j + b_j rbar)"
      " / sum_j w_j",
      sp.simplify(sol - (w1s * (t1_0 + b1s * rb) + w2s * (t2_0 + b2s * rb))
                  / (w1s + w2s)))

# ---------------------------------------------------------------------------
header("3. Gradient: delta Omega under U_k -> U_k e^{i eps A_k}"
       " (A_k Hermitian generator; eps series to first order)")
# ---------------------------------------------------------------------------
# Consistent Nk=3 ring, 2 wannier functions, neighbors b = +-1 (w=1/2 each),
# overlaps M^{k,+} in {M01, M02, M12} (exact complex rationals) and
# M^{k+b,-b} = (M^{k,+})^dagger. All three A_k symbolic Hermitian (12 params).
# Gauge action U_k -> U_k e^{i eps A_k}:  M^{k,+} -> (I - i eps A_k) M (I + i eps A_{k+1}).
eps = sp.Symbol("epsilon", real=True)


def rc(seed):
    import random
    random.seed(seed)
    return (sp.Rational(random.randint(-9, 9), random.randint(1, 9))
            + I * sp.Rational(random.randint(-9, 9), random.randint(1, 9)))


M01 = sp.Matrix([[rc(1), rc(2)], [rc(3), rc(4)]])
M02 = sp.Matrix([[rc(5), rc(6)], [rc(7), rc(8)]])
M12 = sp.Matrix([[rc(9), rc(10)], [rc(11), rc(12)]])
syms = {}
As = []
for t in ("a0", "a1", "a2"):
    d1_, d2_, xr_, xi_ = sp.symbols(f"{t}_d1 {t}_d2 {t}_xr {t}_xi", real=True)
    syms[t] = (d1_, d2_, xr_, xi_)
    As.append(sp.Matrix([[d1_, xr_ + I * xi_], [xr_ - I * xi_, d2_]]))
A0, A1, A2 = As
E = sp.eye(2)


def Lf(A):  # e^{-i eps A} to first order
    return E - I * eps * A


def Rf(A):  # e^{+i eps A} to first order
    return E + I * eps * A


w = sp.Rational(1, 2)
entries = [
    ("k0+", Lf(A0) * M01 * Rf(A1), +1), ("k0-", Lf(A0) * M02 * Rf(A2), -1),
    ("k1+", Lf(A1) * M12 * Rf(A2), +1), ("k1-", Lf(A1) * M01.H * Rf(A0), -1),
    ("k2+", Lf(A2) * M02.H * Rf(A0), +1), ("k2-", Lf(A2) * M12.H * Rf(A1), -1),
]


def theta_series(Me, n):
    return sp.im(sp.expand(sp.log(Me[n, n]).series(eps, 0, 2).removeO()))


Nk_ring = 3
Om = 0  # NOTE: Om accumulates Nk*Omega (no overall 1/Nk); normalised below
th = {}
for name, Me, bj in entries:
    Om += w * (2 - sum(Me[n, n] * sp.conjugate(Me[n, n]) for n in range(2)))
    for n in range(2):
        th[(name, n)] = theta_series(Me, n)
rbar_eps = [-sp.Rational(1, Nk_ring) * sum(w * bj * th[(name, n)]
                                           for name, _, bj in entries)
            for n in range(2)]
for name, Me, bj in entries:
    for n in range(2):
        Om += w * (th[(name, n)] + bj * rbar_eps[n]) ** 2
dOm = sp.expand(sp.expand_complex(Om)).coeff(eps)
print("d(Nk Omega)/deps|_0 is linear in the 12 generator parameters; record:")
for tg in ("a0", "a1", "a2"):
    for v in syms[tg]:
        print(f"     d/d{v} (Nk*Omega) =", sp.nsimplify(sp.simplify(dOm.coeff(v))))


# w90 wann_domega transcription (non-selective branch, wannierise.F90 2168-2188):
#   ln_tmp(n) = w_b (Im ln M_nn - sheet)          [NOTE: includes w_b]
#   rave = -(1/Nk) sum bk * ln_tmp                 [NO extra wb: it is inside]
#   cr(m,n)  = M(m,n) conj(M(n,n));  crt(m,n) = M(m,n)/M(n,n)
#   rnkb(n)  = b . rbar_n
#   g(m,n) += w_b (cr - cr^dag)/2 + (i/2)(crt ln_tmp + h.c.)
#             + (i/2) w_b (crt rnkb + h.c.)
#   G_k = (4/Nk) g_k   [w90 line 2188: cdodq_loc = cdodq_loc/num_kpts*4]
def cdodq_k(nns, rbar0):
    g = sp.zeros(2, 2)
    for Mk, bj, wb in nns:
        cr = sp.Matrix(2, 2, lambda m, n: Mk[m, n] * sp.conjugate(Mk[n, n]))
        crt = sp.Matrix(2, 2, lambda m, n: Mk[m, n] / Mk[n, n])
        lnt = [wb * sp.arg(Mk[n, n]) for n in range(2)]
        rnk = [bj * rbar0[n] for n in range(2)]
        for m in range(2):
            for n in range(2):
                t1 = wb * sp.Rational(1, 2) * (cr[m, n] - sp.conjugate(cr[n, m]))
                t2 = sp.Rational(1, 2) * I * (crt[m, n] * lnt[n]
                                              + sp.conjugate(crt[n, m] * lnt[m]))
                t3 = sp.Rational(1, 2) * I * wb * (crt[m, n] * rnk[n]
                                                   + sp.conjugate(crt[n, m] * rnk[m]))
                g[m, n] += t1 + t2 + t3
    return g


rbar0 = [sp.simplify(sp.expand_complex(rb_.subs(eps, 0))) for rb_ in rbar_eps]
g0 = cdodq_k([(M01, +1, w), (M02, -1, w)], rbar0)
g1 = cdodq_k([(M12, +1, w), (M01.H, -1, w)], rbar0)
g2 = cdodq_k([(M02.H, +1, w), (M12.H, -1, w)], rbar0)


def trAX(Gk, Ak):
    X = 2 * I * (Gk - Gk.H)
    return sp.expand(sp.trace(Ak * X))


check("3a d(Nk Omega)/deps|_0 == sum_k Tr[ A_k . 2i (g_k - g_k^dag) ]"
      "  (g_k = w90 cdodq terms BEFORE the final 4/Nk factor)",
      sp.expand_complex(trAX(g0, A0) + trAX(g1, A1) + trAX(g2, A2) - dOm))
# same identity for the NORMALISED Omega (1/Nk included) against w90's FINAL
# cdodq scaling G_k = (4/Nk) g_k:
pred_norm = 0
for gk, Ak in ((g0, A0), (g1, A1), (g2, A2)):
    Gk = 4 * gk / Nk_ring
    pred_norm += sp.expand(sp.trace(Ak * (sp.I / 2 * (Gk - Gk.H))))
check("3b d(Omega)/deps|_0 == sum_k Tr[ A_k . (i/2)(G_k - G_k^dag) ],"
      "  G_k = w90 cdodq INCLUDING final 4/Nk",
      sp.expand_complex(pred_norm - dOm / Nk_ring))
print("   descent step (normalised): A_k = -(i/2)(G_k - G_k^dag);"
      " update U_k <- U_k e^{i eps A_k}")

# ---------------------------------------------------------------------------
header("4. Continuum limit: why the Mmn form is the right discretisation")
# ---------------------------------------------------------------------------
b, alpha, beta, gamma = sp.symbols("b alpha beta gamma", real=True)
Mcont = 1 + b * (I * alpha) + b**2 / 2 * (beta + I * gamma)
lnM = (Mcont - 1) - (Mcont - 1) ** 2 / 2
theta_c = sp.im(sp.expand_complex(lnM))
omod = sp.expand_complex(1 - Mcont * sp.conjugate(Mcont))
check("4a 1-|M_nn|^2 = -(alpha^2+beta) b^2 + O(b^3)",
      sp.series(omod + (alpha**2 + beta) * b**2, b, 0, 3).removeO())
check("4b theta~ = alpha b + gamma b^2/2 + O(b^3)",
      sp.series(theta_c - (alpha * b + gamma * b**2 / 2), b, 0, 3).removeO())

# 4c. EXECUTED identification of alpha, beta on an explicit normalised
#     family u(b) = (1, f(b))/||.|| with f = fr(b) + i fi(b) parameterised
#     by REAL exact-rational Taylor coefficients (no conjugate(f(b))
#     derivatives; conj(f) = fr - i fi handled explicitly).
fr0, fr1, fr2 = sp.Rational(1, 3), sp.Rational(-2, 7), sp.Rational(1, 2)
fi0, fi1, fi2 = sp.Rational(1, 5), sp.Rational(3, 4), sp.Rational(-2, 9)
fr = fr0 + fr1 * b + fr2 * b**2 / 2
fi = fi0 + fi1 * b + fi2 * b**2 / 2
absf2 = fr**2 + fi**2
nrm = sp.sqrt(1 + absf2)
ub = sp.Matrix([1, fr + I * fi]) / nrm
u0 = ub.subs(b, 0)
up = sp.diff(ub, b)
upp = sp.diff(ub, b, 2)
norm_deriv = sp.diff((ub.H * ub)[0, 0], b)
check("4c(i) normalisation: d/db <u(b)|u(b)> = 0",
      sp.simplify(norm_deriv))
ident = sp.simplify(sp.expand_complex(
    (up.H * up)[0, 0].subs(b, 0) + sp.re((ub.H * upp)[0, 0].subs(b, 0))))
check("4c(ii) <u'|u'> + Re<u|u''> = 0  (=> beta = -<u'|u'>)", ident)
Mb = sp.series((u0.H * ub)[0, 0], b, 0, 3).removeO()   # <u(0)|u(b)>, u0=(1,0)
alpha_c = sp.im((ub.H * up)[0, 0].subs(b, 0))
target = b**2 * (sp.re((up.H * up)[0, 0].subs(b, 0)) - alpha_c**2)
check("4c(iii) 1-|<u_0|u_b>|^2 = b^2(<u'|u'>-alpha^2) + O(b^3)"
      "  (alpha = -<r> in the periodic gauge, MV97)",
      sp.series(sp.expand_complex(1 - Mb * sp.conjugate(Mb)) - target,
                b, 0, 3).removeO())

# 4d. FINITE-MESH FINDING, EXECUTED: R-space sums built from M~ phases alone
#     do NOT reproduce the Mmn moments on a finite mesh (Nk=3 ring, generic
#     phases phi=(0, 1/3, 7/5)).
phis_c = [sp.Integer(0), sp.Rational(1, 3), sp.Rational(7, 5)]


def Cph(R):
    return sum(sp.exp(2 * sp.pi * I * k * R / 3) * sp.exp(I * phis_c[k])
               for k in range(3)) / 3


# (DFT Parseval makes sum_R |C(R)|^2 == 1 identically -- the failure is in
#  the MOMENTS, not the norm:)
lin1 = sp.simplify(sp.expand_complex(
    Cph(1) * sp.conjugate(Cph(1)) - Cph(2) * sp.conjugate(Cph(2))))
check_true("4d(i) counterexample: e^{ikR}-phase construction, linear moment"
           " != Mmn rbar (=0 on the ring)", lin1 != 0, f": {lin1}")


def Cth(R):
    expr = 0
    for k in range(3):
        for s, sgn in (("+", 1), ("-", -1)):
            th = phis_c[(k + sgn) % 3] - phis_c[k]
            expr += (sp.Rational(1, 2) * sp.exp(I * th)
                     * sp.exp(sgn * 2 * sp.pi * I * R / 3))
    return sp.expand_complex(expr / 3)


linC = sp.simplify(sp.expand_complex(
    Cth(1) * sp.conjugate(Cth(1)) - Cth(2) * sp.conjugate(Cth(2))))
check_true("4d(ii) counterexample: R-space linear moment != Mmn rbar (=0"
           " on the ring; M-phase + b-phase construction)", linC != 0,
           f": {linC}")
# Resolution: the Mmn/theta^2 form IS the discrete definition (MV97; w90
# r2ave 1785-1798 computes all moments from Mmn). Rdeg-weighted R-space
# reporting equals it only in the continuum limit; consistency is validated
# numerically (story-008 differential test), not by a discrete identity.
# NOTE: this finding must be adjudicated against architecture-mlwf ADR-004
# (which mandates Rdeg-weighted R-space reporting of Omega_I) at the story
# gate -- see mlwf_domega_derivation.md, Finding 1.

# ---------------------------------------------------------------------------
header("5. Analytic oracles")
# ---------------------------------------------------------------------------
# Winding-1 two-band chain: H(k) = cos k sig_z + sin k sig_x;
# eigenvectors u_+k = (cos(k/2), sin(k/2)), u_-k = (-sin(k/2), cos(k/2)).
def chain_U(Nkv):
    ks = [sp.pi * 2 * n / Nkv for n in range(Nkv)]

    def evecs(k):
        t = sp.atan2(sp.sin(k), sp.cos(k))
        return sp.Matrix.hstack(sp.Matrix([sp.cos(t / 2), sp.sin(t / 2)]),
                                sp.Matrix([-sp.sin(t / 2), sp.cos(t / 2)]))
    return ks, [evecs(k) for k in ks]


def omega_parts(U_k, Nkv):
    """Omega_I, Omega_D, Omega_OD with principal-branch theta~ (no w90
    sheet/csheet tracking): Omega_D values are branch-dependent; the
    gauge-fixed checks below use analytic phases instead."""
    bv = sp.pi * 2 / Nkv
    wb = 1 / (2 * bv**2)
    omI = omD = omOD = 0
    thetas = []
    for k in range(Nkv):
        for s, sgn in (("+", 1), ("-", -1)):
            kp = (k + 1) % Nkv if s == "+" else (k - 1) % Nkv
            Mt = U_k[k].H * U_k[kp]
            omI += wb * (Mt.shape[1] - sp.re(sp.trace(Mt.H * Mt)))
            omOD += wb * sum(sp.re(Mt[m, n] * sp.conjugate(Mt[m, n]))
                             for m in range(Mt.shape[0])
                             for n in range(Mt.shape[0]) if m != n)
            thetas.append([sp.arg(Mt[n, n]) for n in range(Mt.shape[0])])
    nw = U_k[0].shape[1]
    for n in range(nw):
        rb = -sp.Rational(1, Nkv) * sum(
            wb * thetas[2 * k + j][n] * (1 if j == 0 else -1)
            for k in range(Nkv) for j in (0, 1))
        for k in range(Nkv):
            for j in (0, 1):
                sgn = 1 if j == 0 else -1
                omD += wb * (thetas[2 * k + j][n] + sgn * rb) ** 2
    return (sp.simplify(omI / Nkv), sp.simplify(omD / Nkv),
            sp.simplify(omOD / Nkv))


def closed_I(Nkv):
    return sp.simplify(Nkv**2 / (4 * sp.pi**2) * sp.sin(sp.pi / Nkv) ** 2)

# xy-plane winding model (a two-site chain): H(k) = cos k sig_x + sin k sig_y,
# u_+-k = (1, +-e^{i phi_k})/sqrt(2), phi_k = k. Composite span is k-
# independent (rank 2), so its full-space optimum is atomic (see 5c); the
# isolated single band is the nontrivial case.
def model_U(Nkv):
    ks_ = [sp.pi * 2 * n / Nkv for n in range(Nkv)]
    return [sp.Matrix.hstack(sp.Matrix([1, sp.exp(I * ks_[k])]) / sp.sqrt(2),
                             sp.Matrix([1, -sp.exp(I * ks_[k])]) / sp.sqrt(2))
            for k in range(Nkv)]


# 5a. single isolated band {u_+}: M_nn^{k,+} = e^{+i b/2} cos(b/2),
#     M_nn^{k,-} = e^{-i b/2} cos(b/2): theta~_- = -theta~_+ exactly, so
#     rbar = -1/2 and Omega_D = 0 ALREADY in the identity gauge (the gauge
#     structure is symmetric); Omega = Omega_I closed form.
for Nkv in (4, 8):
    bvv = sp.pi * 2 / Nkv
    wbb = 1 / (2 * bvv**2)
    Um = model_U(Nkv)
    omI1 = 0
    th = []
    for k in range(Nkv):
        for s, sgn in (("+", 1), ("-", -1)):
            kp = (k + 1) % Nkv if s == "+" else (k - 1) % Nkv
            M11 = (Um[k].H * Um[kp])[0, 0]
            omI1 += wbb * (1 - M11 * sp.conjugate(M11))
            th.append((sgn, sp.arg(sp.expand_complex(M11))))
    check(f"5a(Nk={Nkv}) theta~_+ = b/2, theta~_- = -b/2 (identity gauge)",
          len({(s, sp.simplify(tv)) for s, tv in th}
              - {(1, bvv / 2), (-1, -bvv / 2)}))
    rb = -sp.Rational(1, Nkv) * sum(wbb * s * bvv * tv for s, tv in th)
    check_true(f"5a(Nk={Nkv}) rbar = -1/2 (band centre at the second site)",
               sp.simplify(sp.expand_complex(rb - sp.Rational(1, 2))) == 0
               or sp.nsimplify(sp.N(rb, 30)) == -sp.Rational(1, 2),
               f": {sp.simplify(rb)}")
    omD1 = sum(wbb * (tv + s * rb * bvv) ** 2 for s, tv in th)
    check(f"5a(Nk={Nkv}) Omega_D = sum w(theta~ + b.rbar)^2 == 0",
          sp.simplify(omD1 / Nkv))
    check(f"5a(Nk={Nkv}) Omega_I == Nk^2/(4 pi^2) sin^2(pi/Nk)"
          "  =>  Omega = Omega_I",
          sp.expand_complex(omI1 / Nkv - closed_I(Nkv)))

# executed generic-Nk limit: the closed form itself, symbolically in Nk
Nsg = sp.Symbol("N", positive=True)
check("5a(symbolic) lim_{Nk->inf} Nk^2/(4 pi^2) sin^2(pi/Nk) = 1/4",
      sp.limit(Nsg**2 / (4 * sp.pi**2) * sp.sin(sp.pi / Nsg) ** 2,
               Nsg, sp.oo) - sp.Rational(1, 4))

# 5b. full two-band space: Omega_I = 0 exactly (M unitary); Omega_OD closed
#     form 2 Nk^2/(4 pi^2) sin^2(pi/Nk) (branch-independent, no args).
for Nkv in (4, 8):
    ks, U_k = chain_U(Nkv)
    omI, omD, omOD = omega_parts(U_k, Nkv)
    check(f"5b(Nk={Nkv}) full two-band space: Omega_I == 0",
          sp.expand_complex(omI))
    check(f"5b(Nk={Nkv}) Omega_OD == 2 Nk^2/(4 pi^2) sin^2(pi/Nk)",
          omOD - 2 * closed_I(Nkv))
    print(f"    Omega_D(principal branch, sheet-free) = {sp.N(omD, 8)}"
          " -- branch-dependent, not a claimed equation")
    # executed FULL-U optimal gauge: V_k = U_evecs(k)^dagger puts the
    # composite in the atomic-orbital basis; every overlap becomes I exactly
    # (numeric-exact products; unitarity of each U_k is structural).
    wbv = 1 / (2 * (sp.pi * 2 / Nkv) ** 2)
    omOD_g = 0
    for k in range(Nkv):
        Uk = U_k[k]
        assert sp.simplify(Uk.H * Uk) == sp.eye(2)
        for s in ("+", "-"):
            kp = (k + 1) % Nkv if s == "+" else (k - 1) % Nkv
            Mt = Uk * (Uk.H * U_k[kp]) * U_k[kp].H
            omOD_g += wbv * sum(sp.re(Mt[m, n] * sp.conjugate(Mt[m, n]))
                                for m in range(2) for n in range(2) if m != n)
    check(f"5b(Nk={Nkv}) full-U gauge V_k=U_k^dag: M~ = I exactly"
          " (Omega_OD = 0, atomic basis)",
          sp.simplify(sp.expand_complex(omOD_g / Nkv)))

# 5c. two-orbital model u_+- = (e_1 +- e^{i phi_k} e_2)/sqrt(2), phi_k = k:
#     both neighbors computed; theta~_+- = +- b/2, rbar = -1/2 per band =>
#     Omega_D = 0 in the identity gauge; Omega_OD = (1-cos b)/b^2;
#     optimal gauge is the FULL unitary V_k = U_k^dag (composite span is
#     k-independent => atomic-orbital basis, every overlap I): Omega = 0.
Nkv = 8
ks = [sp.pi * 2 * n / Nkv for n in range(Nkv)]
bv = sp.pi * 2 / Nkv
wb = 1 / (2 * bv**2)


def umodel(k, sign):
    return sp.Matrix([1, sign * sp.exp(I * ks[k])]) / sp.sqrt(2)


omI = omD = omOD = 0
thetas = []
for k in range(Nkv):
    for s, sgn in (("+", 1), ("-", -1)):
        kp = (k + sgn) % Nkv
        Up = sp.Matrix.hstack(umodel(k, 1), umodel(k, -1))
        Upp = sp.Matrix.hstack(umodel(kp, 1), umodel(kp, -1))
        Mt = Up.H * Upp
        omI += wb * (2 - sp.re(sp.trace(Mt.H * Mt)))
        omOD += wb * sum(sp.re(Mt[m, n] * sp.conjugate(Mt[m, n]))
                         for m in range(2) for n in range(2) if m != n)
        thetas.append([sp.arg(sp.expand_complex(Mt[n, n])) for n in range(2)])
for n in range(2):
    rb = -sp.Rational(1, Nkv) * sum(wb * (bv if j == 0 else -bv) * thetas[2 * k + j][n]
                                    for k in range(Nkv) for j in (0, 1))
    for k in range(Nkv):
        for j, sgn in ((0, 1), (1, -1)):
            omD += wb * (thetas[2 * k + j][n] + sgn * rb * bv) ** 2
check("5c two-orbital model Omega_I == 0 (full 2-orbital space, M unitary)",
      sp.expand_complex(omI / Nkv))
check("5c Omega_OD == (1-cos b)/b^2 = 2 Nk^2/(4 pi^2) sin^2(pi/Nk)",
      omOD / Nkv - (1 - sp.cos(bv)) / bv**2)
check("5c identity-gauge Omega_D == 0 (theta~ = +-b/2, rbar=-1/2 per band)",
      omD / Nkv)
# executed optimal gauge: |M_mn| is invariant under diagonal phases (2b),
# so the composite-space optimum needs the full unitary. The composite span
# is k-independent here: V_k = U_k^dag is the atomic-orbital basis and every
# overlap becomes I exactly (Omega = 0: localized site Wannier).
omOD_g = 0
thg = []
for k in range(Nkv):
    Uk_ = sp.Matrix.hstack(umodel(k, 1), umodel(k, -1))
    for s, sgn in (("+", 1), ("-", -1)):
        kp = (k + sgn) % Nkv
        Ukp_ = sp.Matrix.hstack(umodel(kp, 1), umodel(kp, -1))
        Mt = Uk_ * (Uk_.H * Ukp_) * Ukp_.H
        omOD_g += wb * sum(sp.re(Mt[m, n] * sp.conjugate(Mt[m, n]))
                           for m in range(2) for n in range(2) if m != n)
        thg.append([sp.arg(Mt[n, n]) for n in range(2)])
omD_g = 0
for n in range(2):
    rbg = -sp.Rational(1, Nkv) * sum(wb * (bv if j == 0 else -bv) * thg[2 * k + j][n]
                                     for k in range(Nkv) for j in (0, 1))
    for k in range(Nkv):
        for j, sgn in ((0, 1), (1, -1)):
            omD_g += wb * (thg[2 * k + j][n] + sgn * rbg * bv) ** 2
check("5c optimal gauge V_k=U_k^dag (atomic basis): Omega_OD == 0",
      omOD_g / Nkv)
check("5c optimal gauge: Omega_D == 0  =>  Omega = 0 (atomic Wannier)",
      omD_g / Nkv)

# ---------------------------------------------------------------------------
header("SUMMARY")
# ---------------------------------------------------------------------------
nfail = sum(1 for _, ok in CHECKS if not ok)
for name, ok in CHECKS:
    if not ok:
        print("FAILED:", name)
print(f"\n{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
sys.exit(1 if nfail else 0)
